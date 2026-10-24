import sys
import json
import tempfile
import contextlib
import io
from unittest.mock import Mock, patch
from types import SimpleNamespace
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from retrieval import build_memories, evidence_recall, pack_context, rank_scores
from run_answers import request_for, save_answers
from experiment_paths import load_experiment, load_answers, experiment_metadata
from run_answers import main as answers_main
from prepare_retrieval import main as prepare_main


def experiment(root, model='gpt-5.4-mini', budget=512):
    folder = root / 'outputs' / model / f'{budget}_tokens'
    inputs = folder / 'inputs'
    inputs.mkdir(parents=True)
    (folder / 'outputs').mkdir()
    samples = [{
        'sample_id': f'conv-{i // 2}', 'qa_index': i, 'category': 4,
        'question': f'Question {i}?', 'answer': 'PRIVATE REFERENCE', 'evidence': ['D1:1'],
    } for i in range(20)]
    (root / 'src/data').mkdir(parents=True, exist_ok=True)
    (root / 'src/data/selected_qa_20.json').write_text(json.dumps(samples))
    records = [{
        'sample_id': q['sample_id'], 'qa_index': q['qa_index'], 'strategy': strategy,
        'context': f'Memory {i * 3 + j}', 'retrieved_ids': ['D1:1'], 'memory_tokens': 10,
    } for i, q in enumerate(samples) for j, strategy in enumerate(('recency', 'bm25', 'semantic'))]
    (inputs / 'retrieval.json').write_text(json.dumps(records))
    return folder, records


class RetrievalContractTests(unittest.TestCase):
    def test_budget_isolation_accepts_1024_and_rejects_oversized_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, _ = experiment(root, budget=512)
            large, records = experiment(root, budget=1024)
            frozen = (small / 'inputs/retrieval.json').read_bytes()
            records[0]['memory_tokens'] = 900
            (large / 'inputs/retrieval.json').write_text(json.dumps(records))
            folder, rows = load_experiment('gpt-5.4-mini', root, budget=1024)
            self.assertEqual(folder, large)
            self.assertEqual(rows[0]['memory_tokens'], 900)
            self.assertEqual(load_answers(folder, rows), [])
            self.assertEqual((small / 'inputs/retrieval.json').read_bytes(), frozen)
            records[0]['memory_tokens'] = 1025
            (large / 'inputs/retrieval.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'budget'):
                load_experiment('gpt-5.4-mini', root, budget=1024)
            with self.assertRaisesRegex(ValueError, 'budget'):
                load_experiment('gpt-5.4-mini', root, budget=2048)

    def test_different_budget_answer_file_blocks_resume_before_api_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root, budget=1024)
            _, rows = load_experiment('gpt-5.4-mini', root, budget=1024)
            metadata = experiment_metadata('gpt-5.4-mini', rows, 'text-embedding-3-small', budget=512)
            (folder / 'outputs/answer.json').write_text(json.dumps({'metadata': metadata, 'answers': []}))
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini', '--budget', '1024']):
                with self.assertRaisesRegex(ValueError, 'budget'):
                    answers_main()

    def test_retriever_packs_the_requested_budget_with_identical_ranking(self):
        from retrieval import ConversationRetriever
        memories = [{'dia_id': str(i), 'formatted': 'word ' * 60} for i in range(4)]
        encoder = Mock()
        encoder.encode.return_value = [[1., 0.]] * len(memories)
        retriever = ConversationRetriever(memories, encoder, count_tokens=len)
        small = retriever.retrieve('word', 'recency', budget=512)
        large = retriever.retrieve('word', 'recency', budget=1024)
        self.assertLessEqual(small['memory_tokens'], 512)
        self.assertGreater(large['memory_tokens'], 512)
        self.assertLessEqual(large['memory_tokens'], 1024)
        self.assertEqual(small['retrieved_ids'], ['3'])
        self.assertEqual(large['retrieved_ids'], ['1', '2', '3'])
        encoder.encode.assert_called_once()

    def test_retrieval_latency_includes_query_and_packing_but_excludes_index_setup(self):
        from retrieval import ConversationRetriever
        memories = [{'dia_id': str(i), 'formatted': text}
                    for i, text in enumerate(['old park', 'new job'])]
        def encode(texts):
            if isinstance(texts, str):
                self.assertEqual(clock.call_count, 3)
                return [1., 0.]
            self.assertEqual(clock.call_count, 0)
            return [[1., 0.]] * len(texts)
        def count_tokens(text):
            self.assertIn(clock.call_count, (1, 3))
            return len(text)
        encoder = Mock()
        encoder.encode.side_effect = encode
        with patch('retrieval.perf_counter', side_effect=[1., 1.125678, 2., 2.456789]) as clock:
            retriever = ConversationRetriever(memories, encoder, count_tokens)
            clock.assert_not_called()
            recent = retriever.retrieve('park', 'recency')
            semantic = retriever.retrieve('park', 'semantic')
        self.assertEqual(recent['retrieval_latency_ms'], 125.68)
        self.assertEqual(semantic['retrieval_latency_ms'], 456.79)

    def test_latency_validation_accepts_legacy_rows_but_rejects_invalid_or_partial_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, records = experiment(root)
            path = folder / 'inputs/retrieval.json'
            load_experiment('gpt-5.4-mini', root)
            records[0]['retrieval_latency_ms'] = 2.5
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'latency'):
                load_experiment('gpt-5.4-mini', root)
            for invalid in (None, True, -1, '2', float('nan'), float('inf')):
                for row in records:
                    row['retrieval_latency_ms'] = invalid
                path.write_text(json.dumps(records))
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'latency'):
                    load_experiment('gpt-5.4-mini', root)
            for row in records:
                row['retrieval_latency_ms'] = 2.5
            path.write_text(json.dumps(records))
            _, rows = load_experiment('gpt-5.4-mini', root)
            self.assertTrue(all(r['retrieval_latency_ms'] == 2.5 for r in rows))

    def test_partial_legacy_answers_cannot_resume_into_a_timed_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            _, rows = load_experiment('gpt-5.4-mini', root)
            first = rows[0]
            saved = {k: first[k] for k in ('sample_id', 'qa_index', 'strategy', 'question', 'evidence')}
            saved.update(reference_answer=first['answer'], llm_answer='Saved legacy answer', correct=1)
            save_answers(folder / 'outputs/answer.json', [saved],
                         experiment_metadata('gpt-5.4-mini', rows, 'text-embedding-3-small'))
            before = (folder / 'outputs/answer.json').read_bytes()
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini']):
                with self.assertRaisesRegex(SystemExit, 'latency'):
                    answers_main()
            self.assertEqual((folder / 'outputs/answer.json').read_bytes(), before)

    def test_metadata_counts_questions_once_across_the_three_strategies(self):
        rows = [{'sample_id': 'conv-1', 'qa_index': index, 'category': category,
                 'strategy': strategy}
                for index, category in enumerate([1] * 5 + [2] * 5 + [4] * 10)
                for strategy in ('recency', 'bm25', 'semantic')]
        metadata = experiment_metadata('gpt-5-nano', rows, 'text-embedding-3-small')
        self.assertEqual(set(metadata), {'llm_model', 'embedding_model', 'question_count',
                                        'question_categories', 'retrieval_strategies', 'memory_budget_tokens'})
        self.assertEqual(metadata['question_count'], 20)
        self.assertEqual({k: v['count'] for k, v in metadata['question_categories'].items()},
                         {'multi_hop': 5, 'temporal': 5, 'single_hop': 10})
        self.assertTrue(all(set(v) == {'count', 'description'} and v['description']
                            for v in metadata['question_categories'].values()))
        self.assertEqual(metadata['retrieval_strategies'], ['recency', 'bm25', 'semantic'])
        self.assertEqual(metadata['memory_budget_tokens'], 512)

    def test_dry_run_does_not_import_api_client_or_write_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini', '--dry-run']), \
                    contextlib.redirect_stdout(io.StringIO()):
                answers_main()
            self.assertFalse((folder / 'outputs/answer.json').exists())

    def test_real_question_limit_and_resume_need_no_synthetic_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, records = experiment(root, budget=1024)
            for record in records:
                record['retrieval_latency_ms'] = 12.34
            (folder / 'inputs/retrieval.json').write_text(json.dumps(records))
            (folder / 'outputs').rmdir()  # First real answer must create its output directory.
            response = SimpleNamespace(
                id='test-response', model='gpt-5.4-mini', status='completed',
                output_text='Test answer', usage=None,
                model_dump=lambda **kwargs: {'id': 'test-response', 'status': 'completed'},
            )
            create = Mock(return_value=response)
            constructor = Mock(return_value=SimpleNamespace(responses=SimpleNamespace(create=create)))
            with patch('run_answers.perf_counter', side_effect=[1., 2.234567, 3., 5.345678]), \
                    patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=constructor),
                                             'dotenv': SimpleNamespace(load_dotenv=lambda *a, **k: None)}), \
                    patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only', 'OPENAI_TEXT_EMBEDDING_MODEL': 'text-embedding-3-small'}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini', '--limit', '1', '--budget', '1024']), \
                    contextlib.redirect_stdout(io.StringIO()):
                answers_main()
                first_saved = json.loads((folder / 'outputs/answer.json').read_text())
                first_saved['answers'][0]['correct'] = 1
                (folder / 'outputs/answer.json').write_text(json.dumps(first_saved, indent=2))
                with patch.dict('os.environ', {'OPENAI_TEXT_EMBEDDING_MODEL': 'different-current-setting'}):
                    answers_main()
            document = json.loads((folder / 'outputs/answer.json').read_text())
            self.assertEqual(set(document), {'metadata', 'answers'})
            self.assertEqual(document['metadata']['llm_model'], 'gpt-5.4-mini')
            self.assertEqual(document['metadata']['memory_budget_tokens'], 1024)
            self.assertEqual(document['metadata']['latency_measurement'], 'same_run')
            self.assertEqual(document['metadata'], first_saved['metadata'])
            self.assertEqual(document['metadata']['embedding_model'], 'text-embedding-3-small')
            self.assertEqual(document['metadata']['question_count'], 20)
            self.assertEqual(document['metadata']['question_categories']['single_hop']['count'], 20)
            saved = document['answers']
            self.assertEqual(len(saved), 2)
            self.assertEqual([r['answer_api_latency_ms'] for r in saved], [1234.57, 2345.68])
            self.assertEqual([r['retrieval_latency_ms'] for r in saved], [12.34, 12.34])
            self.assertEqual(saved[0], first_saved['answers'][0])
            self.assertIn('\n    {', (folder / 'outputs/answer.json').read_text())
            self.assertEqual(create.call_count, 2)
            self.assertEqual([r['strategy'] for r in saved], [r['strategy'] for r in records[:2]])
            self.assertEqual(set(saved[0]), {'sample_id', 'qa_index', 'strategy', 'question',
                                              'reference_answer', 'evidence', 'llm_answer', 'correct', 'retrieval_latency_ms', 'answer_api_latency_ms'})
            self.assertEqual([r['llm_answer'] for r in saved], ['Test answer'] * 2)
            self.assertEqual(saved[0]['evidence'], ['D1:1'])
            self.assertEqual(saved[0]['reference_answer'], 'PRIVATE REFERENCE')
            self.assertEqual(saved[0]['correct'], 1)
            self.assertIsNone(saved[1]['correct'])
            self.assertFalse((folder / 'answer_review.json').exists())
            self.assertIn(records[1]['context'], create.call_args.kwargs['input'])
            self.assertNotIn('PRIVATE REFERENCE', str(create.call_args_list))
            self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)
            self.assertFalse((folder / 'pilot_answers.json').exists())

    def test_failed_json_replacement_keeps_previous_complete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'answer.json'
            previous = [{'llm_answer': 'Already saved'}]
            metadata = {'llm_model': 'gpt-5-nano', 'embedding_model': 'text-embedding-3-small'}
            original = {'metadata': metadata, 'answers': previous}
            path.write_text(json.dumps(original))
            with patch.object(Path, 'replace', side_effect=OSError('simulated interruption')):
                with self.assertRaises(OSError):
                    save_answers(path, previous + [{'llm_answer': 'New answer'}], metadata)
            self.assertEqual(json.loads(path.read_text()), original)

    def test_api_failure_stops_without_automatic_retry_or_fake_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            create = Mock(side_effect=RuntimeError('simulated failure'))
            constructor = Mock(return_value=SimpleNamespace(responses=SimpleNamespace(create=create)))
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=constructor),
                                             'dotenv': SimpleNamespace(load_dotenv=lambda *a, **k: None)}), \
                    patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only', 'OPENAI_TEXT_EMBEDDING_MODEL': 'text-embedding-3-small'}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini']), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, 'No automatic retry'):
                    answers_main()
            self.assertEqual(create.call_count, 1)
            self.assertFalse((folder / 'outputs/answer.json').exists())

    def test_interrupted_json_save_blocks_further_calls_until_inspected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            (folder / 'outputs/answer.json.tmp').write_text('[{"llm_answer": "saved to temporary file"}]')
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini']):
                with self.assertRaisesRegex(SystemExit, 'Interrupted JSON save'):
                    answers_main()
            self.assertFalse((folder / 'outputs/answer.json').exists())

    def test_retrieval_preparation_preserves_existing_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / 'retrieval.json'
            frozen.write_text('original frozen context')
            with patch.object(sys, 'argv', ['prepare_retrieval.py', '--output', str(frozen)]):
                with self.assertRaisesRegex(SystemExit, 'preserving'):
                    prepare_main()
            self.assertEqual(frozen.read_text(), 'original frozen context')

    def test_retrieval_preparation_does_not_replace_inputs_when_outputs_contain_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'outputs/gpt-5-nano/512_tokens/outputs'
            output.mkdir(parents=True)
            (output / 'answer.json').write_text('existing answers')
            with patch('prepare_retrieval.ROOT', root), \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['prepare_retrieval.py']):
                with self.assertRaisesRegex(SystemExit, 'Existing answers'):
                    prepare_main()
            self.assertEqual((output / 'answer.json').read_text(), 'existing answers')

    def test_model_inputs_and_results_resolve_to_separate_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ('gpt-5-nano', 'gpt-5.4-mini'):
                experiment(root, model)
            nano, _ = load_experiment('gpt-5-nano', root=root)
            mini, _ = load_experiment('gpt-5.4-mini', root=root)
            self.assertNotEqual(nano / 'outputs/answer.json', mini / 'outputs/answer.json')
            self.assertEqual((nano / 'inputs/retrieval.json').read_bytes(),
                             (mini / 'inputs/retrieval.json').read_bytes())
            changed = json.loads((mini / 'inputs/retrieval.json').read_text())
            changed[1] = changed[0]
            (mini / 'inputs/retrieval.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, '60 unique'):
                load_experiment('gpt-5.4-mini', root=root)
            load_experiment('gpt-5-nano', root=root)

    def test_unknown_model_and_saved_response_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            with self.assertRaisesRegex(ValueError, 'model'):
                load_experiment('other-model', root)
            _, rows = load_experiment('gpt-5.4-mini', root)
            saved = {k: rows[0][k] for k in ('sample_id', 'qa_index', 'strategy', 'question', 'answer', 'evidence')}
            saved['reference_answer'] = saved.pop('answer')
            saved.update(llm_answer='Yes', correct=None)
            for invalid in ([saved, saved], [{**saved, 'qa_index': 99}], [{**saved, 'llm_answer': ' '}], [{**saved, 'evidence': ['D9:9']}],
                            [{**saved, 'question': 'Different?'}], [{**saved, 'reference_answer': 'Incorrect reference'}],
                            [{**saved, 'correct': True}]):
                (folder / 'outputs/answer.json').write_text(json.dumps({'metadata': experiment_metadata('gpt-5.4-mini', rows, 'text-embedding-3-small'), 'answers': invalid}))
                with self.assertRaises(ValueError):
                    load_answers(folder, rows)
            (folder / 'outputs/answer.json').write_text(json.dumps({'metadata': experiment_metadata('gpt-5.4-mini', rows, 'text-embedding-3-small'), 'answers': [saved]}))
            with self.assertRaisesRegex(ValueError, '60'):
                load_answers(folder, rows, complete=True)

    def test_answer_file_with_different_model_blocks_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            (folder / 'outputs/answer.json').write_text(json.dumps({'metadata': {'llm_model': 'gpt-5-nano'}, 'answers': []}))
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini']):
                with self.assertRaisesRegex(ValueError, 'model'):
                    answers_main()

    def test_incomplete_response_is_not_saved_as_a_completed_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = experiment(root)
            create = Mock(return_value=SimpleNamespace(status='incomplete', output_text='Partial'))
            constructor = Mock(return_value=SimpleNamespace(responses=SimpleNamespace(create=create)))
            with patch('run_answers.load_experiment', side_effect=lambda model, budget=512: load_experiment(model, root, budget=budget)), \
                    patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=constructor),
                                             'dotenv': SimpleNamespace(load_dotenv=lambda *a, **k: None)}), \
                    patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only', 'OPENAI_TEXT_EMBEDDING_MODEL': 'text-embedding-3-small'}), \
                    patch.object(sys, 'argv', ['run_answers.py', '--model', 'gpt-5.4-mini']), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, 'Incomplete or empty'):
                    answers_main()
            self.assertEqual(create.call_count, 1)
            self.assertFalse((folder / 'outputs/answer.json').exists())

    def test_answer_request_contains_only_question_and_context_not_evaluation_labels(self):
        request = request_for('When?', '[D1 | 8 May 2023 | A]\nYesterday.')
        self.assertEqual(request['model'], 'gpt-5-nano')
        self.assertFalse(request['store'])
        self.assertIn('Yesterday.', request['input'])
        self.assertNotIn('gold_answer', str(request))
        self.assertNotIn('temperature', request)

    def test_memory_uses_original_text_dates_and_numeric_session_order_only(self):
        sample = {'conversation': {
            'session_10': [{'dia_id': 'D10:1', 'speaker': 'B', 'text': 'Later'}],
            'session_10_date_time': '10 May 2023',
            'session_2': [{'dia_id': 'D2:1', 'speaker': 'A', 'text': 'Yesterday',
                           'blip_caption': 'SECRET CAPTION', 'img_url': ['SECRET URL']}],
            'session_2_date_time': '2 May 2023',
        }, 'qa': [{'answer': 'SECRET ANSWER'}], 'session_summary': 'SECRET SUMMARY'}
        memories = build_memories(sample)
        self.assertEqual([m['dia_id'] for m in memories], ['D2:1', 'D10:1'])
        self.assertIn('2 May 2023', memories[0]['formatted'])
        self.assertIn('Yesterday', memories[0]['formatted'])
        self.assertNotIn('SECRET', str(memories))

    def test_budget_counts_formatted_context_skips_long_turns_and_orders_by_time(self):
        memories = [{'dia_id': str(i), 'order': i, 'formatted': text}
                    for i, text in enumerate(['old', 'x' * 100, 'new'])]
        result = pack_context(memories, [2, 1, 0], count_tokens=len, budget=8)
        self.assertEqual(result['context'], 'old\n\nnew')
        self.assertEqual(result['retrieved_ids'], ['0', '2'])
        self.assertEqual(result['memory_tokens'], 8)
        self.assertEqual(evidence_recall(['0', '0', '1'], ['0', '0']), 0.5)
        self.assertEqual(evidence_recall(['0'], []), 0)

    def test_evidence_recall_requires_valid_gold_and_score_ties_are_deterministic(self):
        self.assertEqual(rank_scores([0.1, 0.8, 0.8]), [2, 1, 0])
        with self.assertRaises(ValueError):
            evidence_recall([], ['0'])
        with self.assertRaises(ValueError):
            pack_context([], [], count_tokens=len, budget=-1)


if __name__ == '__main__':
    unittest.main()
