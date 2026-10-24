import copy
import contextlib
import io
import json
import tempfile
from unittest.mock import patch
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_results import summarize, main as analysis_main
from experiment_paths import experiment_metadata


class AnswerScoringTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'question': f'Question {i // 3}',
                      'llm_answer': f'Answer {i}', 'answer': f'Reference {i // 3}',
                      'evidence': ['D1:1'], 'retrieved_ids': ['D1:1'], 'correct': 0,
                      'sample_id': f'conv-{i // 6}', 'qa_index': i // 3,
                      'strategy': ('recency', 'bm25', 'semantic')[i % 3], 'category': 4}
                     for i in range(60)]
        self.rows[0]['correct'] = 1

    def test_analysis_reads_answers_and_writes_summary_in_outputs_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'gpt-5.4-mini' / '1024_tokens'
            folder.mkdir(parents=True)
            answers = [{**{k: r[k] for k in ('sample_id', 'qa_index', 'strategy', 'question',
                                              'evidence', 'llm_answer', 'correct')},
                        'reference_answer': r['answer']}
                       for r in self.rows]
            answers[0]['correct'] = None
            output = folder / 'outputs'
            output.mkdir()
            path = output / 'answer.json'
            path.write_text(json.dumps({'metadata': experiment_metadata('gpt-5.4-mini', self.rows, 'text-embedding-3-small', budget=1024),
                                        'answers': answers}, indent=2))
            before = path.read_bytes()
            with patch('analyze_results.load_experiment', return_value=(folder, self.rows)) as load, \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None, 'matplotlib': None}), \
                    patch.object(sys, 'argv', ['analyze_results.py', '--model', 'gpt-5.4-mini', '--budget', '1024']), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, 'Fill all 60.*answer.json'):
                    analysis_main()
            load.assert_called_once_with('gpt-5.4-mini', budget=1024)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in output.iterdir()], ['answer.json'])
            answers[0]['correct'] = 1
            document = {'metadata': experiment_metadata('gpt-5.4-mini', self.rows, 'text-embedding-3-small', budget=1024),
                        'answers': answers}
            path.write_text(json.dumps(document, indent=2))
            scored = path.read_bytes()
            with patch('analyze_results.load_experiment', return_value=(folder, self.rows)) as load, \
                    patch.dict(sys.modules, {'openai': None, 'dotenv': None}), \
                    patch.object(sys, 'argv', ['analyze_results.py', '--model', 'gpt-5.4-mini', '--budget', '1024']), \
                    contextlib.redirect_stdout(io.StringIO()):
                analysis_main()
            load.assert_called_once_with('gpt-5.4-mini', budget=1024)
            self.assertEqual(path.read_bytes(), scored)
            self.assertEqual(json.loads((output / 'summary.json').read_text()), summarize(self.rows))
            self.assertFalse((folder / 'answer.json').exists())
            self.assertFalse((folder / 'summary.json').exists())
            self.assertTrue((folder / 'figures/accuracy.png').exists())

    def test_summary_reads_correct_and_preserves_answers_and_evidence(self):
        self.rows[0]['evidence'] = ['D1:1', 'D1:2']
        before = copy.deepcopy(self.rows)
        result = summarize(self.rows)
        self.assertEqual(self.rows, before)
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result[0]), {'retrieval_method', 'total_questions', 'correct_answers',
                                         'answer_accuracy_percent', 'evidence_recall_percent'})
        self.assertEqual(result[0]['retrieval_method'], 'recency')
        self.assertEqual(result[0]['total_questions'], 20)
        self.assertEqual(result[0]['correct_answers'], 1)
        self.assertEqual(result[0]['answer_accuracy_percent'], 5)
        self.assertEqual(result[0]['evidence_recall_percent'], 97.5)

    def test_rounds_final_macro_recall_to_two_decimals_and_keeps_numbers_numeric(self):
        self.rows[0]['evidence'] = ['D1:1', 'D1:2', 'D1:3']
        result = summarize(self.rows)
        self.assertEqual(result[0]['evidence_recall_percent'], 96.67)
        self.assertIsInstance(result[0]['evidence_recall_percent'], float)
        self.assertIn('"evidence_recall_percent": 96.67', json.dumps(result))

    def test_latency_means_use_all_questions_and_preserve_original_values(self):
        for i, row in enumerate(self.rows):
            row['retrieval_latency_ms'] = i + 0.123
            row['answer_api_latency_ms'] = 100 + i + 0.456
        before = copy.deepcopy(self.rows)
        result = summarize(self.rows)
        self.assertEqual(result[0]['average_retrieval_latency_ms'], 28.62)
        self.assertEqual(result[0]['average_answer_api_latency_ms'], 128.96)
        self.assertEqual(result[0]['average_total_latency_ms'], 157.58)
        self.assertEqual(self.rows, before)
        self.assertEqual(result[0]['answer_accuracy_percent'], 5)

    def test_partial_or_invalid_latency_is_not_summarized_as_zero_or_a_subset(self):
        self.rows[0]['retrieval_latency_ms'] = 1
        with self.assertRaisesRegex(ValueError, 'latency'):
            summarize(self.rows)
        for row in self.rows:
            row['retrieval_latency_ms'] = 1
        for invalid in (None, True, -1, '1', float('nan'), float('inf')):
            self.rows[0]['retrieval_latency_ms'] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'latency'):
                summarize(self.rows)

    def test_strict_context_mismatch_keeps_full_group_means_null(self):
        for row in self.rows:
            row['retrieval_latency_ms'] = 10
            row['answer_api_latency_ms'] = 100
        self.rows[2]['retrieval_latency_ms'] = None
        with self.assertRaisesRegex(ValueError, 'latency'):
            summarize(self.rows)
        result = summarize(self.rows, allow_missing_retrieval=True)
        self.assertEqual(result[0]['average_total_latency_ms'], 110)
        self.assertEqual(result[2]['average_answer_api_latency_ms'], 100)
        self.assertIsNone(result[2]['average_retrieval_latency_ms'])
        self.assertIsNone(result[2]['average_total_latency_ms'])
        self.assertEqual(result[2]['total_questions'], 20)
        self.assertEqual(result[2]['evidence_recall_percent'], 100)
        self.rows[0]['answer_api_latency_ms'] = None
        with self.assertRaisesRegex(ValueError, 'latency'):
            summarize(self.rows, allow_missing_retrieval=True)

    def test_invalid_or_missing_scores_stop_analysis(self):
        for invalid in (None, True, '1', 2, 0.5):
            with self.subTest(invalid=invalid):
                self.rows[0]['correct'] = invalid
                with self.assertRaisesRegex(ValueError, 'Fill all 60|Scores must'):
                    summarize(self.rows)


if __name__ == '__main__':
    unittest.main()
