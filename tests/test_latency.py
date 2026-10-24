import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from measure_latency import measure_retrieval, measure_answers
from experiment_paths import answer_key


class SupplementaryLatencyTests(unittest.TestCase):
    def group(self, directory, model='gpt-5-nano'):
        folder = Path(directory) / model / '512_tokens'
        context = {'sample_id': 'conv-1', 'qa_index': 0, 'strategy': 'semantic',
                   'question': 'When?', 'context': 'Original context',
                   'retrieved_ids': ['D1:1'], 'memory_tokens': 3}
        answer = {'sample_id': 'conv-1', 'qa_index': 0, 'strategy': 'semantic',
                  'question': 'When?', 'reference_answer': 'Yesterday',
                  'evidence': ['D1:1'], 'llm_answer': 'Original answer', 'correct': 1}
        document = {'metadata': {'llm_model': model}, 'answers': [answer]}
        return {'folder': folder, 'contexts': [context], 'document': document,
                'answers': {answer_key(answer): answer}}

    def test_retrieval_is_shared_and_context_mismatch_stops_without_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = {(model, 512): self.group(directory, model)
                      for model in ('gpt-5-nano', 'gpt-5.4-mini')}
            retriever = Mock()
            retriever.retrieve.return_value = {'context': 'Different context',
                'retrieved_ids': ['D1:1'], 'memory_tokens': 3, 'retrieval_latency_ms': 12.34}
            before = copy.deepcopy(groups)
            with patch('measure_latency.build_memories', return_value=[]), \
                    patch('measure_latency.ConversationRetriever', return_value=retriever), \
                    self.assertRaisesRegex(ValueError, 'context'):
                measure_retrieval(groups, {'conv-1': {}}, Mock(), len)
            self.assertEqual(groups, before)
            self.assertFalse(list(Path(directory).rglob('answer.json')))
            retriever.retrieve.return_value['context'] = 'Original context'
            with patch('measure_latency.build_memories', return_value=[]), \
                    patch('measure_latency.ConversationRetriever', return_value=retriever):
                measure_retrieval(groups, {'conv-1': {}}, Mock(), len)
            for group in groups.values():
                doc = json.loads((group['folder'] / 'outputs/answer.json').read_text())
                self.assertEqual(doc['answers'][0]['retrieval_latency_ms'], 12.34)
                self.assertEqual(doc['answers'][0]['llm_answer'], 'Original answer')
                self.assertEqual(doc['answers'][0]['correct'], 1)
                self.assertEqual(doc['metadata']['latency_measurement'], 'separate_run')
            for group in groups.values():
                group['document']['answers'][0].pop('retrieval_latency_ms')
            retriever.retrieve.return_value['context'] = 'Different context'
            with patch('measure_latency.build_memories', return_value=[]), \
                    patch('measure_latency.ConversationRetriever', return_value=retriever):
                measure_retrieval(groups, {'conv-1': {}}, Mock(), len, skip_mismatches=True)
            self.assertTrue(all(g['document']['answers'][0]['retrieval_latency_ms'] is None for g in groups.values()))
            with patch('measure_latency.ConversationRetriever') as constructor:
                measure_retrieval(groups, {'conv-1': {}}, Mock(), len, skip_mismatches=True)
                constructor.assert_not_called()
            groups['gpt-5.4-mini', 512]['document']['answers'][0].pop('retrieval_latency_ms')
            with patch('measure_latency.ConversationRetriever') as constructor:
                measure_retrieval(groups, {'conv-1': {}}, Mock(), len, skip_mismatches=True)
                constructor.assert_not_called()
            self.assertIsNone(groups['gpt-5.4-mini', 512]['document']['answers'][0]['retrieval_latency_ms'])

    def test_answer_timing_preserves_text_score_and_resumes_without_new_call(self):
        with tempfile.TemporaryDirectory() as directory:
            group = self.group(directory)
            response = SimpleNamespace(status='completed', output_text='A different answer')
            client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
            with patch('measure_latency.perf_counter', side_effect=[1., 2.234567]):
                measure_answers(group, client)
            doc = json.loads((group['folder'] / 'outputs/answer.json').read_text())
            self.assertEqual(doc['answers'][0]['answer_api_latency_ms'], 1234.57)
            self.assertEqual(doc['answers'][0]['llm_answer'], 'Original answer')
            self.assertEqual(doc['answers'][0]['correct'], 1)
            self.assertNotIn('Yesterday', str(client.responses.create.call_args))
            measure_answers(group, client)
            self.assertEqual(client.responses.create.call_count, 1)

    def test_relationship_figures_show_six_retrieval_points_and_twelve_answer_points(self):
        from plot_latency import plot_relationships
        from matplotlib.axes import Axes
        data = {(model, budget): {
                    strategy: {'average_retrieval_latency_ms': 10 * (i + 1),
                               'evidence_recall_percent': i * 30,
                               'average_total_latency_ms': 100 * (i + 1),
                               'answer_accuracy_percent': i * 25}
                    for i, strategy in enumerate(('recency', 'bm25', 'semantic'))}
                for model in ('gpt-5-nano', 'gpt-5.4-mini') for budget in (512, 1024)}
        original = Axes.scatter
        counts = []
        def capture(ax, *args, **kwargs):
            grid = ax.get_subplotspec().get_gridspec()
            counts.append((grid.nrows, grid.ncols, ax, kwargs.get('marker')))
            return original(ax, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(Axes, 'scatter', capture):
            plot_relationships(data, Path(directory))
            self.assertEqual(sum(grid[:2] == (1, 2) for grid in counts), 6)
            self.assertEqual(sum(grid[:2] == (2, 2) for grid in counts), 12)
            self.assertTrue(all(row[3] == 'o' for row in counts))
            for layout, expected_axes in [((1, 2), 2), ((2, 2), 4)]:
                axes = {row[2] for row in counts if row[:2] == layout}
                self.assertEqual(len(axes), expected_axes)
                self.assertEqual(len({ax.get_xlim() for ax in axes}), 1)
                self.assertEqual(len({ax.get_ylim() for ax in axes}), 1)
                for ax in axes:
                    self.assertEqual(ax.get_xscale(), 'linear')
                    self.assertEqual({text.get_text() for text in ax.texts}, {'Recency', 'BM25', 'Semantic'})
                    self.assertFalse(ax.figure.legends)

            for name in ('latency_vs_recall.png', 'latency_vs_accuracy.png'):
                self.assertTrue((Path(directory) / name).read_bytes().startswith(b'\x89PNG'))
            for model in ('gpt-5-nano', 'gpt-5.4-mini'):
                data[model, 1024]['semantic']['average_retrieval_latency_ms'] = None
                data[model, 1024]['semantic']['average_total_latency_ms'] = None
            counts.clear()
            plot_relationships(data, Path(directory))
            self.assertEqual(sum(grid[:2] == (1, 2) for grid in counts), 5)
            self.assertEqual(sum(grid[:2] == (2, 2) for grid in counts), 10)

    def test_failed_or_incomplete_answer_request_does_not_save_a_latency(self):
        for response in (RuntimeError('simulated failure'),
                         SimpleNamespace(status='incomplete', output_text='Partial')):
            with self.subTest(response=type(response).__name__), tempfile.TemporaryDirectory() as directory:
                group = self.group(directory)
                create = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
                client = SimpleNamespace(responses=SimpleNamespace(create=create))
                with self.assertRaises(SystemExit):
                    measure_answers(group, client)
                self.assertNotIn('answer_api_latency_ms', group['document']['answers'][0])
                self.assertFalse(list(Path(directory).rglob('answer.json')))
                self.assertEqual(create.call_count, 1)


if __name__ == '__main__':
    unittest.main()
