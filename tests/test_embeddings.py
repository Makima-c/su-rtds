import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from retrieval import OpenAIEncoder, ConversationRetriever
from prepare_retrieval import main as prepare_main
from experiment_paths import MODELS, load_experiment


def response(texts):
    # Reverse response order to ensure the API's index, not list position, is used.
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=[3., 4.])
              for i in reversed(range(len(texts)))],
        usage=SimpleNamespace(total_tokens=len(texts)),
    )


class EmbeddingTests(unittest.TestCase):
    def test_batches_restore_input_order_normalize_vectors_and_use_selected_model(self):
        create = Mock(side_effect=lambda **kwargs: SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[i + 1., 1.])
                  for i in reversed(range(len(kwargs['input'])))],
            usage=SimpleNamespace(total_tokens=len(kwargs['input'])),
        ))
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        encoder = OpenAIEncoder(client, 'text-embedding-3-small', count_tokens=len)
        vectors = encoder.encode(['text'] * 33)
        self.assertEqual([len(c.kwargs['input']) for c in create.call_args_list], [32, 1])
        self.assertTrue(all(c.kwargs['model'] == 'text-embedding-3-small'
                            and c.kwargs['encoding_format'] == 'float'
                            for c in create.call_args_list))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        np.testing.assert_allclose(vectors[0], [1 / np.sqrt(2), 1 / np.sqrt(2)])
        self.assertEqual(encoder.request_count, 2)
        self.assertEqual(encoder.input_tokens, 33)

    def test_invalid_text_is_rejected_before_any_api_call_and_failure_is_not_retried(self):
        create = Mock(side_effect=RuntimeError('simulated API failure'))
        encoder = OpenAIEncoder(SimpleNamespace(embeddings=SimpleNamespace(create=create)),
                                'text-embedding-3-small', count_tokens=len)
        for invalid in ('', 'x' * 8193):
            with self.subTest(invalid_length=len(invalid)):
                with self.assertRaises(ValueError):
                    encoder.encode(['valid', invalid])
        create.assert_not_called()
        with self.assertRaises(RuntimeError):
            encoder.encode('valid')
        self.assertEqual(create.call_count, 1)

    def test_semantic_uses_query_vectors_but_recency_and_bm25_make_no_query_api_calls(self):
        create = Mock(side_effect=lambda **kwargs: response(kwargs['input']))
        encoder = OpenAIEncoder(SimpleNamespace(embeddings=SimpleNamespace(create=create)),
                                'text-embedding-3-small', count_tokens=len)
        memories = [{'dia_id': str(i), 'order': i, 'formatted': t}
                    for i, t in enumerate(['old park', 'new job'])]
        retriever = ConversationRetriever(memories, encoder, count_tokens=len)
        self.assertEqual(create.call_count, 1)
        retriever.retrieve('park', 'recency')
        retriever.retrieve('park', 'bm25')
        self.assertEqual(create.call_count, 1)
        result = retriever.retrieve('park', 'semantic')
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args.kwargs['input'], ['park'])
        self.assertLessEqual(result['memory_tokens'], 512)

    def test_preparation_publishes_identical_inputs_to_both_models_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src/data').mkdir(parents=True)
            dataset = [{'sample_id': f'conv-{i}', 'conversation': {
                'session_1_date_time': '8 May 2023',
                'session_1': [{'dia_id': f'D1:{j}', 'speaker': 'Alex', 'text': 'Public park text ' * 15} for j in range(1, 4)],
            }} for i in range(10)]
            raw = json.dumps(dataset).encode()
            (root / 'src/data/locomo10.json').write_bytes(raw)
            selected = [
                {'sample_id': d['sample_id'], 'qa_index': j, 'category': 4,
                 'question': 'When?', 'answer': 'PRIVATE REFERENCE', 'evidence': ['D1:1']}
                for d in dataset for j in range(2)
            ]
            (root / 'src/data/selected_qa_20.json').write_text(json.dumps(selected))
            create = Mock(side_effect=lambda **kwargs: response(kwargs['input']))
            constructor = Mock(return_value=SimpleNamespace(embeddings=SimpleNamespace(create=create)))
            tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text), name='test-tokenizer')
            with patch('prepare_retrieval.ROOT', root), \
                    patch.dict('os.environ', {'OPENAI_TEXT_EMBEDDING_MODEL': 'text-embedding-3-small',
                                             'OPENAI_API_KEY': 'test-only'}), \
                    patch.dict(sys.modules, {'dotenv': SimpleNamespace(load_dotenv=lambda *a, **k: None),
                                             'openai': SimpleNamespace(OpenAI=constructor),
                                             'tiktoken': SimpleNamespace(get_encoding=lambda name: tokenizer)}), \
                    patch.object(sys, 'argv', ['prepare_retrieval.py', '--dry-run']), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare_main()
                constructor.assert_not_called()
                self.assertFalse((root / 'outputs' / MODELS[0] / '512_tokens/inputs/retrieval.json').exists())
                with patch.object(sys, 'argv', ['prepare_retrieval.py']):
                    prepare_main()
            nano, _ = load_experiment(MODELS[0], root)
            mini, _ = load_experiment(MODELS[1], root)
            self.assertEqual((nano / 'inputs/retrieval.json').read_bytes(),
                             (mini / 'inputs/retrieval.json').read_bytes())
            output = json.loads((mini / 'inputs/retrieval.json').read_text())
            self.assertEqual(len(output), 60)
            self.assertEqual(set(output[0]), {'sample_id', 'qa_index', 'strategy',
                                               'context', 'retrieved_ids', 'memory_tokens', 'retrieval_latency_ms'})
            self.assertFalse((nano / 'config.json').exists())
            self.assertNotIn('PRIVATE REFERENCE', str(create.call_args_list))
            self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)
            frozen = (nano / 'inputs/retrieval.json').read_bytes()
            with patch('prepare_retrieval.ROOT', root), \
                    patch.dict('os.environ', {'OPENAI_TEXT_EMBEDDING_MODEL': 'text-embedding-3-small',
                                             'OPENAI_API_KEY': 'test-only'}), \
                    patch.dict(sys.modules, {'dotenv': SimpleNamespace(load_dotenv=lambda *a, **k: None),
                                             'openai': SimpleNamespace(OpenAI=constructor),
                                             'tiktoken': SimpleNamespace(get_encoding=lambda name: tokenizer)}), \
                    patch.object(sys, 'argv', ['prepare_retrieval.py', '--budget', '1024']), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare_main()
            large_nano, large_rows = load_experiment(MODELS[0], root, budget=1024)
            large_mini, _ = load_experiment(MODELS[1], root, budget=1024)
            self.assertEqual((nano / 'inputs/retrieval.json').read_bytes(), frozen)
            self.assertEqual((large_nano / 'inputs/retrieval.json').read_bytes(),
                             (large_mini / 'inputs/retrieval.json').read_bytes())
            self.assertTrue(all(512 < r['memory_tokens'] <= 1024 for r in large_rows))
            self.assertTrue(all(r['memory_tokens'] <= 512 for r in output))


if __name__ == '__main__':
    unittest.main()
