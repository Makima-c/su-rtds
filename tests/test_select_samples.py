import copy
import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from select_samples import select_samples, validate_samples


def dataset():
    return [
        {
            "sample_id": f"conv-{i}",
            "conversation": {
                "session_1_date_time": "1:00 pm on 8 May, 2023",
                "session_1": [
                    {"dia_id": "D1:1", "speaker": "Alice", "text": "Yesterday."}
                ],
            },
            "qa": [
                {"category": category, "question": f"Question {j}?",
                 "answer": f"Answer {j}", "evidence": ["D1:1"]}
                for j, category in enumerate([1, 2, 4, 1, 2, 4])
            ],
        }
        for i in range(10)
    ]


class SampleSelectionTests(unittest.TestCase):
    def test_final_samples_match_source_without_review_metadata(self):
        data = dataset()
        selected = select_samples(data)
        validate_samples(data, selected)
        self.assertEqual(set(selected[0]), {'sample_id', 'qa_index', 'category',
                                           'question', 'answer', 'evidence'})
        selected[0]['answer'] = 'changed reference'
        with self.assertRaisesRegex(ValueError, 'source'):
            validate_samples(data, selected)

    def test_reproducible_balanced_selection_preserves_source_and_dates(self):
        data = dataset()
        original = copy.deepcopy(data)
        selected = select_samples(data, seed=42)
        self.assertEqual(selected, select_samples(data, seed=42))
        self.assertEqual(Counter(q["category"] for q in selected), {4: 10, 1: 5, 2: 5})
        self.assertEqual(Counter(q["sample_id"] for q in selected),
                         {s["sample_id"]: 2 for s in data})
        self.assertEqual(len({(q["sample_id"], q["qa_index"]) for q in selected}), 20)
        for q in selected:
            source = next(s for s in data if s["sample_id"] == q["sample_id"])
            qa = source["qa"][q["qa_index"]]
            self.assertEqual(q["question"], qa["question"])
            self.assertEqual(q["answer"], qa["answer"])
            self.assertEqual(q["evidence"], qa["evidence"])
        self.assertEqual(data, original)

    def test_excludes_invalid_annotations_blank_answers_and_nontext_evidence(self):
        data = dataset()
        invalid = [
            {"category": 4, "question": "Bad?", "answer": "Yes", "evidence": evidence}
            for evidence in [[], ["D1:1; D1:2"], ["D"], [None], ["D1:2"]]
        ]
        invalid += [{"category": 4, "question": "Bad?", "answer": "  ",
                     "evidence": ["D1:1"]}]
        for sample in data:
            sample["conversation"]["session_1"].append(
                {"dia_id": "D1:2", "speaker": "Alice", "text": " "})
            sample["qa"] = invalid + sample["qa"]
        selected = select_samples(data)
        self.assertTrue(all(q["qa_index"] >= len(invalid) for q in selected))

    def test_insufficient_eligible_categories_fail_instead_of_changing_quotas(self):
        data = dataset()
        data[0]["qa"] = [q for q in data[0]["qa"] if q["category"] != 2]
        with self.assertRaisesRegex(ValueError, "category 2"):
            select_samples(data)


if __name__ == "__main__":
    unittest.main()
