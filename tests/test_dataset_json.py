import json
import random
import tempfile
import unittest
from pathlib import Path

from flashrag.dataset import Dataset


class TestDatasetJson(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "samples.json"
        self.config = {"dataset_name": "test"}
        self.records = [
            {"id": str(i), "question": f"问题 {i}", "golden_answers": [str(i)]}
            for i in range(4)
        ]

    def load(self, **kwargs):
        return Dataset(config=self.config, dataset_path=str(self.path), **kwargs)

    def test_saved_dataset_can_be_reloaded_with_outputs_and_metadata(self):
        self.records[0]["choices"] = ["甲", "乙"]
        self.records[0]["metadata"] = {"source": "合成样本"}
        dataset = Dataset(config=self.config, data=self.records)
        dataset.update_output("pred", ["0", "1", "x", "3"])
        dataset[0].update_evaluation_score("em", 1.0)
        dataset.save(str(self.path))

        restored = self.load()

        self.assertEqual([item.to_dict() for item in restored], json.loads(self.path.read_text(encoding="utf-8")))
        self.assertEqual(restored.id, dataset.id)
        self.assertEqual(restored.question, dataset.question)
        self.assertEqual(restored.pred, ["0", "1", "x", "3"])
        self.assertEqual(restored[0].output["metric_score"], {"em": 1.0})
        self.assertEqual(restored[0].metadata, {"source": "合成样本"})
        self.assertEqual(restored[0].choices, ["甲", "乙"])

    def test_json_arrays_accept_compact_pretty_and_leading_whitespace(self):
        for indent, prefix in [(None, ""), (4, ""), (2, " \n\t")]:
            with self.subTest(indent=indent, prefix=prefix):
                self.path.write_text(prefix + json.dumps(self.records, indent=indent), encoding="utf-8")
                self.assertEqual(self.load().id, ["0", "1", "2", "3"])

    def test_saved_empty_dataset_can_be_reloaded(self):
        Dataset(config=self.config, data=[]).save(str(self.path))
        self.assertEqual(len(self.load()), 0)

    def test_json_array_preserves_prefix_sampling(self):
        self.path.write_text(json.dumps(self.records), encoding="utf-8")
        self.assertEqual(self.load(sample_num=2).id, ["0", "1"])
        self.assertEqual(len(self.load(sample_num=0)), 0)

    def test_json_array_preserves_random_sampling(self):
        self.path.write_text(json.dumps(self.records), encoding="utf-8")
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(42)
        expected = random.sample(self.records, 2)
        random.seed(42)
        self.assertEqual(self.load(sample_num=2, random_sample=True).id, [item["id"] for item in expected])

    def test_legacy_line_delimited_json_with_either_extension_is_preserved(self):
        for suffix in [".json", ".jsonl"]:
            with self.subTest(suffix=suffix):
                self.path = self.path.with_suffix(suffix)
                self.path.write_text("\n".join(json.dumps(item) for item in self.records) + "\n", encoding="utf-8")
                self.assertEqual(self.load().id, ["0", "1", "2", "3"])

    def test_single_json_object_remains_a_single_record(self):
        self.path.write_text(json.dumps(self.records[0]), encoding="utf-8")
        self.assertEqual(self.load().id, ["0"])

    def test_empty_files_remain_empty_datasets(self):
        for suffix in [".json", ".jsonl"]:
            with self.subTest(suffix=suffix):
                self.path = self.path.with_suffix(suffix)
                self.path.write_text("", encoding="utf-8")
                self.assertEqual(len(self.load()), 0)

    def test_malformed_array_still_raises_json_decode_error(self):
        for contents in ['[{"id": "0"}', '[{"id": "0"}] trailing']:
            with self.subTest(contents=contents):
                self.path.write_text(contents, encoding="utf-8")
                with self.assertRaises(json.JSONDecodeError):
                    self.load()


if __name__ == "__main__":
    unittest.main()
