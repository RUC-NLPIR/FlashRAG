import json
import tempfile
import unittest
from pathlib import Path

from flashrag.dataset import Dataset
from flashrag.dataset.utils import filter_dataset


class TestFilterDataset(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "samples.jsonl"
        self.path.write_text(
            "\n".join(json.dumps({"id": str(i), "question": f"Question {i}"}) for i in range(4)),
            encoding="utf-8",
        )
        self.dataset = Dataset(config={"dataset_name": "test"}, dataset_path=str(self.path))

    def test_consecutive_rejections_are_all_removed(self):
        for rejected in [{"0", "1"}, {"1", "2"}, {"2", "3"}]:
            with self.subTest(rejected=rejected):
                dataset = Dataset(config=self.dataset.config, dataset_path=str(self.path))
                filtered = filter_dataset(dataset, lambda item, rejected=rejected: item.id not in rejected)
                self.assertEqual(filtered.id, [str(i) for i in range(4) if str(i) not in rejected])

    def test_predicate_visits_every_item_once_in_order(self):
        visited = []

        def keep(item):
            visited.append(item.id)
            return item.id == "3"

        filtered = filter_dataset(self.dataset, keep)
        self.assertEqual(visited, ["0", "1", "2", "3"])
        self.assertEqual(filtered.id, ["3"])

    def test_filtering_preserves_source_and_retained_items(self):
        original_items = list(self.dataset.data)
        filtered = filter_dataset(self.dataset, lambda item: item.id in {"1", "3"})
        self.assertEqual(self.dataset.data, original_items)
        self.assertIsNot(filtered.data, self.dataset.data)
        self.assertIs(filtered[0], original_items[1])
        self.assertIs(filtered[1], original_items[3])
        self.assertIs(filtered.config, self.dataset.config)

    def test_rejecting_all_items_returns_empty_dataset(self):
        for size in [1, 4]:
            with self.subTest(size=size):
                dataset = Dataset(config=self.dataset.config, data=self.dataset.data[:size])
                filtered = filter_dataset(dataset, lambda item: False)
                self.assertEqual(len(filtered), 0)
                self.assertEqual(filtered.question, [])
                self.assertEqual(len(dataset), size)

    def test_filtering_empty_file_returns_empty_dataset(self):
        self.path.write_text("", encoding="utf-8")
        dataset = Dataset(config=self.dataset.config, dataset_path=str(self.path))
        filtered = filter_dataset(dataset, lambda item: self.fail("Empty input must not invoke the predicate"))
        self.assertEqual(len(filtered), 0)

    def test_accepting_all_items_preserves_order(self):
        filtered = filter_dataset(self.dataset, lambda item: True)
        self.assertEqual(filtered.data, self.dataset.data)
        self.assertIsNot(filtered.data, self.dataset.data)

    def test_no_predicate_returns_original_dataset(self):
        self.assertIs(filter_dataset(self.dataset), self.dataset)


if __name__ == "__main__":
    unittest.main()
