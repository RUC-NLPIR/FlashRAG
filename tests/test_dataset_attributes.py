import copy
import pickle
import unittest

from flashrag.dataset import Dataset, Item


class TestItemAttributes(unittest.TestCase):
    def setUp(self):
        self.item = Item({"id": "0", "question": "Question 0", "golden_answers": ["a"], "subject": "math"})
        self.item.update_output("pred", "a")

    def test_missing_attribute_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            self.item.not_a_field

        self.assertFalse(hasattr(self.item, "not_a_field"))
        self.assertIsNone(getattr(self.item, "not_a_field", None))
        self.assertEqual(self.item.subject, "math")
        self.assertEqual(self.item.pred, "a")

    def test_item_can_be_copied_and_pickled(self):
        for clone in (copy.copy(self.item), copy.deepcopy(self.item), pickle.loads(pickle.dumps(self.item))):
            with self.subTest(clone=clone):
                self.assertEqual(clone.question, "Question 0")
                self.assertEqual(clone.pred, "a")

        clone = copy.deepcopy(self.item)
        clone.update_output("pred", "b")
        self.assertEqual(self.item.pred, "a")


class TestDatasetAttributes(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(
            config={"dataset_name": "test"},
            data=[{"id": str(i), "question": f"Question {i}", "golden_answers": [str(i)]} for i in range(3)],
        )
        self.dataset.update_output("pred", ["0", "1", "x"])

    def test_dataset_can_be_copied_and_pickled(self):
        for clone in (copy.deepcopy(self.dataset), pickle.loads(pickle.dumps(self.dataset))):
            with self.subTest(clone=type(clone)):
                self.assertEqual(clone.question, self.dataset.question)
                self.assertEqual(clone.pred, ["0", "1", "x"])

        clone = copy.deepcopy(self.dataset)
        clone.update_output("pred", ["a", "b", "c"])
        self.assertEqual(self.dataset.pred, ["0", "1", "x"])

    def test_missing_attribute_raises_attribute_error(self):
        self.assertFalse(hasattr(self.dataset, "not_a_field"))

    def test_attribute_helpers_read_item_fields(self):
        self.assertEqual(self.dataset.get_attr_data("pred"), ["0", "1", "x"])
        self.assertEqual(self.dataset.get_attr_data("question"), self.dataset.question)
        self.assertEqual(list(self.dataset.get_batch_data("pred", 2)), [["0", "1"], ["x"]])


if __name__ == "__main__":
    unittest.main()
