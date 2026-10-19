import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from flashrag.dataset import Dataset, Item
from flashrag.prompt import MMPromptTemplate


class TestDatasetPromptExport(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (2, 2), color="white")
        self.item = Item({"id": "0", "question": "What is shown?", "image": self.image})
        template = MMPromptTemplate({}, system_prompt="Describe the image.", user_prompt="{question}")
        self.prompt = template.get_string(self.item)
        self.content = self.prompt[1]["content"]
        self.item.update_output("prompt", self.prompt)
        self.item.update_output("pred", "A white square.")

    def assert_original_prompt_preserved(self):
        self.assertIs(self.item.prompt, self.prompt)
        self.assertIs(self.prompt[1]["content"], self.content)
        self.assertEqual(len(self.content), 2)
        self.assertEqual(self.content[0]["type"], "image")
        self.assertIs(self.content[0]["image"], self.image)
        self.assertIs(self.item.image, self.image)

    def assert_export_is_clean(self, exported):
        self.assertNotIn("image", exported)
        self.assertEqual(exported["output"]["prompt"], [
            {"role": "system", "content": "Describe the image."},
            {"role": "user", "content": [{"type": "text", "text": "What is shown?"}]},
        ])
        self.assertEqual(exported["output"]["pred"], "A white square.")

    def test_to_dict_preserves_prompt_and_callers_aliases(self):
        exported = self.item.to_dict()
        self.assert_original_prompt_preserved()
        self.assert_export_is_clean(exported)

    def test_printing_item_preserves_prompt(self):
        exported = json.loads(str(self.item))
        self.assert_original_prompt_preserved()
        self.assert_export_is_clean(exported)

    def test_saving_dataset_preserves_prompt(self):
        dataset = Dataset(config={"dataset_name": "test"}, data=[self.item])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intermediate_data.json"
            dataset.save(str(path))
            exported = json.loads(path.read_text(encoding="utf-8"))

        self.assert_original_prompt_preserved()
        self.assert_export_is_clean(exported[0])

    def test_repeated_export_preserves_input_and_output(self):
        first = self.item.to_dict()
        second = self.item.to_dict()
        self.assertEqual(first, second)
        self.assert_original_prompt_preserved()
        self.assert_export_is_clean(second)

    def test_base64_image_blocks_are_omitted_only_from_export(self):
        self.content[0]["image"] = "data:image/png;base64,AA=="
        exported = self.item.to_dict()
        self.assertIs(self.prompt[1]["content"], self.content)
        self.assertEqual(self.item.prompt[1]["content"][0]["image"], "data:image/png;base64,AA==")
        self.assert_export_is_clean(exported)

    def test_text_prompt_and_numpy_outputs_keep_existing_serialization(self):
        self.item.update_output("prompt", "A plain text prompt.")
        self.item.update_output("scores", np.array([1, 2]))
        self.item.update_output("count", np.int64(2))
        exported = self.item.to_dict()
        self.assertEqual(exported["output"]["prompt"], "A plain text prompt.")
        self.assertEqual(exported["output"]["scores"], [1, 2])
        self.assertEqual(exported["output"]["count"], 2)
        self.assertIsInstance(self.item.output["scores"], np.ndarray)
        self.assertIsInstance(self.item.output["count"], np.int64)


if __name__ == "__main__":
    unittest.main()
