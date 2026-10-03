import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import flashrag.refiner as refiner_module
from flashrag.utils import get_refiner


class TestGetRefiner(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def model_path(self, model_type, architecture):
        path = Path(self.directory.name) / model_type
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps({"model_type": model_type, "architectures": [architecture]}),
            encoding="utf-8",
        )
        return str(path)

    def resolve(self, refiner_name, model_path):
        with mock.patch.object(
            refiner_module, "AbstractiveRecompRefiner", lambda config: "AbstractiveRecompRefiner"
        ), mock.patch.object(refiner_module, "ExtractiveRefiner", lambda config: "ExtractiveRefiner"):
            return get_refiner({"refiner_name": refiner_name, "refiner_model_path": model_path})

    def test_documented_abstractive_name_resolves_seq2seq_models(self):
        for model_type, architecture in [
            ("bart", "BartForConditionalGeneration"),
            ("t5", "T5ForConditionalGeneration"),
        ]:
            with self.subTest(model_type=model_type):
                path = self.model_path(model_type, architecture)
                self.assertEqual(self.resolve("abstractive", path), "AbstractiveRecompRefiner")

    def test_bert_models_still_resolve_to_extractive_refiner(self):
        path = self.model_path("bert", "BertModel")
        self.assertEqual(self.resolve("extractive", path), "ExtractiveRefiner")


if __name__ == "__main__":
    unittest.main()
