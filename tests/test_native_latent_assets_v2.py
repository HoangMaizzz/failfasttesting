import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from native_latent_assets_v2 import prepare_native_model_assets


class NativeLatentAssetsV2Tests(unittest.TestCase):
    def make_assets(self, root):
        repo = root / "repo"
        code = repo / "Fast_dLLM_v2_1_5B"
        code.mkdir(parents=True)
        (code / "modeling.py").write_text(
            "wm_v2_capture_native_latent = True\nwm_v2_native_latent_block = []\n",
            encoding="utf-8",
        )
        weights = root / "downloaded_weights"
        weights.mkdir()
        (weights / "configuration.py").write_text("class Config: pass\n", encoding="utf-8")
        (weights / "config.json").write_text(json.dumps({"auto_map": {
            "AutoConfig": "configuration.Config",
            "AutoModelForCausalLM": "modeling.Model",
        }}), encoding="utf-8")
        return repo, weights

    def test_separate_weights_directory_gets_exact_instrumented_source(self):
        with TemporaryDirectory() as tmp:
            repo, weights = self.make_assets(Path(tmp))
            weights_config = (weights / "config.json").read_bytes()
            # Reproduce the Kaggle failure: weights/config exist, code does not.
            self.assertFalse((weights / "modeling.py").exists())
            digest = prepare_native_model_assets(repo, weights)
            expected = (repo / "Fast_dLLM_v2_1_5B" / "modeling.py").read_bytes()
            self.assertEqual((weights / "modeling.py").read_bytes(), expected)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            self.assertEqual((weights / "config.json").read_bytes(), weights_config)
            # A stale, uninstrumented model from an earlier session is replaced.
            (weights / "modeling.py").write_text("stale = True\n", encoding="utf-8")
            self.assertEqual(prepare_native_model_assets(repo, weights), digest)

    def test_missing_configuration_fails_before_model_load(self):
        with TemporaryDirectory() as tmp:
            repo, weights = self.make_assets(Path(tmp))
            (weights / "configuration.py").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "configuration.py"):
                prepare_native_model_assets(repo, weights)

    def test_source_without_capture_and_wrong_auto_map_are_rejected(self):
        with TemporaryDirectory() as tmp:
            repo, weights = self.make_assets(Path(tmp))
            (weights / "config.json").write_text(json.dumps({"auto_map": {
                "AutoModelForCausalLM": "other_module.Model",
            }}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local modeling module"):
                prepare_native_model_assets(repo, weights)
            (repo / "Fast_dLLM_v2_1_5B" / "modeling.py").write_text(
                "uninstrumented = True\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "V2 capture support"):
                prepare_native_model_assets(repo, weights)


if __name__ == "__main__":
    unittest.main()
