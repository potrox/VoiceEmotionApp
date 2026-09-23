import unittest
from pathlib import Path

from scripts.check_publication import allowed_path, content_problems, released_model_problems


class PublicationGuardTests(unittest.TestCase):
    def test_only_source_paths_are_allowed(self):
        prefix = "VoiceEmotionApp_refactor/"
        self.assertTrue(allowed_path(prefix + "app/asr.py", prefix))
        self.assertTrue(allowed_path(prefix + "config.example.json", prefix))
        self.assertFalse(allowed_path(prefix + "config.json", prefix))
        self.assertFalse(allowed_path(prefix + "models/model.pkl", prefix))
        self.assertTrue(allowed_path(prefix + "models/model_v11_dusha_62k.pkl", prefix))
        self.assertTrue(allowed_path(prefix + "server/server_app/main.py", prefix))
        self.assertFalse(allowed_path(prefix + "server/.env", prefix))
        self.assertFalse(allowed_path(prefix + "scripts/private_data.csv", prefix))
        self.assertFalse(allowed_path(prefix + "app/secrets.py", prefix))
        self.assertFalse(allowed_path("PROJECT_MAP.md", prefix))

    def test_text_with_private_key_is_rejected(self):
        value = "-----BEGIN " + "PRIVATE KEY-----"
        self.assertIn("private key", content_problems(value.encode()))

    def test_assigned_secret_is_rejected(self):
        value = "api_" + "key = " + repr("not-a-real-credential")
        self.assertIn("assigned credential-like value", content_problems(value.encode()))

    def test_runtime_token_reference_is_not_a_hardcoded_secret(self):
        self.assertEqual([], content_problems(b"tok" + b"en=issued_token\n"))

    def test_binary_and_large_files_are_rejected(self):
        self.assertIn("binary file", content_problems(b"a\x00b"))
        self.assertTrue(content_problems(b"x" * 1_000_001))

    def test_normal_source_is_allowed(self):
        self.assertEqual([], content_problems(b"from pathlib import Path\n"))

    def test_changed_release_model_is_rejected(self):
        self.assertTrue(released_model_problems(b"different model"))

    def test_build_does_not_bundle_local_config(self):
        build_source = (Path(__file__).resolve().parents[1] / "build.py").read_text(encoding="utf-8")
        self.assertNotIn("config.json", build_source)
        self.assertNotIn("--add-data", build_source)


if __name__ == "__main__":
    unittest.main()
