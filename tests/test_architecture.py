import tempfile
import unittest
from pathlib import Path

from scripts.check_architecture import audit


class ArchitectureTests(unittest.TestCase):
    def test_project_has_no_cycles_or_inverted_ui_dependency(self):
        self.assertEqual(audit(Path(__file__).resolve().parents[1]), [])

    def test_cycle_and_core_to_ui_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app").mkdir()
            (root / "scripts").mkdir()
            (root / "app" / "a.py").write_text("from .gui import MainWindow\n", encoding="utf-8")
            (root / "app" / "gui.py").write_text("from .a import value\n", encoding="utf-8")
            result = audit(root)
            self.assertTrue(any("import cycle" in message for message in result))
            self.assertTrue(any("imports UI" in message for message in result))

    def test_server_cycles_and_desktop_imports_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "server" / "server_app"
            package.mkdir(parents=True)
            (package / "a.py").write_text("from .b import value\nfrom app.gui import MainWindow\n", encoding="utf-8")
            (package / "b.py").write_text("from .a import value\n", encoding="utf-8")
            result = audit(root)
            self.assertTrue(any("import cycle" in message for message in result))
            self.assertTrue(any("imports desktop" in message for message in result))


if __name__ == "__main__":
    unittest.main()
