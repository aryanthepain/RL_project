"""Structural and syntax validation tests for notebooks/colab_tqc_benchmark.ipynb."""

import json
import os
import re
import unittest


class TestColabNotebook(unittest.TestCase):
    """Verify structural validity and syntax of the Colab benchmark notebook."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook_path = os.path.join(
            os.path.dirname(__file__), "..", "notebooks", "colab_tqc_benchmark.ipynb"
        )
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.nb = json.load(f)

    def test_notebook_metadata_and_format(self) -> None:
        """Verify notebook conforms to nbformat 4 with GPU accelerator metadata."""
        self.assertEqual(self.nb.get("nbformat"), 4)
        metadata = self.nb.get("metadata", {})
        self.assertEqual(metadata.get("accelerator"), "GPU")
        cells = self.nb.get("cells", [])
        self.assertGreaterEqual(len(cells), 8)

    def test_code_cell_compilation(self) -> None:
        """Ensure all code cells compile as valid Python after stripping shell magics."""
        cells = self.nb.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        for idx, cell in enumerate(code_cells):
            raw_lines = cell.get("source", [])
            clean_lines = []
            for line in raw_lines:
                stripped = line.strip()
                # Skip IPython / bash shell magics for python compilation test
                if stripped.startswith("!") or stripped.startswith("%"):
                    indent = line[: len(line) - len(line.lstrip())]
                    clean_lines.append(f"{indent}pass  # [magic skipped]: {stripped}\n")
                else:
                    clean_lines.append(line)

            code_text = "".join(clean_lines)
            try:
                compile(code_text, f"<cell_{idx}>", "exec")
            except SyntaxError as e:
                self.fail(f"Cell {idx} failed Python syntax compilation: {e}\nCode:\n{code_text}")

    def test_mandatory_cells_and_invariants(self) -> None:
        """Verify all critical invariants are present in the notebook cells."""
        all_text = ""
        for cell in self.nb.get("cells", []):
            all_text += "".join(cell.get("source", [])) + "\n"

        # D-12: Headless EGL rendering configuration
        self.assertIn("MUJOCO_GL", all_text)
        self.assertIn("egl", all_text)

        # D-01: GitHub clone and editable installation
        self.assertIn("git clone", all_text)
        self.assertIn("pip install", all_text)

        # D-02 & D-03: Google Drive mount and crash resilience fallback
        self.assertIn("drive.mount", all_text)
        self.assertIn("tqc_runs", all_text)

        # D-09 & D-10: Environment and Preset selections
        self.assertIn("HalfCheetah-v4", all_text)
        self.assertIn("Humanoid-v4", all_text)
        self.assertIn("quick", all_text)
        self.assertIn("smoke", all_text)

        # D-11: train_tqc and auto-resumption
        self.assertIn("train_tqc", all_text)
        self.assertIn("AUTO_RESUME", all_text)

        # D-13 & D-14: Inactivity-truncated video playback and plotting
        self.assertIn("--truncate-inactive", all_text)
        self.assertIn("IPython.display", all_text)
        self.assertIn("eval_return_mean", all_text)

        # D-04: clean_runs.py integration
        self.assertIn("clean_runs.py", all_text)


if __name__ == "__main__":
    unittest.main()
