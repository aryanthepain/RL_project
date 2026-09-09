import unittest
from src.tqc.eval_table import PAPER_REFERENCE_SCORES, generate_comparison_table


class TestEvalTable(unittest.TestCase):
    def test_paper_reference_scores_completeness(self):
        expected_envs = [
            "HalfCheetah-v4",
            "Hopper-v4",
            "Walker2d-v4",
            "Ant-v4",
            "Humanoid-v4",
        ]
        for env in expected_envs:
            self.assertIn(env, PAPER_REFERENCE_SCORES)
            self.assertIn("tqc_paper", PAPER_REFERENCE_SCORES[env])
            self.assertIn("sac_paper", PAPER_REFERENCE_SCORES[env])

    def test_generate_comparison_table_with_mock_data(self):
        mock_data = {
            "HalfCheetah-v4": {
                "tqc": "14,100 ± 210",
                "sac": "11,520 ± 240",
            },
            "Hopper-v4": {
                "tqc": "3,710 ± 140",
                "sac": "3,440 ± 95",
            },
        }

        md_tbl, tex_tbl = generate_comparison_table(empirical_data=mock_data)

        # Assert Markdown structure
        self.assertIn("| Environment |", md_tbl)
        self.assertIn("14,028 ± 232", md_tbl)
        self.assertIn("14,100 ± 210", md_tbl)
        self.assertIn("Humanoid-v4", md_tbl)

        # Assert LaTeX structure
        self.assertIn(r"\begin{table}", tex_tbl)
        self.assertIn(r"\begin{tabular}", tex_tbl)
        self.assertIn(r"\end{table}", tex_tbl)
        self.assertIn(r"14,028 \pm 232", tex_tbl)

    def test_generate_comparison_table_empty_data(self):
        md_tbl, tex_tbl = generate_comparison_table(empirical_data={})
        self.assertIn("Pending Run", md_tbl)
        self.assertIn("Pending", tex_tbl)


if __name__ == "__main__":
    unittest.main()
