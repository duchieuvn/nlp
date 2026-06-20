from pathlib import Path
import sys
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

import export_equation_windows


class ExportEquationWindowsTests(unittest.TestCase):
    def test_embeds_equation_and_replaces_all_exact_self_references(self):
        result = export_equation_windows.enrich_window(
            "paper",
            "2.1",
            "a = b + c",
            "From Eq. ( 2.1 ) and Eq.(2.1), we obtain [EQUATION].",
        )

        self.assertEqual(
            result,
            "From <eqref> and <eqref>, we obtain "
            "<starteqn> a = b + c <endeqn>.",
        )

    def test_replaces_other_and_plural_equation_references(self):
        result = export_equation_windows.enrich_window(
            "paper",
            "4a",
            "x = y",
            "See Eq. (4b), Eqs. (4a, 4b), and Eq. ( 4a ) before [EQUATION].",
        )

        self.assertEqual(
            result,
            "See <other_eqref>, <other_eqref>, and <eqref> before "
            "<starteqn> x = y <endeqn>.",
        )

    def test_grouped_reference_containing_current_id_is_other_reference(self):
        result = export_equation_windows.enrich_window(
            "paper",
            "1",
            "x = y",
            "Using Eqs. (1, 2), we obtain [EQUATION].",
        )

        self.assertEqual(
            result,
            "Using <other_eqref>, we obtain <starteqn> x = y <endeqn>.",
        )

    def test_rejects_missing_or_repeated_equation_marker(self):
        for window, expected_count in (
            ("No marker", 0),
            ("[EQUATION] and [EQUATION]", 2),
        ):
            with self.subTest(window=window):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"paper 'paper'.*equation '1'; found {expected_count}",
                ):
                    export_equation_windows.enrich_window(
                        "paper", "1", "x = y", window
                    )

    def test_extract_windows_keeps_equation_field(self):
        records = export_equation_windows.extract_windows({
            "paper": {
                "1": {
                    "equation": "x = y",
                    "surrounding_text": {"window": "Before [EQUATION] after"},
                }
            }
        })

        self.assertEqual(records[0]["equation"], "x = y")
        self.assertEqual(records[0]["equation_id"], "1")
        self.assertEqual(
            records[0]["window"],
            "Before <starteqn> x = y <endeqn> after",
        )


if __name__ == "__main__":
    unittest.main()
