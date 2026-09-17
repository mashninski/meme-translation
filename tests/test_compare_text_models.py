"""Проверки локальной логики эксперимента без API-запросов."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_text_models.py"
spec = importlib.util.spec_from_file_location("compare_text_models", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def test_normalize_keeps_belarusian_letters_digits_and_apostrophe(self):
        self.assertEqual(module.normalize("  П'яць,  аб’ектаў!  № 23\nЁсць?  "), "п'яць аб'ектаў 23 ёсць")

    def test_editor_request_contains_only_text(self):
        body = module.editor_body(["Hello", "world"], "Вітаю, свет!")
        self.assertEqual(body["model"], "gpt-5.6-terra")
        self.assertEqual([part["type"] for part in body["input"][0]["content"]], ["input_text"])
        self.assertNotIn("image_url", str(body))

    def test_image_cost_is_incomplete_without_image_rate(self):
        usage = {"input_tokens": 100, "output_tokens": 20}
        estimate = module.cost("gpt-5.6-luna", usage, True)
        self.assertFalse(estimate["complete"])
        self.assertAlmostEqual(estimate["known_subtotal_usd"], 20 * 1.20 / 1_000_000)

    def test_empty_target_must_have_empty_translation(self):
        value = {"target_text": [], "ignored_text": [], "ignored_reason": "", "translation": "Тэкст", "ambiguous": False}
        with self.assertRaises(ValueError):
            module.validate_vision(value)

    def test_sol_review_uses_only_saved_text(self):
        body = module.sol_editor_body(["EVEN IN RUIN"], "Нават у руінах")
        self.assertEqual(body["model"], "gpt-5.6-sol")
        self.assertEqual([part["type"] for part in body["input"][0]["content"]], ["input_text"])
        self.assertNotIn("image_url", str(body))
        self.assertIn("EVEN IN RUIN", body["input"][0]["content"][0]["text"])

    def test_sol_summary_counts_exact_translation_changes(self):
        items = [
            {"terra_translation": "Прывітанне!", "sol_record": {"raw_structured": {"translation": "Прывітанне!"}, "error": None, "usage": {"input_tokens": 10, "output_tokens": 10}, "cost": {"complete": True, "known_subtotal_usd": 0.00024}}},
            {"terra_translation": "Як справы?", "sol_record": {"raw_structured": {"translation": "Як маешся?"}, "error": None, "usage": {"input_tokens": 10, "output_tokens": 10}, "cost": {"complete": True, "known_subtotal_usd": 0.00024}}},
        ]
        summary = module.sol_summary(items)
        self.assertEqual(summary["completed_reviews"], 2)
        self.assertEqual(summary["changed_translations"], 1)
        self.assertAlmostEqual(summary["total_review_cost_usd"], 0.00048)

    def test_sol_summary_accepts_pending_images(self):
        summary = module.sol_summary([{"terra_translation": "Прывітанне", "sol_record": None}])
        self.assertEqual(summary["completed_reviews"], 0)
        self.assertIsNone(summary["total_review_cost_usd"])

    def test_conservative_review_uses_only_terra_text(self):
        body = module.conservative_editor_body(["Hello"], "Вітаю")
        self.assertEqual(body["model"], "gpt-5.6-sol")
        self.assertEqual([part["type"] for part in body["input"][0]["content"]], ["input_text"])
        self.assertNotIn("image_url", str(body))
        self.assertNotIn("old_sol_translation", str(body))
        self.assertIn("Вітаю", body["input"][0]["content"][0]["text"])

    def test_conservative_change_requires_specific_reason(self):
        with self.assertRaises(ValueError):
            module.validate_conservative({"final_translation": "Вітаю!", "changed": True, "reason": ""}, "Вітаю")
        with self.assertRaises(ValueError):
            module.validate_conservative({"final_translation": "Вітаю", "changed": True, "reason": "Памылка"}, "Вітаю")
        module.validate_conservative({"final_translation": "Вітаю", "changed": False, "reason": ""}, "Вітаю")

    def test_conservative_partial_save_and_summary(self):
        items = [{"filename": "x.jpg", "target_text": ["Hello"], "terra_translation": "Вітаю", "old_sol_translation": "Прывітанне", "conservative_record": None}]
        summary = module.conservative_summary(items)
        self.assertEqual(summary["completed_reviews"], 0)
        self.assertIsNone(summary["total_review_cost_usd"])
        self.assertIn("Старый Sol-review", module.render_conservative_comparison(items))


if __name__ == "__main__":
    unittest.main()
