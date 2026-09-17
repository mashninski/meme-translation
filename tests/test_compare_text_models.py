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


if __name__ == "__main__":
    unittest.main()
