"""Отдельный эксперимент по выбору текста мема и беларусскому переводу."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "test-input"
OUTPUT = ROOT / "test-output"
API_URL = "https://api.openai.com/v1/responses"
MODELS = {"A": "gpt-5.6-luna", "B_editor": "gpt-5.6-terra", "C": "gpt-5.6-terra"}
RATES = {"gpt-5.6-luna": (0.20, 1.20), "gpt-5.6-terra": (2.00, 12.00)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "target_text": {"type": "array", "items": {"type": "string"}},
        "ignored_text": {"type": "array", "items": {"type": "string"}},
        "ignored_reason": {"type": "string"},
        "translation": {"type": "string"},
        "ambiguous": {"type": "boolean"},
    },
    "required": ["target_text", "ignored_text", "ignored_reason", "translation", "ambiguous"],
    "additionalProperties": False,
}
EDITOR_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
    "additionalProperties": False,
}
VISION_PROMPT = """Рассмотри изображение как мем. Выбери только текст, который пользователь захотел бы заново разместить как содержание мема. Это смысловая, а не OCR-задача: не копируй весь видимый текст. Игнорируй бренды, логотипы, вывески в сцене, интерфейс, системные подписи и случайный фон, если они не несут шутку. У твита/поста выбирай текст по роли: если мемом служит добавленный комментарий, исходный текст поста игнорируй; если сам текст поста несёт мем, выбирай его. Сохраняй порядок отдельных сегментов в target_text. В ignored_text укажи замеченный, но отвергнутый текст, а в ignored_reason кратко объясни выбор. При сомнении между несколькими трактовками выставь ambiguous=true; не угадывай молча. Если содержательного текста нет, верни пустые target_text и translation. Переведи выбранный текст на естественный беларусский без русизмов, с правильными падежами, смыслом, шуткой и разговорным регистром. Не добавляй содержание и не переводи имена и бренды механически. Сохрани нормальную пунктуацию."""
EDITOR_PROMPT = """Ты редактор беларусского перевода. Получаешь только выбранный исходный текст мема и черновой перевод. Проверь смысл, шутку, разговорный регистр, грамматику и отсутствие русизмов. Исправь лишь необходимое. Если всё хорошо, оставь текст без изменений. Не добавляй нового содержания. Если исходный текст пуст, верни пустой перевод. Сохрани обычную пунктуацию."""


def normalize(text: str) -> str:
    """Оставить буквы, цифры, пробел и апостроф; убрать прочую пунктуацию."""
    kept = []
    for char in text.lower():
        if char.isalpha() or char.isdecimal() or char == "'":
            kept.append(char)
        elif char == "’":
            kept.append("'")
        elif char.isspace():
            kept.append(" ")
    return " ".join("".join(kept).split())


def load_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                return match.group(1).strip().strip('"\'') or None
    return None


def image_files() -> list[Path]:
    return sorted(p for p in INPUT.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def image_content(path: Path) -> dict:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{media_type};base64,{encoded}", "detail": "auto"}


def request_body(model: str, content: list[dict], schema: dict, name: str, instructions: str) -> dict:
    return {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
        "reasoning": {"effort": "none"},
        "max_output_tokens": 1000,
        "store": False,
    }


def vision_body(model: str, path: Path) -> dict:
    return request_body(model, [{"type": "input_text", "text": "Определи текст мема и переведи его."}, image_content(path)], VISION_SCHEMA, "meme_translation", VISION_PROMPT)


def editor_body(target_text: list[str], draft: str) -> dict:
    source = json.dumps({"target_text": target_text, "draft_translation": draft}, ensure_ascii=False)
    return request_body(MODELS["B_editor"], [{"type": "input_text", "text": source}], EDITOR_SCHEMA, "edited_translation", EDITOR_PROMPT)


def call_api(key: str, body: dict) -> dict:
    request = urllib.request.Request(API_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def extract_result(response: dict) -> dict:
    if response.get("status") != "completed":
        raise ValueError(f"Ответ не завершён: {response.get('status')}; {response.get('incomplete_details')}")
    texts = [part["text"] for item in response.get("output", []) for part in item.get("content", []) if part.get("type") == "output_text"]
    if not texts:
        raise ValueError("API не вернул output_text")
    return json.loads("".join(texts))


def validate_vision(value: dict) -> dict:
    if not isinstance(value.get("target_text"), list) or not all(isinstance(x, str) for x in value["target_text"]):
        raise ValueError("Некорректный target_text")
    if not isinstance(value.get("ignored_text"), list) or not all(isinstance(x, str) for x in value["ignored_text"]):
        raise ValueError("Некорректный ignored_text")
    if not isinstance(value.get("ignored_reason"), str) or not isinstance(value.get("translation"), str) or not isinstance(value.get("ambiguous"), bool):
        raise ValueError("Некорректные поля результата")
    if not value["target_text"] and value["translation"]:
        raise ValueError("При пустом target_text перевод должен быть пустым")
    return value


def cost(model: str, usage: dict | None, has_image: bool) -> dict:
    if not isinstance(usage, dict) or not isinstance(usage.get("input_tokens"), int) or not isinstance(usage.get("output_tokens"), int):
        return {"complete": False, "known_subtotal_usd": None, "reason": "usage без числа входных или выходных токенов"}
    input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    input_rate, output_rate = RATES[model]
    output_cost = output_tokens * output_rate / 1_000_000
    if has_image:
        return {"complete": False, "known_subtotal_usd": output_cost, "reason": "входные токены включают изображение; подтверждённой ставки для этой части нет"}
    if cached:
        return {"complete": False, "known_subtotal_usd": output_cost, "reason": "есть кэшированные входные токены; их ставка не включена в исходные цены"}
    return {"complete": True, "known_subtotal_usd": input_tokens * input_rate / 1_000_000 + output_cost, "reason": None}


def run_call(key: str, body: dict, has_image: bool) -> dict:
    response = call_api(key, body)
    record = {"model": body["model"], "response_id": response.get("id"), "status": response.get("status"), "usage": response.get("usage"), "cost": cost(body["model"], response.get("usage"), has_image), "raw_structured": None, "error": None}
    try:
        record["raw_structured"] = extract_result(response)
    except (ValueError, json.JSONDecodeError) as error:
        record["error"] = str(error)
    return record


def fail_record(model: str, error: Exception) -> dict:
    return {"model": model, "response_id": None, "usage": None, "cost": {"complete": False, "known_subtotal_usd": None, "reason": "запрос не дал usage"}, "raw_structured": None, "error": str(error)}


def evaluate(path: Path, key: str) -> dict:
    item = {"filename": path.name, "variants": {}}
    try:
        a = run_call(key, vision_body(MODELS["A"], path), True)
        if a["raw_structured"] is not None:
            try:
                validate_vision(a["raw_structured"])
            except ValueError as error:
                a["error"] = str(error)
        item["variants"]["A"] = a
    except Exception as error:
        item["variants"]["A"] = fail_record(MODELS["A"], error)
    a_data = item["variants"]["A"].get("raw_structured")
    if a_data and not item["variants"]["A"].get("error"):
        try:
            editor = run_call(key, editor_body(a_data["target_text"], a_data["translation"]), False)
            edited = editor.get("raw_structured")
            if edited is not None and not isinstance(edited.get("translation"), str):
                editor["error"] = "Некорректный перевод редактора"
            item["variants"]["B_editor"] = editor
        except Exception as error:
            item["variants"]["B_editor"] = fail_record(MODELS["B_editor"], error)
    else:
        item["variants"]["B_editor"] = fail_record(MODELS["B_editor"], RuntimeError("Пропущено: нет результата Luna"))
    try:
        c = run_call(key, vision_body(MODELS["C"], path), True)
        if c["raw_structured"] is not None:
            try:
                validate_vision(c["raw_structured"])
            except ValueError as error:
                c["error"] = str(error)
        item["variants"]["C"] = c
    except Exception as error:
        item["variants"]["C"] = fail_record(MODELS["C"], error)
    return item


def summary(items: list[dict]) -> dict:
    unique_calls = [record for item in items for record in item["variants"].values() if record.get("usage")]
    amounts = [record["cost"]["known_subtotal_usd"] for record in unique_calls]
    return {
        "api_calls_with_usage": len(unique_calls),
        "complete": bool(unique_calls) and all(record["cost"]["complete"] for record in unique_calls),
        "known_subtotal_usd": sum(x for x in amounts if x is not None),
        "note": "Известная часть расхода; полный итог неизвестен, если есть запросы с изображениями или неполный usage.",
    }


def cell(value: object) -> str:
    if isinstance(value, list):
        value = "\n".join(value)
    return html.escape(str(value if value is not None else "—"))


def render(items: list[dict], key_missing: bool) -> str:
    cards = []
    for item in items:
        variants = item["variants"]
        columns = []
        for label, record_key in (("A · Luna", "A"), ("B · Luna + Terra", "B_editor"), ("C · Terra", "C")):
            record = variants.get(record_key, {})
            data = record.get("raw_structured") or {}
            source = (variants.get("A", {}).get("raw_structured") or {}) if record_key == "B_editor" else data
            translation = data.get("translation", "")
            cost_info = record.get("cost", {})
            if record_key == "B_editor":
                luna_cost = variants.get("A", {}).get("cost", {})
                known = [x for x in (cost_info.get("known_subtotal_usd"), luna_cost.get("known_subtotal_usd")) if x is not None]
                cost_info = {"known_subtotal_usd": sum(known) if known else None, "complete": bool(cost_info.get("complete") and luna_cost.get("complete")), "reason": "Включены Luna и редактор Terra; вход изображения не оценён."}
            usage_view = ({"Luna": variants.get("A", {}).get("usage"), "Terra_editor": record.get("usage")} if record_key == "B_editor" else record.get("usage"))
            cost_text = (f"${cost_info['known_subtotal_usd']:.6f}" if cost_info.get("known_subtotal_usd") is not None else "—") + (" · неполная оценка" if not cost_info.get("complete") else "")
            columns.append(f"<div class='variant'><h3>{label}</h3><dl><dt>target_text</dt><dd>{cell(source.get('target_text'))}</dd><dt>ignored_text</dt><dd>{cell(source.get('ignored_text'))}</dd><dt>ignored_reason</dt><dd>{cell(source.get('ignored_reason'))}</dd><dt>ambiguous</dt><dd>{cell(source.get('ambiguous'))}</dd><dt>Перевод</dt><dd>{cell(translation)}</dd><dt>normalized</dt><dd>{cell(normalize(translation))}</dd><dt>usage</dt><dd><code>{cell(json.dumps(usage_view, ensure_ascii=False))}</code></dd><dt>Учтённая стоимость</dt><dd>{cell(cost_text)}<br><small>{cell(cost_info.get('reason'))}</small></dd><dt>Ошибка</dt><dd>{cell(record.get('error'))}</dd></dl></div>")
        image_src = "../test-input/" + urllib.request.pathname2url(item["filename"])
        cards.append(f"<section class='card'><h2>{cell(item['filename'])}</h2><div class='layout'><img src='{html.escape(image_src, quote=True)}' alt='{cell(item['filename'])}'><div class='variants'>{''.join(columns)}</div></div></section>")
    banner = "Ключ OPENAI_API_KEY отсутствует: API-тест не выполнен. Заполните локальный .env или переменную окружения и запустите скрипт повторно." if key_missing else "Вариант B использует результат A; его стоимость включает запрос Luna и текстовое редактирование Terra."
    return f"""<!doctype html><html lang='ru'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Сравнение переводов мемов</title><style>body{{font:15px/1.45 system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}}h1{{margin-bottom:6px}}.notice{{background:#fff3cb;padding:12px;border-radius:8px}}.card{{background:white;margin:20px 0;padding:18px;border-radius:10px;box-shadow:0 2px 8px #0001}}.layout{{display:grid;grid-template-columns:minmax(220px,32%) 1fr;gap:18px}}img{{max-width:100%;max-height:650px;object-fit:contain;align-self:start}}.variants{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.variant{{background:#f7f9fc;padding:12px;border-radius:8px;min-width:0}}.variant h3{{margin:0 0 8px}}dt{{font-weight:700;margin-top:8px}}dd{{white-space:pre-wrap;overflow-wrap:anywhere;margin:2px 0 0}}code{{font-size:12px}}small{{color:#666}}@media(max-width:1050px){{.layout{{grid-template-columns:1fr}}.variants{{grid-template-columns:1fr}}}}</style><h1>Сравнение текстовой обработки мемов</h1><p class='notice'>{cell(banner)}</p>{''.join(cards)}</html>"""


def save(items: list[dict], key_missing: bool) -> None:
    OUTPUT.mkdir(exist_ok=True)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "not_run_missing_key" if key_missing else "completed_with_possible_errors", "models": MODELS, "pricing_usd_per_million_text_tokens": RATES, "summary": summary(items), "images": items}
    (OUTPUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "comparison.html").write_text(render(items, key_missing), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Показать локальные изображения без вызовов API")
    args = parser.parse_args()
    if not INPUT.is_dir():
        parser.error(f"Папка не найдена: {INPUT}")
    files = image_files()
    if not files:
        parser.error("В test-input нет поддерживаемых изображений")
    key = load_key()
    if args.render_only or not key:
        items = [{"filename": path.name, "variants": {name: fail_record(model, RuntimeError("API-тест не выполнен")) for name, model in MODELS.items()}} for path in files]
        save(items, True)
        print(f"API-тест не выполнен. Подготовлены {OUTPUT / 'comparison.html'} и results.json для {len(files)} изображений.")
        print("Для запуска заполните OPENAI_API_KEY в окружении или локальном игнорируемом .env.")
        return 0 if args.render_only else 2
    items = []
    for index, path in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        items.append(evaluate(path, key))
        save(items, False)
    print(f"Готово: {OUTPUT / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
