"""Отдельный эксперимент по выбору текста мема и беларусскому переводу."""

from __future__ import annotations

import argparse
import base64
import hashlib
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
RATES = {"gpt-5.6-luna": (0.20, 1.20), "gpt-5.6-terra": (2.00, 12.00), "gpt-5.6-sol": (4.00, 20.00)}
D_RESULTS = OUTPUT / "terra-sol-results.json"
D_HTML = OUTPUT / "terra-sol-comparison.html"
E_RESULTS = OUTPUT / "terra-sol-conservative-results.json"
E_HTML = OUTPUT / "terra-sol-conservative-comparison.html"
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
CONSERVATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_translation": {"type": "string"},
        "changed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["final_translation", "changed", "reason"],
    "additionalProperties": False,
}
VISION_PROMPT = """Рассмотри изображение как мем. Выбери только текст, который пользователь захотел бы заново разместить как содержание мема. Это смысловая, а не OCR-задача: не копируй весь видимый текст. Игнорируй бренды, логотипы, вывески в сцене, интерфейс, системные подписи и случайный фон, если они не несут шутку. У твита/поста выбирай текст по роли: если мемом служит добавленный комментарий, исходный текст поста игнорируй; если сам текст поста несёт мем, выбирай его. Сохраняй порядок отдельных сегментов в target_text. В ignored_text укажи замеченный, но отвергнутый текст, а в ignored_reason кратко объясни выбор. При сомнении между несколькими трактовками выставь ambiguous=true; не угадывай молча. Если содержательного текста нет, верни пустые target_text и translation. Переведи выбранный текст на естественный беларусский без русизмов, с правильными падежами, смыслом, шуткой и разговорным регистром. Не добавляй содержание и не переводи имена и бренды механически. Сохрани нормальную пунктуацию."""
EDITOR_PROMPT = """Ты редактор беларусского перевода. Получаешь только выбранный исходный текст мема и черновой перевод. Проверь смысл, шутку, разговорный регистр, грамматику и отсутствие русизмов. Исправь лишь необходимое. Если всё хорошо, оставь текст без изменений. Не добавляй нового содержания. Если исходный текст пуст, верни пустой перевод. Сохрани обычную пунктуацию."""
SOL_EDITOR_PROMPT = """Ты редактор беларусского перевода мема. Тебе переданы только уже выбранный исходный текст мема и черновой беларусский перевод. Изображения у тебя нет. Проверь соответствие смысла исходному тексту, русизмы, грамматику, естественность и разговорный регистр. Исправляй лишь необходимое; если перевод хорош, верни его без изменений. Не добавляй содержание, не меняй выбор текста и не пытайся заново распознавать изображение. Сохраняй обычную пунктуацию."""
CONSERVATIVE_PROMPT = """Ты не создаёшь новый перевод, а проверяешь уже хороший беларусский перевод мема. У тебя есть только исходный выбранный текст и перевод Terra, изображения нет. Меняй текст только по конкретной причине: потерян или добавлен смысл; неверно переведено слово или конструкция; грамматическая или падежная ошибка; явный русизм; неестественная для беларусского языка конструкция; потерян важный оттенок оригинала. Если перевод корректный, верни его БЕЗ ИЗМЕНЕНИЙ. Не заменяй нормальное беларусское слово другим из стилистических предпочтений. Не делай текст литературнее или разговорнее оригинала. Сохраняй степень грубости, сленга и мата; не смягчай и не цензурируй мат и грубую лексику. Сохраняй мемный и разговорный регистр, но не добавляй сленг к нейтральному исходнику. Не исправляй нормальную беларусскую форму только потому, что есть другой допустимый вариант. Не меняй имена собственные и бренды без необходимости. Не меняй выбор target_text и не пытайся заново распознавать изображение. Если исправил, укажи короткую конкретную причину; если не исправил, reason должен быть пустой строкой, changed=false, а final_translation точной копией черновика."""


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


def sol_editor_body(target_text: list[str], draft: str) -> dict:
    source = json.dumps({"target_text": target_text, "draft_translation": draft}, ensure_ascii=False)
    return request_body("gpt-5.6-sol", [{"type": "input_text", "text": source}], EDITOR_SCHEMA, "sol_edited_translation", SOL_EDITOR_PROMPT)


def conservative_editor_body(target_text: list[str], draft: str) -> dict:
    source = json.dumps({"target_text": target_text, "terra_translation": draft}, ensure_ascii=False)
    return request_body("gpt-5.6-sol", [{"type": "input_text", "text": source}], CONSERVATIVE_SCHEMA, "conservative_translation_review", CONSERVATIVE_PROMPT)


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


def terra_sources(data: dict) -> list[dict]:
    """Взять только готовые результаты C из сохранённого первого эксперимента."""
    images = data.get("images")
    if not isinstance(images, list) or len(images) != 15:
        raise ValueError("Ожидалось ровно 15 сохранённых изображений в results.json")
    sources = []
    for item in images:
        record = item.get("variants", {}).get("C", {})
        if record.get("error") or record.get("model") != "gpt-5.6-terra":
            raise ValueError(f"Нет успешного результата Terra C: {item.get('filename')}")
        value = validate_vision(record.get("raw_structured") or {})
        sources.append({"filename": item["filename"], "target_text": value["target_text"], "terra_translation": value["translation"], "terra_usage": record.get("usage")})
    return sources


def sol_summary(items: list[dict]) -> dict:
    completed = [item for item in items if (item.get("sol_record") or {}).get("raw_structured") and not item["sol_record"].get("error")]
    costs = [item["sol_record"].get("cost", {}) for item in items if (item.get("sol_record") or {}).get("usage")]
    complete = len(completed) == len(items) and len(costs) == len(items) and all(value.get("complete") for value in costs)
    return {
        "completed_reviews": len(completed),
        "changed_translations": sum(item["terra_translation"] != item["sol_record"]["raw_structured"]["translation"] for item in completed),
        "cost_complete": complete,
        "total_review_cost_usd": sum(value["known_subtotal_usd"] for value in costs) if complete else None,
        "known_subtotal_usd": sum(value.get("known_subtotal_usd") or 0 for value in costs),
    }


def render_sol_comparison(items: list[dict]) -> str:
    cards = []
    for item in items:
        record = item.get("sol_record") or {}
        edited = (record.get("raw_structured") or {}).get("translation")
        changed = "да" if edited is not None and edited != item["terra_translation"] else "нет" if edited is not None else "—"
        amount = record.get("cost", {}).get("known_subtotal_usd")
        price = f"${amount:.6f}" if amount is not None else "—"
        image_src = "../test-input/" + urllib.request.pathname2url(item["filename"])
        cards.append(f"<section class='card'><h2>{cell(item['filename'])}</h2><div class='layout'><img src='{html.escape(image_src, quote=True)}' alt='{cell(item['filename'])}'><div><p><b>target_text</b><br>{cell(item['target_text'])}</p><div class='pair'><div><h3>C · Terra</h3><p>{cell(item['terra_translation'])}</p><small>normalized: {cell(normalize(item['terra_translation']))}</small></div><div><h3>D · Terra + Sol</h3><p>{cell(edited)}</p><small>normalized: {cell(normalize(edited or ''))}</small></div></div><p><b>Sol изменил перевод:</b> {changed} · <b>Стоимость Sol-review:</b> {price}</p><p><b>usage Sol:</b> <code>{cell(json.dumps(record.get('usage'), ensure_ascii=False))}</code></p><p><b>Ошибка:</b> {cell(record.get('error'))}</p></div></div></section>")
    s = sol_summary(items)
    total = f"${s['total_review_cost_usd']:.6f}" if s["cost_complete"] else "неполная оценка"
    return f"""<!doctype html><html lang='ru'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Terra и Terra + Sol</title><style>body{{font:15px/1.45 system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}}.card{{background:#fff;margin:20px 0;padding:18px;border-radius:10px;box-shadow:0 2px 8px #0001}}.layout{{display:grid;grid-template-columns:minmax(220px,32%) 1fr;gap:18px}}img{{max-width:100%;max-height:600px;object-fit:contain;align-self:start}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.pair>div{{background:#f7f9fc;padding:12px;border-radius:8px;min-width:0}}.pair p{{white-space:pre-wrap;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}small{{color:#555}}@media(max-width:850px){{.layout,.pair{{grid-template-columns:1fr}}}}</style><h1>Terra и текстовый review Sol</h1><p>Проверено {s['completed_reviews']} из {len(items)} · изменено переводов: {s['changed_translations']} · стоимость 15 Sol-review: {total}. Sol получал только текст из сохранённого результата Terra.</p>{''.join(cards)}</html>"""


def save_sol_results(data: dict) -> None:
    OUTPUT.mkdir(exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["summary"] = sol_summary(data["images"])
    temporary = D_RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(D_RESULTS)
    D_HTML.write_text(render_sol_comparison(data["images"]), encoding="utf-8")


def review_saved_terra(key: str | None) -> int:
    source_path = OUTPUT / "results.json"
    if not source_path.is_file():
        print("Не найден test-output/results.json с результатами Terra.", file=sys.stderr)
        return 2
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    sources = terra_sources(json.loads(source_bytes))
    if D_RESULTS.exists():
        data = json.loads(D_RESULTS.read_text(encoding="utf-8"))
        if data.get("source_sha256") != source_hash or [x.get("filename") for x in data.get("images", [])] != [x["filename"] for x in sources]:
            raise ValueError("Исходный results.json изменился после сохранения D; проверьте существующий terra-sol-results.json")
    else:
        data = {"source_file": "results.json", "source_sha256": source_hash, "model": "gpt-5.6-sol", "pricing_usd_per_million_text_tokens": {"input": 4.00, "output": 20.00}, "images": [{**source, "sol_record": None} for source in sources]}
    pending = [item for item in data["images"] if not item.get("sol_record") or item["sol_record"].get("error")]
    if pending and not key:
        print("Для текстового Sol-review нужен OPENAI_API_KEY в окружении или локальном .env.", file=sys.stderr)
        return 2
    for index, item in enumerate(pending, 1):
        print(f"Sol-review [{index}/{len(pending)}] {item['filename']}", flush=True)
        try:
            record = run_call(key, sol_editor_body(item["target_text"], item["terra_translation"]), False)
            edited = record.get("raw_structured")
            if edited is not None and not isinstance(edited.get("translation"), str):
                record["error"] = "Некорректный перевод Sol"
            item["sol_record"] = record
        except Exception as error:
            item["sol_record"] = fail_record("gpt-5.6-sol", error)
        save_sol_results(data)
    save_sol_results(data)
    print(f"Сохранено: {D_HTML}")
    print(json.dumps(data["summary"], ensure_ascii=False))
    return 0 if data["summary"]["completed_reviews"] == len(sources) else 1


def validate_conservative(value: dict, draft: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("final_translation"), str) or not isinstance(value.get("changed"), bool) or not isinstance(value.get("reason"), str):
        raise ValueError("Некорректные поля результата E")
    if value["changed"] != (value["final_translation"] != draft):
        raise ValueError("Поле changed не соответствует тексту перевода")
    if value["changed"] != bool(value["reason"].strip()):
        raise ValueError("Для правки нужна причина, без правки reason должен быть пустым")


def conservative_summary(items: list[dict]) -> dict:
    completed = [item for item in items if (item.get("conservative_record") or {}).get("raw_structured") and not item["conservative_record"].get("error")]
    costs = [item["conservative_record"].get("cost", {}) for item in items if (item.get("conservative_record") or {}).get("usage")]
    complete = len(completed) == len(items) and len(costs) == len(items) and all(x.get("complete") for x in costs)
    return {
        "completed_reviews": len(completed),
        "changed_translations": sum(item["conservative_record"]["raw_structured"]["changed"] for item in completed),
        "cost_complete": complete,
        "total_review_cost_usd": sum(x["known_subtotal_usd"] for x in costs) if complete else None,
        "known_subtotal_usd": sum(x.get("known_subtotal_usd") or 0 for x in costs),
    }


def render_conservative_comparison(items: list[dict]) -> str:
    cards = []
    for item in items:
        record = item.get("conservative_record") or {}
        value = record.get("raw_structured") or {}
        edited = value.get("final_translation")
        changed = value.get("changed") is True
        amount = record.get("cost", {}).get("known_subtotal_usd")
        price = f"${amount:.6f}" if amount is not None else "—"
        image_src = "../test-input/" + urllib.request.pathname2url(item["filename"])
        cards.append(f"<section class='card{' changed' if changed else ''}'><h2>{cell(item['filename'])}{' · ИЗМЕНЁН' if changed else ''}</h2><div class='layout'><img src='{html.escape(image_src, quote=True)}' alt='{cell(item['filename'])}'><div><p><b>target_text</b><br>{cell(item['target_text'])}</p><div class='triple'><div><h3>Terra</h3><p>{cell(item['terra_translation'])}</p></div><div><h3>Старый Sol-review</h3><p>{cell(item['old_sol_translation'])}</p></div><div class='new'><h3>Консервативный Sol-review</h3><p>{cell(edited)}</p></div></div><p><b>Изменил Terra:</b> {cell(value.get('changed'))} · <b>Причина:</b> {cell(value.get('reason'))}</p><p><b>Стоимость E:</b> {price} · <b>usage:</b> <code>{cell(json.dumps(record.get('usage'), ensure_ascii=False))}</code></p><p><b>Ошибка:</b> {cell(record.get('error'))}</p></div></div></section>")
    summary = conservative_summary(items)
    total = f"${summary['total_review_cost_usd']:.6f}" if summary["cost_complete"] else "неполная оценка"
    return f"""<!doctype html><html lang='ru'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Terra · старый Sol · консервативный Sol</title><style>body{{font:15px/1.45 system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}}.card{{background:#fff;margin:20px 0;padding:18px;border-radius:10px;box-shadow:0 2px 8px #0001}}.card.changed{{border:3px solid #da8520}}.layout{{display:grid;grid-template-columns:minmax(220px,25%) 1fr;gap:18px}}img{{max-width:100%;max-height:560px;object-fit:contain;align-self:start}}.triple{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.triple>div{{background:#f7f9fc;padding:12px;border-radius:8px;min-width:0}}.changed .new{{background:#fff0ce}}.triple p{{white-space:pre-wrap;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}@media(max-width:1100px){{.layout,.triple{{grid-template-columns:1fr}}}}</style><h1>Terra | старый Sol | консервативный Sol</h1><p>Проверено {summary['completed_reviews']} из {len(items)} · консервативный Sol изменил {summary['changed_translations']} переводов · стоимость E: {total}. Оранжевой рамкой отмечены отличия от Terra.</p>{''.join(cards)}</html>"""


def save_conservative_results(data: dict) -> None:
    OUTPUT.mkdir(exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["summary"] = conservative_summary(data["images"])
    temporary = E_RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(E_RESULTS)
    E_HTML.write_text(render_conservative_comparison(data["images"]), encoding="utf-8")


def review_terra_conservatively(key: str | None) -> int:
    original_path = OUTPUT / "results.json"
    if not original_path.is_file() or not D_RESULTS.is_file():
        print("Нужны сохранённые results.json и terra-sol-results.json.", file=sys.stderr)
        return 2
    original_bytes = original_path.read_bytes()
    old_bytes = D_RESULTS.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    sources = terra_sources(json.loads(original_bytes))
    old = json.loads(old_bytes)
    if old.get("source_sha256") != original_hash or len(old.get("images", [])) != len(sources):
        raise ValueError("Старый Sol-review не соответствует сохранённым результатам Terra")
    for source, previous in zip(sources, old["images"]):
        if previous.get("filename") != source["filename"] or previous.get("target_text") != source["target_text"] or previous.get("terra_translation") != source["terra_translation"]:
            raise ValueError(f"Не совпадает исходный перевод: {source['filename']}")
        old_record = previous.get("sol_record") or {}
        if old_record.get("error") or not isinstance((old_record.get("raw_structured") or {}).get("translation"), str):
            raise ValueError(f"Нет старого Sol-review: {source['filename']}")
        source["old_sol_translation"] = old_record["raw_structured"]["translation"]
    if E_RESULTS.exists():
        data = json.loads(E_RESULTS.read_text(encoding="utf-8"))
        if data.get("terra_source_sha256") != original_hash or data.get("old_sol_source_sha256") != old_hash:
            raise ValueError("Исходные сохранённые результаты изменились после запуска E")
    else:
        data = {"terra_source_file": "results.json", "terra_source_sha256": original_hash, "old_sol_source_file": D_RESULTS.name, "old_sol_source_sha256": old_hash, "model": "gpt-5.6-sol", "pricing_usd_per_million_text_tokens": {"input": 4.00, "output": 20.00}, "images": [{**source, "conservative_record": None} for source in sources]}
    pending = [item for item in data["images"] if not item.get("conservative_record") or item["conservative_record"].get("error")]
    if pending and not key:
        print("Для консервативного Sol-review нужен OPENAI_API_KEY.", file=sys.stderr)
        return 2
    save_conservative_results(data)
    for index, item in enumerate(pending, 1):
        print(f"Консервативный Sol-review [{index}/{len(pending)}] {item['filename']}", flush=True)
        try:
            record = run_call(key, conservative_editor_body(item["target_text"], item["terra_translation"]), False)
            value = record.get("raw_structured")
            if value is not None:
                try:
                    validate_conservative(value, item["terra_translation"])
                except ValueError as error:
                    record["error"] = str(error)
            item["conservative_record"] = record
        except Exception as error:
            item["conservative_record"] = fail_record("gpt-5.6-sol", error)
        save_conservative_results(data)
    print(f"Сохранено: {E_HTML}")
    print(json.dumps(data["summary"], ensure_ascii=False))
    return 0 if data["summary"]["completed_reviews"] == len(sources) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Показать локальные изображения без вызовов API")
    parser.add_argument("--review-terra-with-sol", action="store_true", help="Выполнить только текстовый Sol-review сохранённых результатов C без повторного анализа изображений")
    parser.add_argument("--review-terra-with-sol-conservative", action="store_true", help="Выполнить консервативный текстовый Sol-review сохранённых результатов C")
    args = parser.parse_args()
    if args.review_terra_with_sol_conservative:
        return review_terra_conservatively(load_key())
    if args.review_terra_with_sol:
        return review_saved_terra(load_key())
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
