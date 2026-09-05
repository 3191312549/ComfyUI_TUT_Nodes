"""Anima artist prompt composition and bundled artist-catalog search."""

from __future__ import annotations

from collections import OrderedDict
from difflib import SequenceMatcher
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata


ARTIST_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "anima_artists.sqlite3"
ARTIST_DATA_VERSION = 1
MIN_ARTIST_WEIGHT = 0.1
MAX_ARTIST_WEIGHT = 3.0

_PUNCTUATION_TRANSLATION = str.maketrans(
    {"，": ",", "、": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?", "（": "(", "）": ")"}
)
_WEIGHTED_ARTIST_RE = re.compile(
    r"^\(\s*(?P<tag>@.+):(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(r"^\d+\s*(?:girl|boy|other)s?$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^year\s+\d{4}$", re.IGNORECASE)
_SCORE_RE = re.compile(r"^score_[1-9]$", re.IGNORECASE)

_CONTROL_TAGS = {
    "masterpiece", "very aesthetic", "best quality", "good quality", "normal quality",
    "low quality", "worst quality", "highres", "absurdres", "anime screencap",
    "jpeg artifacts", "official art", "anime screenshot", "newest", "recent", "mid", "early", "old",
    "safe", "sensitive", "nsfw", "explicit",
}
_GENERIC_TAGS = {
    "solo", "smile", "frown", "blush", "open mouth", "closed mouth", "crying",
    "looking at viewer", "looking away", "closed eyes", "one eye closed", "standing",
    "sitting", "kneeling", "lying", "walking", "running", "jumping", "wariza",
    "indoors", "outdoors", "simple background", "white background", "black background",
    "city", "street", "classroom", "bedroom", "forest", "beach", "sky", "night",
    "day", "sunset", "rain", "snow", "wind", "backlighting", "rim light",
    "depth of field", "blurry background", "close-up", "upper body", "cowboy shot",
    "full body", "wide shot", "from above", "from below", "from behind",
}
_GENERIC_WORDS = {
    "hair", "eyes", "dress", "shirt", "skirt", "pants", "shorts", "uniform", "jacket",
    "coat", "sweater", "hoodie", "hat", "gloves", "shoes", "boots", "stockings",
    "ribbon", "bow", "necklace", "earrings", "weapon", "sword", "gun", "staff",
    "holding", "sitting", "standing", "kneeling", "lying", "walking", "running",
    "smile", "blush", "background", "lighting", "light", "shadow", "view", "shot",
}


def normalize_prompt_punctuation(value: object) -> str:
    """Convert common Chinese punctuation and normalize top-level tag separators."""

    text = str(value or "").translate(_PUNCTUATION_TRANSLATION)
    return ", ".join(split_top_level_commas(text))


def split_top_level_commas(text: str) -> list[str]:
    """Split commas outside brackets and quotes while preserving nested content."""

    parts: list[str] = []
    buffer: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    quoted = False
    escaped = False
    for character in str(text):
        if escaped:
            buffer.append(character)
            escaped = False
            continue
        if character == "\\":
            buffer.append(character)
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
        elif not quoted and character in depths:
            depths[character] += 1
        elif not quoted and character in closing:
            opener = closing[character]
            depths[opener] = max(0, depths[opener] - 1)
        if character == "," and not quoted and not any(depths.values()):
            part = "".join(buffer).strip()
            if part:
                parts.append(part)
            buffer = []
        else:
            buffer.append(character)
    part = "".join(buffer).strip()
    if part:
        parts.append(part)
    return parts


def _validate_weight(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("画师权重必须是数字。")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("画师权重必须是数字。") from exc
    if not math.isfinite(weight):
        raise ValueError("画师权重必须是有限数字。")
    if not MIN_ARTIST_WEIGHT <= weight <= MAX_ARTIST_WEIGHT:
        raise ValueError("画师权重必须在 0.1 到 3.0 之间。")
    rounded = round(weight * 10) / 10
    if abs(weight - rounded) > 1e-7:
        raise ValueError("画师权重必须使用 0.1 的步长。")
    return rounded


def _clean_artist_name(value: object) -> str:
    name = str(value or "").translate(_PUNCTUATION_TRANSLATION).strip()
    if name.startswith("@"):
        name = name[1:].strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        raise ValueError("画师名称不能为空。")
    if "," in name or "\n" in name or "\r" in name:
        raise ValueError("单个画师名称不能包含逗号或换行。")
    return name


def _artist_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold().strip()


@lru_cache(maxsize=1)
def _known_artist_keys() -> frozenset[str]:
    if not ARTIST_DATABASE_PATH.is_file():
        return frozenset()
    try:
        uri = f"file:{ARTIST_DATABASE_PATH.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            return frozenset(_artist_key(str(row[0])[1:]) for row in connection.execute("SELECT tag FROM artists"))
        finally:
            connection.close()
    except sqlite3.Error:
        return frozenset()


def parse_artist_segment(segment: str) -> dict[str, object] | None:
    """Parse a complete unweighted or weighted Anima artist tag segment."""

    value = str(segment).strip()
    if value.startswith("@"):
        return {"name": _clean_artist_name(value), "weight": 1.0}
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        if inner.startswith("@"):
            try:
                inner_name = _clean_artist_name(inner)
            except ValueError:
                inner_name = ""
            if inner_name and _artist_key(inner_name) in _known_artist_keys():
                return {"name": inner_name, "weight": 1.0}
    weighted = _WEIGHTED_ARTIST_RE.fullmatch(value)
    if weighted:
        name = _clean_artist_name(weighted.group("tag"))
        return {"name": name, "weight": _validate_weight(weighted.group("weight"))}
    return None


def parse_artist_data(value: object) -> list[dict[str, object]]:
    """Validate the stable JSON payload serialized by the capsule editor."""

    try:
        payload = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError("画师数据不是有效的 JSON。") from exc
    if (
        not isinstance(payload, dict)
        or isinstance(payload.get("version"), bool)
        or payload.get("version") != ARTIST_DATA_VERSION
    ):
        raise ValueError("画师数据版本无效。")
    artists = payload.get("artists")
    if not isinstance(artists, list):
        raise ValueError("画师数据中的 artists 必须是列表。")

    result: OrderedDict[str, dict[str, object]] = OrderedDict()
    for item in artists:
        if not isinstance(item, dict):
            raise ValueError("每个画师数据项必须是对象。")
        name = _clean_artist_name(item.get("name"))
        entry = {"name": name, "weight": _validate_weight(item.get("weight", 1.0))}
        result[_artist_key(name)] = entry
    return list(result.values())


def format_artist_tag(artist: dict[str, object]) -> str:
    name = _clean_artist_name(artist.get("name"))
    weight = _validate_weight(artist.get("weight", 1.0))
    return f"@{name}" if weight == 1.0 else f"(@{name}:{weight:.1f})"


def _is_control_tag(segment: str) -> bool:
    value = segment.casefold().strip()
    return value in _CONTROL_TAGS or bool(_YEAR_RE.fullmatch(value) or _SCORE_RE.fullmatch(value))


def _is_generic_or_natural(segment: str) -> bool:
    value = segment.casefold().strip()
    if value in _GENERIC_TAGS:
        return True
    if "\n" in value or re.search(r"[.!?]", value):
        return True
    words = set(re.findall(r"[a-z0-9]+", value))
    if words & _GENERIC_WORDS:
        return True
    ordered_words = re.findall(r"[a-z0-9]+", value)
    if len(ordered_words) >= 6 and words & {"a", "an", "the", "with", "in", "on", "at", "beside", "wearing", "stands", "sits"}:
        return True
    return len(re.findall(r"[\u3400-\u9fff]", value)) >= 8


def _smart_insertion_index(segments: list[str]) -> int:
    count_index = next((i for i, part in enumerate(segments) if _COUNT_RE.fullmatch(part.strip())), None)
    if count_index is not None:
        for index in range(count_index + 1, len(segments)):
            if _is_generic_or_natural(segments[index]):
                return index
        return len(segments)

    index = 0
    while index < len(segments) and _is_control_tag(segments[index]):
        index += 1
    return index


def mix_anima_artist_prompt(prompt: object, artist_data: object) -> str:
    """Normalize punctuation, merge artists, and place them in Anima tag order."""

    normalized = normalize_prompt_punctuation(prompt)
    segments = split_top_level_commas(normalized)
    editor_artists = parse_artist_data(artist_data)

    merged: OrderedDict[str, dict[str, object]] = OrderedDict()
    clean_segments: list[str] = []
    first_artist_index: int | None = None
    for segment in segments:
        artist = parse_artist_segment(segment)
        if artist is None:
            clean_segments.append(segment)
            continue
        if first_artist_index is None:
            first_artist_index = len(clean_segments)
        merged[_artist_key(str(artist["name"]))] = artist

    for artist in editor_artists:
        merged[_artist_key(str(artist["name"]))] = artist

    if not merged:
        return ", ".join(clean_segments)

    insertion_index = first_artist_index
    if insertion_index is None:
        insertion_index = _smart_insertion_index(clean_segments)
    artist_segments = [format_artist_tag(artist) for artist in merged.values()]
    result = clean_segments[:insertion_index] + artist_segments + clean_segments[insertion_index:]
    return ", ".join(result)


def normalize_artist_search(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).translate(_PUNCTUATION_TRANSLATION)
    text = text.strip().casefold().replace("_", " ")
    if text.startswith("@"):
        text = text[1:].strip()
    return re.sub(r"\s+", " ", text)


def fold_artist_search(value: object) -> str:
    normalized = normalize_artist_search(value)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _trigrams(value: str) -> set[str]:
    compact = f"  {value}  "
    return {compact[index:index + 3] for index in range(max(0, len(compact) - 2))}


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_payload(row: sqlite3.Row) -> dict[str, object]:
    return {
        "tag": row["tag"],
        "usage_count": int(row["usage_count"]),
        "style_description": row["style_description"] or "",
    }


def search_anima_artists(query: object = "", limit: object = 20, database_path: Path | None = None) -> list[dict[str, object]]:
    """Search the bundled artist catalog with exact-to-fuzzy deterministic ranking."""

    normalized = normalize_artist_search(query)
    if len(normalized) > 128:
        raise ValueError("画师搜索内容不能超过 128 个字符。")
    try:
        result_limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("搜索结果数量必须是整数。") from exc
    result_limit = max(1, min(50, result_limit))
    path = Path(database_path or ARTIST_DATABASE_PATH).resolve()
    if not path.is_file():
        raise RuntimeError(f"Anima 画师数据库不存在：{path}")

    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        with connection:
            if not normalized:
                rows = connection.execute(
                    "SELECT tag, normalized_name, folded_name, usage_count, style_description "
                    "FROM artists ORDER BY usage_count DESC, tag LIMIT ?",
                    (result_limit,),
                ).fetchall()
                return [_row_payload(row) for row in rows]

            folded = fold_artist_search(normalized)
            candidates: dict[int, sqlite3.Row] = {}
            pattern = f"%{_like_escape(normalized)}%"
            folded_pattern = f"%{_like_escape(folded)}%" if folded else pattern
            rows = connection.execute(
                "SELECT id, tag, normalized_name, folded_name, usage_count, style_description "
                "FROM artists WHERE normalized_name LIKE ? ESCAPE '\\' OR folded_name LIKE ? ESCAPE '\\' "
                "ORDER BY usage_count DESC LIMIT 300",
                (pattern, folded_pattern),
            ).fetchall()
            candidates.update({int(row["id"]): row for row in rows})

            if len(folded) >= 3:
                grams = sorted(_trigrams(folded))
                placeholders = ",".join("?" for _ in grams)
                fuzzy_rows = connection.execute(
                    f"SELECT a.id, a.tag, a.normalized_name, a.folded_name, a.usage_count, a.style_description, "
                    f"COUNT(*) AS overlap FROM artist_trigrams g JOIN artists a ON a.id = g.artist_id "
                    f"WHERE g.gram IN ({placeholders}) GROUP BY a.id ORDER BY overlap DESC, a.usage_count DESC LIMIT 300",
                    grams,
                ).fetchall()
                candidates.update({int(row["id"]): row for row in fuzzy_rows})
    except sqlite3.Error as exc:
        raise RuntimeError(f"读取 Anima 画师数据库失败：{exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    def rank(row: sqlite3.Row):
        name = str(row["normalized_name"])
        folded_name = str(row["folded_name"])
        if name == normalized:
            tier = 0
        elif name.startswith(normalized) or (folded and folded_name.startswith(folded)):
            tier = 1
        elif any(word.startswith(folded) for word in folded_name.split()) if folded else False:
            tier = 2
        elif normalized in name or (folded and folded in folded_name):
            tier = 3
        else:
            tier = 4
        similarity = SequenceMatcher(None, folded or normalized, folded_name or name).ratio()
        return tier, -similarity, -int(row["usage_count"]), str(row["tag"])

    ranked = sorted(candidates.values(), key=rank)
    filtered = [row for row in ranked if rank(row)[0] < 4 or -rank(row)[1] >= 0.42]
    return [_row_payload(row) for row in filtered[:result_limit]]
