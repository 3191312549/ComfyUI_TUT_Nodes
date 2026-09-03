"""Portable font discovery and loading for TUT_Nodes."""

from __future__ import annotations

import os
import platform
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_FONT_DIR = PLUGIN_ROOT / "fonts"
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
DEFAULT_FONT_TOKEN = "default/Pillow"


@dataclass(frozen=True)
class FontMetadata:
    family: str
    style: str
    postscript_name: str


@dataclass(frozen=True)
class FontPreviewAsset:
    path: Path | None
    data: bytes | None
    content_type: str
    filename: str


def _name_value(name_table, name_id: int) -> str:
    value = name_table.getDebugName(name_id)
    return str(value).strip() if value else ""


def _system_font_roots() -> list[Path]:
    system = platform.system()
    if system == "Windows":
        roots = []
        windows_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if windows_dir:
            roots.append(Path(windows_dir) / "Fonts")
        if local_app_data:
            roots.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
        return roots
    if system == "Darwin":
        return [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
    return [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local" / "share" / "fonts",
    ]


def _font_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS),
            key=lambda path: str(path).casefold(),
        )
    except OSError:
        return []


@lru_cache(maxsize=1)
def font_catalog() -> dict[str, Path | None]:
    """Return stable UI tokens mapped to real font files."""

    catalog: dict[str, Path | None] = {DEFAULT_FONT_TOKEN: None}

    for path in _font_files(BUNDLED_FONT_DIR):
        relative = path.relative_to(BUNDLED_FONT_DIR).as_posix()
        catalog[f"builtin/{relative}"] = path

    for root in _system_font_roots():
        for path in _font_files(root):
            relative = path.relative_to(root).as_posix()
            token = f"system/{relative}"
            if token in catalog and catalog[token] != path:
                root_key = str(root.resolve()).casefold().encode("utf-8", errors="replace")
                digest = hashlib.sha1(root_key).hexdigest()[:8]
                token = f"system/{root.name}-{digest}/{relative}"
                suffix = 2
                base_token = token
                while token in catalog and catalog[token] != path:
                    token = f"{base_token}-{suffix}"
                    suffix += 1
            catalog[token] = path

    return catalog


def font_options() -> tuple[str, ...]:
    catalog = font_catalog()
    bundled = sorted((key for key in catalog if key.startswith("builtin/")), key=str.casefold)
    system = sorted((key for key in catalog if key.startswith("system/")), key=str.casefold)
    return tuple(bundled + system + [DEFAULT_FONT_TOKEN])


def resolve_font(font_name: str) -> Path | None:
    catalog = font_catalog()
    if font_name in catalog:
        return catalog[font_name]

    candidate = Path(font_name).expanduser()
    if candidate.is_file() and candidate.suffix.lower() in FONT_EXTENSIONS:
        return candidate.resolve()

    basename = candidate.name.casefold()
    matches = [path for path in catalog.values() if path is not None and path.name.casefold() == basename]
    if matches:
        return matches[0]

    raise ValueError(
        f"找不到字体 '{font_name}'。请将字体放入 TUT_Nodes/fonts，或在系统中安装后重启 ComfyUI。"
    )


@lru_cache(maxsize=1024)
def _font_codepoints(font_name: str) -> frozenset[int] | None:
    """Return the first font face's Unicode coverage when fontTools is available."""

    path = resolve_font(font_name)
    if path is None:
        return None
    try:
        from fontTools.ttLib import TTFont, TTLibError
    except ImportError:
        return None
    try:
        with path.open("rb") as stream:
            font = TTFont(stream, fontNumber=0, lazy=True)
            try:
                cmap = font.getBestCmap() or {}
                return frozenset(cmap)
            finally:
                font.close()
    except (KeyError, OSError, ValueError, TTLibError):
        return None


def _required_codepoints(text: str) -> frozenset[int]:
    return frozenset(ord(character) for character in str(text) if not character.isspace())


@lru_cache(maxsize=512)
def font_for_text(font_name: str, text: str) -> str:
    """Choose a catalog font that can render the text, preserving the requested token when possible."""

    required = _required_codepoints(text)
    if not required:
        return font_name
    coverage = _font_codepoints(font_name)
    if coverage is None or required.issubset(coverage):
        return font_name

    preferred_names = (
        "misans", "microsoft yahei", "yahei", "noto sans cjk", "source han sans",
        "pingfang", "simhei", "simsun", "wenquanyi",
    )
    candidates = []
    for token in font_options():
        name = token.casefold()
        rank = next((index for index, marker in enumerate(preferred_names) if marker in name), len(preferred_names))
        candidates.append((rank, name, token))
    for _, _, token in sorted(candidates):
        candidate_coverage = _font_codepoints(token)
        if candidate_coverage is not None and required.issubset(candidate_coverage):
            return token
    return font_name


@lru_cache(maxsize=512)
def font_metadata(font_name: str) -> FontMetadata:
    """Read portable font names with fontTools, falling back to the file name."""

    path = resolve_font(font_name)
    if path is None:
        return FontMetadata("Pillow 默认字体", "Regular", "")
    family, style, postscript_name = path.stem, "", ""
    try:
        from fontTools.ttLib import TTFont, TTLibError
    except ImportError:
        return FontMetadata(family, style, postscript_name)

    try:
        # Passing an owned stream prevents malformed system fonts from leaving
        # a file descriptor open when TTFont raises during construction.
        with path.open("rb") as stream:
            font = TTFont(stream, fontNumber=0, lazy=False)
            try:
                names = font["name"]
                family = _name_value(names, 16) or _name_value(names, 1) or family
                style = _name_value(names, 17) or _name_value(names, 2)
                postscript_name = _name_value(names, 6)
            finally:
                font.close()
    except (KeyError, OSError, ValueError, TTLibError):
        pass
    return FontMetadata(family, style, postscript_name)


def font_display_name(font_name: str) -> str:
    metadata = font_metadata(font_name)
    return " ".join(part for part in (metadata.family, metadata.style) if part).strip() or font_name


def _font_source(font_name: str) -> str:
    if font_name.startswith("builtin/"):
        return "插件"
    if font_name.startswith("system/"):
        return "系统"
    return "内置"


@lru_cache(maxsize=1)
def font_ui_catalog() -> tuple[dict[str, str], ...]:
    """Return browser-safe labels while preserving stable font tokens."""

    records = []
    for token in font_options():
        metadata = font_metadata(token)
        if token == DEFAULT_FONT_TOKEN:
            try:
                default_family, default_style = ImageFont.load_default().getname()
                name = f"Pillow 默认字体（{default_family} {default_style}）"
            except (AttributeError, OSError, TypeError, ValueError):
                name = "Pillow 默认字体"
        else:
            name = " ".join(part for part in (metadata.family, metadata.style) if part).strip() or token
        source = _font_source(token)
        base_label = f"{name} · {source}"
        records.append((token, metadata, source, base_label))

    collisions = Counter(label.casefold() for _, _, _, label in records)
    result = []
    for token, metadata, source, base_label in records:
        label = base_label
        path = font_catalog().get(token)
        if collisions[base_label.casefold()] > 1 and path is not None:
            label = f"{base_label} · {path.name}"
        family_key = hashlib.sha1(token.encode("utf-8", errors="replace")).hexdigest()[:12]
        result.append({
            "token": token,
            "display_name": label,
            "family": metadata.family,
            "style": metadata.style,
            "source": source,
            "search_text": " ".join((label, token, metadata.family, metadata.style, metadata.postscript_name)),
            "preview_family": f"TUTComicFont_{family_key}",
        })
    return tuple(result)


@lru_cache(maxsize=128)
def font_preview_asset(font_name: str) -> FontPreviewAsset:
    """Resolve a catalog token to a browser font without accepting arbitrary paths."""

    catalog = font_catalog()
    if font_name not in catalog:
        raise ValueError("字体预览 token 无效")
    path = catalog[font_name]
    if path is None:
        default_font = ImageFont.load_default()
        source = getattr(default_font, "path", None)
        data = source.getvalue() if hasattr(source, "getvalue") else None
        if not data:
            raise ValueError("无法读取 Pillow 默认字体数据")
        return FontPreviewAsset(None, bytes(data), "font/ttf", "pillow-default.ttf")
    suffix = path.suffix.lower()
    if suffix == ".ttc":
        try:
            from fontTools.ttLib import TTFont, TTLibError
        except ImportError as exc:
            raise ValueError("TTC 字体预览需要 fontTools") from exc

        try:
            buffer = io.BytesIO()
            font = TTFont(str(path), fontNumber=0, lazy=False)
            try:
                font.save(buffer)
            finally:
                font.close()
            return FontPreviewAsset(None, buffer.getvalue(), "font/ttf", f"{path.stem}.ttf")
        except (KeyError, OSError, ValueError, TTLibError) as exc:
            raise ValueError(f"TTC 字体无法转换为浏览器预览：{path.name}") from exc
    content_type = "font/otf" if suffix == ".otf" else "font/ttf"
    return FontPreviewAsset(path, None, content_type, path.name)


def load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = resolve_font(font_name)
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size=max(1, int(size)))
        except OSError as exc:
            raise ValueError(f"字体无法读取：{path}") from exc

    try:
        return ImageFont.load_default(size=max(1, int(size)))
    except TypeError:
        return ImageFont.load_default()
