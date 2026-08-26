"""Portable font discovery and loading for TUT_Nodes."""

from __future__ import annotations

import os
import platform
import hashlib
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


@lru_cache(maxsize=512)
def font_metadata(font_name: str) -> FontMetadata:
    """Read portable font names with fontTools, falling back to the file name."""

    path = resolve_font(font_name)
    if path is None:
        return FontMetadata("Pillow 默认字体", "Regular", "")
    family, style, postscript_name = path.stem, "", ""
    try:
        from fontTools.ttLib import TTFont

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
    except (ImportError, KeyError, OSError, ValueError):
        pass
    return FontMetadata(family, style, postscript_name)


def font_display_name(font_name: str) -> str:
    metadata = font_metadata(font_name)
    return " ".join(part for part in (metadata.family, metadata.style) if part).strip() or font_name


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
