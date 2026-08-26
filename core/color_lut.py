"""RGB curve and LUT primitives with strict, batch-safe contracts."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .color import _cv2


CURVE_CHANNELS = ("RGB", "R", "G", "B")
MAX_CONTROL_POINTS = 16
LUT_SAMPLES = 1024
LUT_CHUNK_PIXELS = 1_048_576
MAX_3D_LUT_SIZE = 128
MAX_1D_LUT_SIZE = 65536
MAX_LUT_FILE_BYTES = 64 * 1024 * 1024
SUPPORTED_LUT_EXTENSIONS = (".cube", ".3dl", ".1dlut", ".png")


def _finite_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数字")
    return result


def _json_object(value, label):
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{label} 不是有效 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须是对象")
    if payload.get("version") != 1:
        raise ValueError(f"{label} version 必须为 1")
    return payload


def parse_curve_data(value):
    payload = _json_object(value, "曲线数据")
    channels = payload.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("曲线数据缺少 channels 对象")
    parsed = {}
    for channel in CURVE_CHANNELS:
        points = channels.get(channel)
        if not isinstance(points, list) or not 2 <= len(points) <= MAX_CONTROL_POINTS:
            raise ValueError(f"曲线 {channel} 必须包含 2 到 {MAX_CONTROL_POINTS} 个控制点")
        clean = []
        for index, point in enumerate(points):
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError(f"曲线 {channel} 第 {index + 1} 个控制点必须包含 x 和 y")
            x = _finite_float(point["x"], f"曲线 {channel} 的 x")
            y = _finite_float(point["y"], f"曲线 {channel} 的 y")
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(f"曲线 {channel} 的 x/y 必须在 0 到 1 之间")
            clean.append((x, y))
        clean.sort(key=lambda item: item[0])
        if any(abs(clean[i][0] - clean[i - 1][0]) <= 1e-7 for i in range(1, len(clean))):
            raise ValueError(f"曲线 {channel} 的 x 不能重复")
        if abs(clean[0][0]) > 1e-7 or abs(clean[-1][0] - 1.0) > 1e-7:
            raise ValueError(f"曲线 {channel} 必须包含 x=0 和 x=1 两个端点")
        parsed[channel] = np.asarray(clean, dtype=np.float32)
    return parsed


def _monotone_slopes(xs, ys):
    widths = np.diff(xs).astype(np.float64)
    secants = np.diff(ys).astype(np.float64) / widths
    slopes = np.empty_like(xs, dtype=np.float64)
    slopes[0], slopes[-1] = secants[0], secants[-1]
    if len(xs) > 2:
        for index in range(1, len(xs) - 1):
            left, right = secants[index - 1], secants[index]
            if left == 0.0 or right == 0.0 or left * right < 0.0:
                slopes[index] = 0.0
            else:
                w1 = 2.0 * widths[index] + widths[index - 1]
                w2 = widths[index] + 2.0 * widths[index - 1]
                slopes[index] = (w1 + w2) / (w1 / left + w2 / right)
    for index, secant in enumerate(secants):
        if secant == 0.0:
            slopes[index] = slopes[index + 1] = 0.0
            continue
        a, b = slopes[index] / secant, slopes[index + 1] / secant
        magnitude = a * a + b * b
        if magnitude > 9.0:
            scale = 3.0 / math.sqrt(magnitude)
            slopes[index] = scale * a * secant
            slopes[index + 1] = scale * b * secant
    return slopes


def curve_lut(points, interpolation, samples=LUT_SAMPLES):
    points = np.asarray(points, dtype=np.float64)
    xs, ys = points[:, 0], points[:, 1]
    query = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    if interpolation != "单调三次":
        raise ValueError(f"不支持的曲线插值方式：{interpolation!r}")
    if len(points) == 2:
        return np.interp(query, xs, ys).astype(np.float32)
    slopes = _monotone_slopes(xs, ys)
    intervals = np.clip(np.searchsorted(xs, query, side="right") - 1, 0, len(xs) - 2)
    x0, x1 = xs[intervals], xs[intervals + 1]
    y0, y1 = ys[intervals], ys[intervals + 1]
    width = x1 - x0
    t = (query - x0) / width
    t2, t3 = t * t, t * t * t
    values = ((2 * t3 - 3 * t2 + 1) * y0
              + (t3 - 2 * t2 + t) * width * slopes[intervals]
              + (-2 * t3 + 3 * t2) * y1
              + (t3 - t2) * width * slopes[intervals + 1])
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _sample_1d_lut(values, lut):
    coordinates = np.clip(values, 0.0, 1.0) * (len(lut) - 1)
    lower = np.floor(coordinates).astype(np.int32)
    upper = np.minimum(lower + 1, len(lut) - 1)
    fraction = coordinates - lower
    if np.asarray(lut).ndim > 1:
        fraction = fraction[..., None]
    return lut[lower] * (1.0 - fraction) + lut[upper] * fraction


def curves_are_identity(curves):
    for points in curves.values():
        if not np.allclose(points[:, 1], points[:, 0], rtol=0.0, atol=1e-7):
            return False
    return True


def apply_curves(image, curves, interpolation):
    result = np.asarray(image, dtype=np.float32).copy()
    combined = curve_lut(curves["RGB"], interpolation)
    result = _sample_1d_lut(result, combined).astype(np.float32)
    for channel_index, channel in enumerate(("R", "G", "B")):
        lut = curve_lut(curves[channel], interpolation)
        result[..., channel_index] = _sample_1d_lut(result[..., channel_index], lut)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _comfy_input_directory():
    try:
        import folder_paths

        return Path(folder_paths.get_input_directory())
    except (ImportError, AttributeError):
        return None


def lut_directories():
    roots = [("plugin", Path(__file__).resolve().parents[1] / "luts")]
    input_root = _comfy_input_directory()
    if input_root is not None:
        roots.append(("input", input_root / "TUT_Nodes" / "luts"))
    return roots


def list_lut_files():
    values = []
    for prefix, root in lut_directories():
        if not root.is_dir():
            continue
        try:
            files = sorted(
                (path for path in root.rglob("*")
                 if path.is_file() and path.suffix.lower() in SUPPORTED_LUT_EXTENSIONS),
                key=lambda path: str(path).casefold(),
            )
        except OSError:
            continue
        values.extend(f"{prefix}:{path.relative_to(root).as_posix()}" for path in files)
    return values


def resolve_lut_path(value):
    token = str(value).strip().strip('"').strip("'")
    if not token:
        raise ValueError("lut_path 不能为空")
    path = None
    for prefix, root in lut_directories():
        marker = f"{prefix}:"
        if token.startswith(marker):
            candidate = (root / token[len(marker):]).resolve()
            resolved_root = root.resolve()
            if candidate != resolved_root and resolved_root not in candidate.parents:
                raise ValueError("LUT 文件路径不能超出指定目录")
            path = candidate
            break
    if path is None:
        candidate = Path(token).expanduser()
        if candidate.is_absolute():
            path = candidate
        else:
            plugin_path = Path(__file__).resolve().parents[1] / "luts" / candidate
            input_root = _comfy_input_directory()
            input_path = input_root / candidate if input_root is not None else None
            path = plugin_path if plugin_path.exists() or input_path is None else input_path
    try:
        path = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"找不到 LUT 文件：{path}") from exc
    if not path.is_file():
        raise ValueError(f"LUT 路径不是文件：{path}")
    if path.suffix.lower() not in SUPPORTED_LUT_EXTENSIONS:
        raise ValueError("LUT 只支持 .cube、.3dl、.1dlut 和 PNG")
    try:
        if path.stat().st_size > MAX_LUT_FILE_BYTES:
            raise ValueError(f"LUT 文件不能超过 {MAX_LUT_FILE_BYTES // (1024 * 1024)} MB")
    except OSError as exc:
        raise ValueError(f"无法读取 LUT 文件信息：{path}") from exc
    return path


def _read_lut_lines(path, label):
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} 文件不是有效 UTF-8 文本：{path}") from exc


def _parse_cube(path):
    size_1d = None
    size_3d = None
    domain_min = np.zeros(3, dtype=np.float32)
    domain_max = np.ones(3, dtype=np.float32)
    entries = []
    for line_number, raw_line in enumerate(_read_lut_lines(path, ".cube"), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        parts = line.split()
        try:
            if upper.startswith("TITLE"):
                continue
            if parts[0].upper() in ("LUT_1D_SIZE", "LUT_3D_SIZE"):
                if len(parts) != 2:
                    raise ValueError(f"{parts[0]} 格式错误")
                if parts[0].upper() == "LUT_1D_SIZE":
                    size_1d = int(parts[1])
                else:
                    size_3d = int(parts[1])
                continue
            if parts[0].upper() in ("DOMAIN_MIN", "DOMAIN_MAX"):
                if len(parts) != 4:
                    raise ValueError(f"{parts[0]} 必须包含三个数值")
                target = np.asarray([float(item) for item in parts[1:]], dtype=np.float32)
                if not np.all(np.isfinite(target)):
                    raise ValueError(f"{parts[0]} 包含非有限数值")
                if parts[0].upper() == "DOMAIN_MIN":
                    domain_min = target
                else:
                    domain_max = target
                continue
            if len(parts) != 3:
                raise ValueError("数据行必须包含三个数值")
            entry = [float(item) for item in parts]
            if not np.all(np.isfinite(entry)):
                raise ValueError("数据行包含非有限数值")
            entries.append(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(f".cube 第 {line_number} 行无效：{exc}") from exc
    if size_1d is not None and size_3d is not None:
        raise ValueError("暂不支持同时包含 1D 与 3D 表的复合 .cube")
    if size_1d is not None:
        if not 2 <= size_1d <= MAX_1D_LUT_SIZE:
            raise ValueError(f"LUT_1D_SIZE 必须在 2 到 {MAX_1D_LUT_SIZE} 之间")
        size, kind, expected = size_1d, "1d", size_1d
    elif size_3d is not None:
        if not 2 <= size_3d <= MAX_3D_LUT_SIZE:
            raise ValueError(f"LUT_3D_SIZE 必须在 2 到 {MAX_3D_LUT_SIZE} 之间")
        size, kind, expected = size_3d, "3d", size_3d ** 3
    else:
        raise ValueError(".cube 缺少 LUT_1D_SIZE 或 LUT_3D_SIZE")
    if len(entries) != expected:
        raise ValueError(f".cube 数据量错误：需要 {expected} 行，实际 {len(entries)} 行")
    if np.any(domain_max <= domain_min):
        raise ValueError("DOMAIN_MAX 每个通道都必须大于 DOMAIN_MIN")
    lut = np.asarray(entries, dtype=np.float32)
    if kind == "3d":
        lut = lut.reshape(size, size, size, 3)
    return kind, lut, domain_min, domain_max, {}


def _parse_1dlut(path):
    declared_size = None
    entries = []
    for line_number, raw_line in enumerate(_read_lut_lines(path, ".1dlut"), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        parts = line.replace(",", " ").split()
        upper = parts[0].upper()
        if upper in ("LUT_1D_SIZE", "LUT_SIZE", "SIZE"):
            if len(parts) != 2:
                raise ValueError(f".1dlut 第 {line_number} 行尺寸格式错误")
            declared_size = int(parts[1])
            continue
        try:
            values = [_finite_float(item, f".1dlut 第 {line_number} 行") for item in parts]
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if len(values) == 1:
            values *= 3
        if len(values) != 3:
            raise ValueError(f".1dlut 第 {line_number} 行必须包含一个或三个数值")
        entries.append(values)
    size = declared_size or len(entries)
    if not 2 <= size <= MAX_1D_LUT_SIZE:
        raise ValueError(f"1D LUT 尺寸必须在 2 到 {MAX_1D_LUT_SIZE} 之间")
    if len(entries) != size:
        raise ValueError(f".1dlut 数据量错误：需要 {size} 行，实际 {len(entries)} 行")
    table = np.asarray(entries, dtype=np.float32)
    return "1d", _normalise_lut_values(table), np.zeros(3, np.float32), np.ones(3, np.float32), {}


def _normalise_lut_values(table):
    table = np.asarray(table, dtype=np.float32)
    if not np.all(np.isfinite(table)):
        raise ValueError("LUT 数据包含非有限数值")
    maximum = float(table.max(initial=0.0))
    if maximum > 1.0:
        table = table / maximum
    return table.astype(np.float32)


def _parse_3dl(path):
    declared_size = None
    input_grid = None
    input_bits = None
    output_bits = None
    entries = []
    for line_number, raw_line in enumerate(_read_lut_lines(path, ".3dl"), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        parts = line.replace(",", " ").split()
        upper = parts[0].upper()
        if upper == "LUT_3D_SIZE":
            if len(parts) != 2:
                raise ValueError(".3dl 的 LUT_3D_SIZE 格式错误")
            declared_size = int(parts[1])
            continue
        if upper == "3DMESH":
            continue
        if upper == "MESH":
            if len(parts) != 3:
                raise ValueError(".3dl 的 Mesh 头必须包含输入与输出位深")
            input_bits, output_bits = int(parts[1]), int(parts[2])
            if not 1 <= input_bits <= 7 or not 1 <= output_bits <= 16:
                raise ValueError(".3dl 的 Mesh 位深超出支持范围")
            continue
        try:
            values = [_finite_float(item, f".3dl 第 {line_number} 行") for item in parts]
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not entries and input_grid is None and len(values) != 3 and len(values) >= 2:
            input_grid = np.asarray(values, dtype=np.float32)
            continue
        if len(values) != 3:
            raise ValueError(f".3dl 第 {line_number} 行必须包含三个数值")
        entries.append(values)
    size = declared_size or (2 ** input_bits + 1 if input_bits is not None else None)
    size = size or (
        len(input_grid) if input_grid is not None
        else round(len(entries) ** (1.0 / 3.0))
    )
    if not 2 <= size <= MAX_3D_LUT_SIZE:
        raise ValueError(f"3D LUT 尺寸必须在 2 到 {MAX_3D_LUT_SIZE} 之间")
    expected = size ** 3
    if len(entries) != expected:
        raise ValueError(f".3dl 数据量错误：需要 {expected} 行，实际 {len(entries)} 行")
    table = np.asarray(entries, dtype=np.float32)
    if not np.all(np.isfinite(table)) or float(table.min(initial=0.0)) < 0.0:
        raise ValueError(".3dl 输出必须是有限的非负数值")
    if output_bits is not None:
        divisor = float(2 ** output_bits - 1)
    elif float(table.max(initial=0.0)) <= 1.0:
        divisor = 1.0
    elif input_grid is not None and float(table.max()) <= float(input_grid[-1]):
        divisor = float(input_grid[-1])
    else:
        maximum = float(table.max())
        divisor = next((float(value) for value in (255, 1023, 4095, 16383, 65535) if value >= maximum), 0.0)
        if divisor == 0.0:
            raise ValueError(".3dl 输出范围无法确定或超过 16 bit")
    table = np.clip(table / max(divisor, 1.0), 0.0, 1.0)
    # Autodesk 3DL rows advance blue first, then green, then red.  The internal
    # contract is [blue, green, red, output_channel].
    table = table.reshape(size, size, size, 3).transpose(2, 1, 0, 3).astype(np.float32)
    metadata = {}
    if input_grid is not None:
        if len(input_grid) != size or np.any(np.diff(input_grid) <= 0.0):
            raise ValueError(".3dl 输入网格必须严格递增且数量与 LUT 尺寸一致")
        metadata["input_grid"] = input_grid
    return "3d", table, np.zeros(3, np.float32), np.ones(3, np.float32), metadata


def _parse_png(path):
    try:
        cv2 = _cv2()
        source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if source is None:
            raise ValueError("OpenCV 未返回图像")
        if source.ndim == 2:
            source = np.repeat(source[..., None], 3, axis=-1)
        if source.ndim != 3 or source.shape[-1] not in (3, 4):
            raise ValueError("PNG 必须包含 RGB 或 RGBA 通道")
        source = source[..., :3][..., ::-1]
        if source.dtype == np.uint8:
            image = source.astype(np.float32) / 255.0
        elif source.dtype == np.uint16:
            image = source.astype(np.float32) / 65535.0
        else:
            image = np.clip(source.astype(np.float32), 0.0, 1.0)
    except Exception as exc:
        raise ValueError(f"无法读取 Hald PNG：{path}；{exc}") from exc
    height, width = image.shape[:2]
    if width == height:
        level = round(width ** (1.0 / 3.0))
        if level < 2 or level ** 3 != width:
            raise ValueError("Hald PNG 边长必须是 level³")
        size = level ** 2
        lut = image.reshape(size, size, size, 3)
    elif width == height * height and height >= 2:
        size = height
        lut = image.reshape(size, size, size, 3).transpose(1, 0, 2, 3)
    else:
        raise ValueError("PNG 必须是标准方形 Hald 或 width=height² 的 ReShade LUT")
    if size > MAX_3D_LUT_SIZE:
        raise ValueError(f"PNG 3D LUT 尺寸不能超过 {MAX_3D_LUT_SIZE}")
    return "3d", lut.astype(np.float32), np.zeros(3, np.float32), np.ones(3, np.float32), {}


@lru_cache(maxsize=16)
def _load_lut_cached(path_text, mtime_ns, file_size):
    del mtime_ns, file_size
    path = Path(path_text)
    parser = {
        ".cube": _parse_cube,
        ".3dl": _parse_3dl,
        ".1dlut": _parse_1dlut,
        ".png": _parse_png,
    }[path.suffix.lower()]
    return parser(path)


def load_lut_data(value):
    path = resolve_lut_path(value)
    stat = path.stat()
    kind, table, domain_min, domain_max, metadata = _load_lut_cached(
        str(path), stat.st_mtime_ns, stat.st_size
    )
    return {
        "kind": kind,
        "table": table,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "source": str(path),
        "fingerprint": f"{stat.st_mtime_ns}:{stat.st_size}",
        "format": path.suffix.lower().lstrip("."),
    } | metadata


def load_lut(value):
    data = load_lut_data(value)
    return data["table"], data["domain_min"], data["domain_max"]


def clear_lut_cache():
    _load_lut_cached.cache_clear()


def _trilinear_chunk(pixels, lut, domain_min, domain_max, input_grid=None):
    size = lut.shape[0]
    normalised = np.clip((pixels - domain_min) / (domain_max - domain_min), 0.0, 1.0)
    if input_grid is None:
        coordinates = normalised * (size - 1)
    else:
        grid = np.asarray(input_grid, dtype=np.float32)
        query = normalised * grid[-1]
        lower_grid = np.clip(np.searchsorted(grid, query, side="right") - 1, 0, size - 2)
        upper_grid = lower_grid + 1
        fraction_grid = (query - grid[lower_grid]) / (grid[upper_grid] - grid[lower_grid])
        coordinates = lower_grid + fraction_grid
    lower = np.floor(coordinates).astype(np.int32)
    upper = np.minimum(lower + 1, size - 1)
    fraction = coordinates - lower
    r0, g0, b0 = lower[:, 0], lower[:, 1], lower[:, 2]
    r1, g1, b1 = upper[:, 0], upper[:, 1], upper[:, 2]
    fr, fg, fb = fraction[:, 0:1], fraction[:, 1:2], fraction[:, 2:3]
    c000 = lut[b0, g0, r0]; c100 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]; c110 = lut[b0, g1, r1]
    c001 = lut[b1, g0, r0]; c101 = lut[b1, g0, r1]
    c011 = lut[b1, g1, r0]; c111 = lut[b1, g1, r1]
    c00 = c000 * (1.0 - fr) + c100 * fr
    c10 = c010 * (1.0 - fr) + c110 * fr
    c01 = c001 * (1.0 - fr) + c101 * fr
    c11 = c011 * (1.0 - fr) + c111 * fr
    c0 = c00 * (1.0 - fg) + c10 * fg
    c1 = c01 * (1.0 - fg) + c11 * fg
    return c0 * (1.0 - fb) + c1 * fb


def apply_3d_lut(image, lut, domain_min, domain_max, chunk_pixels=LUT_CHUNK_PIXELS, input_grid=None):
    flat = np.asarray(image, dtype=np.float32).reshape(-1, 3)
    output = np.empty_like(flat)
    for start in range(0, len(flat), int(chunk_pixels)):
        end = min(start + int(chunk_pixels), len(flat))
        output[start:end] = _trilinear_chunk(
            flat[start:end], lut, domain_min, domain_max, input_grid=input_grid
        )
    return np.clip(output.reshape(image.shape), 0.0, 1.0).astype(np.float32)


def apply_1d_lut(image, lut, domain_min, domain_max):
    frame = np.asarray(image, dtype=np.float32)
    coordinates = np.clip((frame - domain_min) / (domain_max - domain_min), 0.0, 1.0)
    result = np.empty_like(frame)
    for channel in range(3):
        result[..., channel] = _sample_1d_lut(coordinates[..., channel], lut[:, channel])
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def validate_lut_data(data):
    if not isinstance(data, dict):
        raise ValueError("lut_data 必须来自 TUT_LUT加载与预览节点")
    kind = data.get("kind")
    table = data.get("table")
    domain_min = np.asarray(data.get("domain_min"), dtype=np.float32)
    domain_max = np.asarray(data.get("domain_max"), dtype=np.float32)
    if kind not in ("1d", "3d") or not isinstance(table, np.ndarray):
        raise ValueError("lut_data 内容无效或版本不兼容")
    if domain_min.shape != (3,) or domain_max.shape != (3,) or np.any(domain_max <= domain_min):
        raise ValueError("lut_data 的输入域无效")
    if not np.all(np.isfinite(table)):
        raise ValueError("lut_data 包含非有限数值")
    if kind == "1d" and (table.ndim != 2 or table.shape[1] != 3):
        raise ValueError("1D lut_data 的表格形状无效")
    if kind == "3d" and (table.ndim != 4 or table.shape[-1] != 3 or len(set(table.shape[:3])) != 1):
        raise ValueError("3D lut_data 的表格形状无效")
    return kind, table.astype(np.float32, copy=False), domain_min, domain_max


def apply_lut_data(image, data):
    kind, table, domain_min, domain_max = validate_lut_data(data)
    if kind == "1d":
        return apply_1d_lut(image, table, domain_min, domain_max)
    return apply_3d_lut(
        image, table, domain_min, domain_max, input_grid=data.get("input_grid")
    )

