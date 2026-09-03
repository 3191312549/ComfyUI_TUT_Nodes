"""Optional ComfyUI HTTP routes used by interactive TUT_Nodes controls."""

from __future__ import annotations

import asyncio


def register_excel_routes() -> bool:
    """Register the Excel inspection endpoint when a PromptServer is available."""

    try:
        from aiohttp import web
        from server import PromptServer
        from .core.excel import inspect_excel_workbook, list_excel_input_files, pick_local_excel_file
    except ImportError:
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return False
    routes = getattr(prompt_server, "routes", None)
    if routes is None or not hasattr(routes, "post"):
        return False
    if getattr(prompt_server, "_tut_excel_routes_registered", False):
        return True

    async def inspect_excel(request):
        try:
            payload = await request.json()
            excel_path = payload.get("excel_path", "") if isinstance(payload, dict) else ""
            return web.json_response(inspect_excel_workbook(excel_path))
        except (RuntimeError, ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def list_excel_files(_request):
        return web.json_response({"files": list_excel_input_files()})

    async def pick_excel_file(_request):
        try:
            selected = await asyncio.to_thread(pick_local_excel_file)
            return web.json_response(selected or {"cancelled": True})
        except (RuntimeError, ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    try:
        routes.post("/tut_nodes/excel/inspect")(inspect_excel)
        routes.get("/tut_nodes/excel/files")(list_excel_files)
        routes.post("/tut_nodes/excel/pick-local")(pick_excel_file)
    except (AttributeError, RuntimeError):
        return False

    prompt_server._tut_excel_routes_registered = True
    return True


def register_font_routes() -> bool:
    """Register read-only comic font metadata and font-data endpoints."""

    try:
        from aiohttp import web
        from server import PromptServer
        from .core.fonts import font_preview_asset, font_ui_catalog
    except ImportError:
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return False
    routes = getattr(prompt_server, "routes", None)
    if routes is None or not hasattr(routes, "get"):
        return False
    if getattr(prompt_server, "_tut_font_routes_registered", False):
        return True

    async def list_fonts(_request):
        try:
            fonts = await asyncio.to_thread(font_ui_catalog)
            return web.json_response({"fonts": list(fonts)})
        except (RuntimeError, ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def load_font_file(request):
        token = request.query.get("token", "")
        try:
            asset = await asyncio.to_thread(font_preview_asset, token)
        except (RuntimeError, ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=404)
        headers = {"Cache-Control": "private, max-age=3600"}
        if asset.path is not None:
            response = web.FileResponse(asset.path, headers=headers)
            response.content_type = asset.content_type
            return response
        return web.Response(body=asset.data or b"", content_type=asset.content_type, headers=headers)

    try:
        routes.get("/tut_nodes/fonts/catalog")(list_fonts)
        routes.get("/tut_nodes/fonts/file")(load_font_file)
    except (AttributeError, RuntimeError):
        return False

    prompt_server._tut_font_routes_registered = True
    return True
