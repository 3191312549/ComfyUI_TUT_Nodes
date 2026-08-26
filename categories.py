"""Central category names used by every TUT node.

ComfyUI treats ``/`` as a submenu separator.  Keeping all category strings here
prevents spelling drift as the plugin grows.
"""

ROOT = "TUT_Nodes"

IMAGE = f"{ROOT}/图片"
IMAGE_TEXT = f"{IMAGE}/文本"
IMAGE_FILTER = f"{IMAGE}/滤镜"
IMAGE_COLOR = f"{IMAGE}/调色"
IMAGE_COMPOSITE = f"{IMAGE}/合成"
IMAGE_ANIMATION = f"{IMAGE}/动画"
IMAGE_KEYING = f"{IMAGE}/抠像"

VIDEO = f"{ROOT}/视频"
VIDEO_ANIMATION = f"{VIDEO}/动画"

TOOLS = f"{ROOT}/工具"
TOOLS_FILE = f"{TOOLS}/文件"
TOOLS_TEXT = f"{TOOLS}/文本"
TOOLS_EXCEL = f"{TOOLS}/Excel"
TOOLS_BATCH = f"{TOOLS}/批次"
TOOLS_WORKFLOW = f"{TOOLS}/流程"
