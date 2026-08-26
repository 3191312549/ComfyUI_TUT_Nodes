"""Video animation nodes for the ``TUT_Nodes/视频/动画`` menu."""

from __future__ import annotations

from ...categories import VIDEO_ANIMATION


GIMMVFI_MODELS = [
    "gimmvfi_f_arb_lpips_fp32.safetensors",
    "gimmvfi_r_arb_lpips_fp32.safetensors",
]
GIMMVFI_PRECISIONS = ["fp32", "bf16", "fp16"]


class TUT_GIMMVFIInterpolate:
    """Expose GIMM-VFI loading and interpolation as one expandable node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "至少两帧、尺寸一致的 IMAGE 批次。"},
                ),
                "model": (GIMMVFI_MODELS,),
                "precision": (GIMMVFI_PRECISIONS, {"default": "fp32"}),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "编译部分模型；需要可用的 Triton 环境。",
                    },
                ),
                "ds_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "内部处理分辨率比例；降低可节省显存，但可能损失运动细节。",
                    },
                ),
                "interpolation_factor": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 100,
                        "step": 1,
                        "tooltip": "2 表示每对原始帧之间生成 1 帧。",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "output_flows": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "额外输出光流可视化，仅建议排障时开启。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "flow_tensors")
    FUNCTION = "interpolate"
    CATEGORY = VIDEO_ANIMATION
    DESCRIPTION = (
        "将 GIMM-VFI 模型加载与补帧合并为一个节点。"
        "依赖已安装并启用的 ComfyUI-GIMM-VFI。"
    )

    def interpolate(
        self,
        images,
        model,
        precision,
        torch_compile,
        ds_factor,
        interpolation_factor,
        seed,
        output_flows,
    ):
        if getattr(images, "ndim", None) != 4:
            raise ValueError("GIMM-VFI 的 images 必须是形状为 [帧数, 高, 宽, 通道] 的 IMAGE 批次。")
        if images.shape[0] < 2:
            raise ValueError("GIMM-VFI 至少需要两帧输入；请先把首帧和尾帧合并为 IMAGE 批次。")
        if images.shape[-1] != 3:
            raise ValueError(f"GIMM-VFI 只接受三通道 RGB 图片，当前通道数为 {images.shape[-1]}。")
        if model not in GIMMVFI_MODELS:
            raise ValueError(f"不支持的 GIMM-VFI 模型：{model!r}")
        if precision not in GIMMVFI_PRECISIONS:
            raise ValueError(f"不支持的 GIMM-VFI 精度：{precision!r}")
        if not 0.01 <= float(ds_factor) <= 1.0:
            raise ValueError(f"DS 因子必须在 0.01 到 1.00 之间：{ds_factor}")
        if not 2 <= int(interpolation_factor) <= 100:
            raise ValueError(f"补帧因子必须是 2 到 100 的整数：{interpolation_factor}")
        if not 0 <= int(seed) <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"随机种子必须在 0 到 18446744073709551615 之间：{seed}")

        import nodes as comfy_nodes

        required_nodes = ("DownloadAndLoadGIMMVFIModel", "GIMMVFI_interpolate")
        missing = [name for name in required_nodes if name not in comfy_nodes.NODE_CLASS_MAPPINGS]
        if missing:
            raise RuntimeError(
                "未检测到 ComfyUI-GIMM-VFI 节点，请确认插件已安装、启用并能正常加载后重启 ComfyUI。"
            )

        from comfy_execution.graph_utils import GraphBuilder

        graph = GraphBuilder()
        loader = graph.node(
            "DownloadAndLoadGIMMVFIModel",
            model=model,
            precision=precision,
            torch_compile=bool(torch_compile),
        )
        interpolator = graph.node(
            "GIMMVFI_interpolate",
            gimmvfi_model=loader.out(0),
            images=images,
            ds_factor=float(ds_factor),
            interpolation_factor=int(interpolation_factor),
            seed=int(seed),
            output_flows=bool(output_flows),
        )
        return {
            "result": (interpolator.out(0), interpolator.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {"TUT_GIMMVFIInterpolate": TUT_GIMMVFIInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_GIMMVFIInterpolate": "TUT_GIMM-VFI补帧"}
