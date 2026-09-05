# ComfyUI_TUT_Nodes

一套以中文界面为主的 ComfyUI 实用节点。本页介绍公开版的 **64 个节点**，覆盖文字排版、图片调色、滤镜、抠像、合成、漫画分镜、漫画对话框、音频裁剪、批次处理、Excel、Anima 画师提示词、LoRA 测试、潜空间放大和动画输出。

![TUT_Nodes 功能总览](docs/images/overview.svg)

## 快速安装

在 **ComfyUI 根目录**执行以下命令，将仓库下载到 `custom_nodes`：

```shell
git clone https://github.com/3191312549/ComfyUI_TUT_Nodes.git custom_nodes/ComfyUI_TUT_Nodes
```

也可以下载仓库 ZIP，解压后将文件夹改名为 `ComfyUI_TUT_Nodes`，放到以下位置（避免多套一层同名文件夹）：

```text
ComfyUI/custom_nodes/ComfyUI_TUT_Nodes
```

然后使用 **ComfyUI 自己的 Python 环境**安装依赖：

```powershell
python -m pip install -r custom_nodes/ComfyUI_TUT_Nodes/requirements.txt
```

重启 ComfyUI，在节点菜单中找到 `TUT_Nodes`。

上面的 `python` 必须指向运行 ComfyUI 的解释器；整合包或便携版请使用其自带 Python。主要依赖为 `uharfbuzz`、OpenCV 和 `openpyxl`，插件不会在加载时自动安装依赖。

> `AI智能抠像` 需要额外安装 `rembg`；`GIMM-VFI补帧` 需要已安装并启用 ComfyUI-GIMM-VFI；SesquiLSR 权重会在首次执行对应格式时按需下载。Anima 画师库随插件内置，搜索与混合不需要联网。

### 更新已安装的版本

通过 Git 安装的用户，在 **ComfyUI 根目录**执行：

```shell
git -C custom_nodes/ComfyUI_TUT_Nodes pull --ff-only
python -m pip install -r custom_nodes/ComfyUI_TUT_Nodes/requirements.txt
```

更新后重启 ComfyUI，并刷新浏览器页面以加载新的节点界面。ZIP 安装的用户可备份自添字体、LUT 等文件后，用新版文件替换插件目录。

## 三分钟认识节点

### 1. 做文字、标题和动图

![文字与动画工作流](docs/images/text-workflow.svg)

先用 `叠加文字`、`区域自适应文字` 或 `文字沿路径` 生成文字，再把 MASK 交给 `文字特效`。需要动画时连接 `动态文字序列`，最后用 `保存动画 GIF` 输出。

### 2. 调色、滤镜与合成

![调色与合成工作流](docs/images/image-workflow.svg)

常用顺序是：`自动基础校色` → `基础明暗调整` → `基础色彩调整` → `图像细节增强`。之后可继续使用 HSL、RGB 曲线、LUT、艺术滤镜或图层合成。

### 3. 批次、漫画和数据驱动

![批次与工具工作流](docs/images/tool-workflow.svg)

`图像到批次` 可把多张图片组成批次并送入 `漫画分镜画布`；Excel 节点可把表格内容转成文本列表；`LoRA强度批量测试` 可一次生成多档模型强度供下游对比。

### 4. 混合 Anima 画师提示词

连接方式：`文本提示词 → TUT_Anima画师提示词混合 → 文本编码节点`。

在节点内搜索画师，点击结果加入画师胶囊，再为每位画师调整权重。内置画师库包含 **42,196 个画师标签**，支持模糊搜索，也可添加自定义画师；权重范围为 `0.1–3.0`，步长为 `0.1`。

例如，输入 `1girl，smile`，添加权重 `1.0` 的 `wlop` 和权重 `1.2` 的 `望月けい`，画师部分会写为 `@wlop, (@望月けい:1.2)`。常用中文标点会转为英文标点，重复画师会合并，同名时以胶囊权重为准。

节点根据标签结构尝试将画师放在角色、作品之后和通用描述之前；无法判断角色或作品边界时会追加到提示词末尾。输出是一段普通文本，可继续编辑后再交给文本编码节点。

### 5. 试听并裁剪音频

使用 `TUT_高级音频加载` 选择或上传音频，在波形上拖动选区两端调整起止时间，也可输入具体秒数。选区支持试听和播放定位；结束时间设为 `0` 表示保留到音频末尾。输出保留原采样率与声道，并提供裁剪时长和实际起止时间。

## 节点分类

| 分类 | 适合做什么 | 代表节点 |
| --- | --- | --- |
| 图片 / 文本 | 标题、水印、文字动画、路径文字 | 叠加文字、文字特效、区域自适应文字 |
| 图片 / 调色 | 基础校色、曲线、LUT、电影风格 | 自动基础校色、RGB曲线、3D LUT调色 |
| 图片 / 滤镜 | 漫画、像素、故障、玻璃折射 | 漫画化滤镜、像素艺术滤镜 |
| 图片 / 抠像 | 颜色、AI、SAM、差异抠像 | 颜色抠像、AI智能抠像、遮罩边缘精修 |
| 图片 / 合成 | 柔边、光线包裹、深度、透视 | 柔边图层合成、四角定位合成 |
| 图片 / 漫画 | 多图自动排版、分镜与对白 | 漫画分镜画布、漫画对话框 |
| 音频 | 波形预览、试听与精确裁剪 | 高级音频加载 |
| 工具 | Excel、画师提示词、文本、批次、等待、帮助 | Anima画师提示词混合、读取Excel、图像到批次 |
| 模型与动画 | LoRA 对比、潜空间放大、补帧、GIF | LoRA强度批量测试、GIMM-VFI补帧 |

## 完整节点清单

<details>
<summary><strong>图片 / 文本（14）</strong></summary>

- `TUT_叠加文字`：在图片上绘制文字。
- `TUT_绘制文字`：生成文字画布。
- `TUT_文字遮罩填充`：把图片填入文字区域。
- `TUT_文字图像合成`：按文字遮罩合成前景和背景。
- `TUT_文字水印`：批量添加定位水印。
- `TUT_自适应反色水印`：根据底图自动反色，保持水印清晰。
- `TUT_选择字体`：选择插件字体或系统字体。
- `TUT_文字特效`：添加描边、阴影、发光、渐变、浮雕等效果。
- `TUT_区域自适应文字`：让文字自动适配指定区域。
- `TUT_逐字逐词遮罩`：按字、词或行输出独立遮罩批次。
- `TUT_字体预览墙`：分页预览可用字体。
- `TUT_动态文字序列`：生成打字、淡入、滑入、扫光等动画。
- `TUT_文字变形`：制作拱形、波浪、斜切和透视文字。
- `TUT_文字沿路径`：沿路径遮罩排列文字。

</details>

<details>
<summary><strong>图片 / 调色（14）</strong></summary>

- `TUT_自动基础校色`、`TUT_自动基础校色（高级）`
- `TUT_基础明暗调整`、`TUT_基础色彩调整`、`TUT_图像细节增强`
- `TUT_HSL基础调整`、`TUT_RGB曲线`
- `TUT_双图颜色匹配`、`TUT_电影色调塑形`
- `TUT_卤化光晕`、`TUT_镜头扩散`、`TUT_色彩压缩器`
- `TUT_LUT加载与预览`、`TUT_3D LUT调色`

</details>

<details>
<summary><strong>图片 / 滤镜与动画（8）</strong></summary>

- `TUT_复古印刷滤镜`
- `TUT_漫画化滤镜`
- `TUT_万花筒滤镜`
- `TUT_像素艺术滤镜`
- `TUT_玻璃折射滤镜`
- `TUT_故障艺术滤镜`
- `TUT_动态滤镜序列`
- `TUT_保存动画 GIF`

</details>

<details>
<summary><strong>图片 / 抠像与合成（11）</strong></summary>

- 抠像：`TUT_颜色抠像`、`TUT_AI智能抠像`、`TUT_SAM遮罩抠像`、`TUT_差异抠像`、`TUT_遮罩边缘精修`
- 合成：`TUT_柔边图层合成`、`TUT_光线包裹合成`、`TUT_深度图合成`、`TUT_四角定位合成`、`TUT_通道布尔合成`、`TUT_图像位移合成`

</details>

<details>
<summary><strong>图片 / 查看与漫画（3）</strong></summary>

- `TUT_图像对比`：使用滑动分割线比较两张图片。
- `TUT_漫画分镜画布`：把图片批次自动排成一至六格漫画，或自由绘制最多二十格；每格可拖动顶点或边中点形成抗锯齿凸形、凹形四边框，并支持逐边开放、图层、吸附和镜头预览。
- `TUT_漫画对话框`：添加可拖动、缩放、分层和可重叠合并的对白框、自然云朵框、爆炸框、闪光框、旁白框或无边框文字；支持字体搜索、真实字体预览、横排与双向竖排，以及爆炸/闪光轮廓调节。

</details>

<details>
<summary><strong>音频（1）</strong></summary>

- `TUT_高级音频加载`：上传或选择音频，在波形上试听并按采样点精确裁剪，输出时长、边界、采样率和声道数。

</details>

<details>
<summary><strong>工具（10）</strong></summary>

- `TUT_节点帮助`：连接任意节点输出，在节点内查看使用说明。
- `TUT_文本分隔批次`：按分隔符把文本拆成列表。
- `TUT_Anima画师提示词混合`：离线搜索并混合加权画师标签，自动去重、转换常用中文标点，并根据提示词结构选择插入位置。
- `TUT_图像到批次`：合并最多十路图片输入，不拉伸不同尺寸图片。
- `TUT_批次按编号加载`：按编号选择列表中的一项。
- `TUT_等待`：延时后原样传递任意类型数据。
- `TUT_读取Excel`：读取 `.xlsx`、`.xlsm` 或 `.csv`。
- `TUT_Excel批次读取`：按行或列输出文本列表。
- `TUT_Excel合并读取`：按行或列合并为一段文本。
- `TUT_Excel只读单行/单列`：读取指定行或列。

</details>

<details>
<summary><strong>模型、潜空间与视频（3）</strong></summary>

- `TUT_LoRA强度批量测试`：生成多档 MODEL、CLIP 和强度列表。
- `TUT_SesquiLSR潜空间放大`：放大 SDXL、Flux、Flux2、Ideogram 4 或 Wan 2.1 LATENT。
- `TUT_GIMM-VFI补帧`：复用 ComfyUI-GIMM-VFI 完成视频补帧。

</details>

## 使用提示

- 节点标题、参数和提示以中文显示；内部 ID 保留英文，以兼容旧工作流。
- 大多数图片节点支持批次，调色、滤镜和抠像节点支持可选 MASK。
- 动态文字和动态滤镜输出 IMAGE 帧批次，帧率在 `保存动画 GIF` 中设置。
- 插件字体放入 `ComfyUI_TUT_Nodes/fonts` 后重启 ComfyUI；支持 `.ttf`、`.otf`、`.ttc`。
- LUT 可放入 `ComfyUI_TUT_Nodes/luts`，也可在加载节点中上传。插件不会自动猜测 Log、Rec.709 或 ACES 色彩空间。
- Excel 采用“读取一次、处理多次”的连接方式：先连接 `读取Excel`，再连接三个 Excel 处理节点。
- `图像到批次` 最多连接十路 IMAGE；不同尺寸会居中补透明边，不拉伸原图。`批次按编号加载` 选择的是执行列表中的一项，不拆分 IMAGE 内部的图片批次。
- 旧版 Excel 工作流若仍直接向处理节点传文件路径，需要添加 `读取Excel` 节点并重新连接。
- SesquiLSR 模型首次使用时下载到 `models/TUT_Nodes/sesqui_lsr/`。第三方来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可

本项目采用 [MIT License](LICENSE)。
