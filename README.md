# YOLO 模型导出、推理与性能对比

本周沿用上次课程提供的装甲板模型，将 PyTorch `.pt` 权重导出为 ONNX，用同一段本人实拍视频分别推理，并在同一台电脑的 CPU 上完成预热和三轮性能对比。无需重新训练。

**本机实验结论：ONNX 平均预测耗时从 58.257 ms/帧降至 42.044 ms/帧，降低约 27.8%。六次完整运行的逐帧检测数量一致。** 本次未定量评估检测精度。

## 模型、素材与结果下载

仓库附有两种模型、原视频和两份 H.264 结果，单个文件均小于 11 MiB，可以直接下载，无需 Git LFS。

| 内容 | 仓库位置 | 大小 | 直接下载 |
|---|---|---:|---|
| 原始 PyTorch 模型 | `models/best.pt` | 5.62 MiB | [best.pt](https://raw.githubusercontent.com/Oliver000-1/yolo-armor-inference/main/models/best.pt) |
| 导出 ONNX 模型 | `models/best.onnx` | 9.72 MiB | [best.onnx](https://raw.githubusercontent.com/Oliver000-1/yolo-armor-inference/main/models/best.onnx) |
| 本人原始测试视频 | `videos/armor_original.mp4` | 4.23 MiB | [原视频](https://raw.githubusercontent.com/Oliver000-1/yolo-armor-inference/main/videos/armor_original.mp4) |
| PyTorch 推理结果 | `results/armor_pt_640.mp4` | 9.92 MiB | [PyTorch 视频](https://raw.githubusercontent.com/Oliver000-1/yolo-armor-inference/main/results/armor_pt_640.mp4) |
| ONNX 推理结果 | `results/armor_onnx_640.mp4` | 9.92 MiB | [ONNX 视频](https://raw.githubusercontent.com/Oliver000-1/yolo-armor-inference/main/results/armor_onnx_640.mp4) |

权重来自课程提供的 `best (1).pt`；架构为 `yolo26n-pose.yaml`，包含 B1/B2/B3/R1/R2/R3 六类，每个目标有四个 XY 预测关键点。测试素材是本人上次在 D414 拍摄的同一段视频：1280×720、30 FPS、410 帧、约 13.67 秒。

## 环境与安装

- 实测环境：Windows 10，Intel Core i7-10850H，Python 3.12.10。
- 本次统一 CPU、FP32。PyTorch 为 `2.14.1+cpu`，ONNX Runtime 使用 `CPUExecutionProvider`；本机 Quadro T2000 显卡未参与本次实验。
- [requirements.txt](requirements.txt) 固定直接依赖：Ultralytics 8.4.176、PyTorch 2.14.1、torchvision 0.29.1、OpenCV 5.0.0.93、NumPy 2.5.4、ONNX 1.23.2、ONNX Runtime 1.31.0、ONNX Slim 0.1.98。

Windows PowerShell：

```powershell
git clone https://github.com/Oliver000-1/yolo-armor-inference.git
cd yolo-armor-inference
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

直接调用虚拟环境 Python，无需激活。若创建虚拟环境时 `ensurepip` 失败，但 `.venv/Scripts/python.exe` 已生成，可使用系统 pip：

```powershell
python -m pip --python .\.venv\Scripts\python.exe install pip -r requirements.txt
```

## 模型导出

[export.py](export.py) 检查模型任务、类别与关键点结构，在临时目录导出、校验后才保存新模型。原始 `.pt` 保持不变，已有同名模型或报告时拒绝覆盖。仓库已附 `best.onnx`；如要重新导出，使用新文件名：

```powershell
.\.venv\Scripts\python.exe export.py --model models/best.pt --output models/best_reexport.onnx --imgsz 640
```

参数为 CPU、FP32（`quantize=32`）、静态输入 `[1, 3, 640, 640]`、batch 1、opset 18、图简化。原模型普通预测分支为 `end2end=False`，使用 `nms=None` 导出原始预测，阈值过滤与 NMS 在后续推理时完成。

检查包括 ONNX 图结构、任务/类别/关键点元数据、输入尺寸和类型，以及 CPU 空白输入试运行。空白输入只验证模型能够执行。实际参数、版本、大小和原始/导出文件 SHA-256 指纹见 [导出报告](models/best.export.json)。

## 两种模型的推理

[infer.py](infer.py) 支持 `.pt` 与 `.onnx`，直接读取每个目标的预测关键点；ONNX 从模型元数据读取类别和关键点结构，由 ONNX Runtime 执行推理。

```powershell
.\.venv\Scripts\python.exe infer.py --model models/best.pt --check-model
.\.venv\Scripts\python.exe infer.py --model models/best.onnx --check-model
.\.venv\Scripts\python.exe infer.py --model models/best.pt --source videos/armor_original.mp4 --output outputs/armor_pt_640.mp4 --device cpu --conf 0.25 --iou 0.7 --imgsz 640
.\.venv\Scripts\python.exe infer.py --model models/best.onnx --source videos/armor_original.mp4 --output outputs/armor_onnx_640.mp4 --device cpu --conf 0.25 --iou 0.7 --imgsz 640
```

两种格式共用 `prediction_options()`：`rect=False` 等比例缩放并补边到完整 640×640，CPU、FP32、conf 0.25、IoU 0.7、batch 1、max_det 300。原视频的分辨率、帧数和播放 FPS 保持一致。使用重新导出的模型时，将对应 `--model` 替换为 `models/best_reexport.onnx`。

每次保存完整 MP4、JSON、最高框置信度截图，以及第 0、90、180、270、360 帧的同帧截图。框和四点按目标配对；点号表示模型输出顺序，目标编号只在单帧内有效。已有结果时需更换 `--output`，脚本拒绝覆盖。

### 实际结果

| 对比项目 | 原始 `.pt` | ONNX |
|---|---|---|
| 模型大小 | 5,892,455 字节，5.62 MiB | 10,196,332 字节，9.72 MiB |
| 输入/输出帧数 | 410 / 410 | 410 / 410 |
| 输出规格 | 1280×720，30 FPS，13.67 秒 | 1280×720，30 FPS，13.67 秒 |
| 每轮逐帧检测总次数 | 634，全部为 B3 | 634，全部为 B3 |
| 平均每帧预测耗时 | 58.257 ms | 42.044 ms |
| 预测吞吐量 | 17.17 FPS | 23.78 FPS |

634 是跨帧累计检测次数，同一物体会在不同帧重复计数。没有人工真值标注，未计算准确率、召回率或关键点误差；相同框数不表示所有浮点坐标完全相同。

同一第 180 帧（约 6 秒），两者均显示两个 B3 目标及各自四个预测关键点：

PyTorch：

![PyTorch 第180帧](docs/pt_640_frame180.jpg)

ONNX：

![ONNX 第180帧](docs/onnx_640_frame180.jpg)

完整记录见 [PyTorch 推理统计](docs/pt_640_report.json)、[ONNX 推理统计](docs/onnx_640_report.json)。其中 `elapsed_seconds` 包含读写、绘图和输出检查，正式测速使用下一节独立记录。

## 公平测速

[benchmark.py](benchmark.py) 校验两种模型的导出来源，复用上述预测参数。每个模型先预热 10 帧，再按 `.pt → ONNX → .pt → ONNX → .pt → ONNX` 顺序各运行三遍完整视频。

```powershell
.\.venv\Scripts\python.exe -u benchmark.py --warmup 10 --repeats 3 --imgsz 640 --conf 0.25 --iou 0.7 --output docs/benchmark_cpu_reproduce.json --report comparison_reproduce.md
```

这里使用新输出名保留仓库附带的实验记录；再次复现时需继续更换 `--output` 和 `--report`。若使用重新导出的模型，增加 `--onnx-model models/best_reexport.onnx`。

计时使用 `time.perf_counter_ns()`，只覆盖同步返回的 `model.predict(...)`，包含预处理、网络前向和后处理，排除视频解码、模型加载、预热、计数、绘图、日志和文件保存。FPS 是预测调用吞吐量，结果视频仍以 30 FPS 播放。

[原始测速记录](docs/benchmark_cpu.json) 包含全部 2460 次正式预测的逐帧时间、三轮计数、执行顺序、模型指纹、CPU、库版本和线程配置。PyTorch 使用 8 个算子线程，ONNX Runtime 使用默认自动线程策略。每轮结果、波动与解释见 [comparison.md](comparison.md)。

本机本次 ONNX 平均每帧预测耗时降低约 27.8%，吞吐量约为 PyTorch 的 1.39 倍。差异可能受运行时算子优化、图优化、线程策略及系统负载影响，本实验未逐项消融这些因素；结论适用于此处记录的 CPU 环境。

## 问题与理解

1. **输入尺寸如何统一？** `.pt` 可能自动使用较小矩形输入，静态 ONNX 固定 640×640。本次统一 `rect=False` 并重新测量 `.pt`，上周结果作为历史保留。
2. **为什么 ONNX 文件更大？** 检查到原 checkpoint 参数以 FP16 保存，本次 ONNX 按 FP32 导出，两者 CPU 推理都为 FP32。图结构和文件组织也会影响大小，格式转换不保证文件更小。
3. **如何检查 ONNX 关键点？** ONNX 保存计算图，不能直接读取 Python 网络最后一层。本次从元数据及预测结果检查类别、输入和四点结构。
4. **如何保证可播放？** OpenCV 输出 `mp4v` 后转换成 H.264，完整解码核对帧数、分辨率和 FPS。用户已确认 ONNX 视频可以播放，两份 H.264 交付文件的 SHA-256 相同。
5. **效果有哪些局限？** 本视频主要验证 B3。侧面大角度目标有漏检，快速转动有运动模糊；保留全部帧和真实预测，不手动修正关键点，其他五类未得到充分验证。

`.pt` 用于 PyTorch 生态，ONNX 用计算图表达模型，交给 ONNX Runtime 等工具执行。导出后还要验证加载、真实推理、效果和性能，速度取决于硬件及运行时。

H.264 转换命令（FFmpeg 是可选的独立视频工具）：

```powershell
ffmpeg -hide_banner -n -i outputs/armor_pt_640.mp4 -map 0:v:0 -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart -an outputs/armor_pt_640_h264.mp4
ffmpeg -hide_banner -n -i outputs/armor_onnx_640.mp4 -map 0:v:0 -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart -an outputs/armor_onnx_640_h264.mp4
```

`results/` 中的交付视频是上述播放文件的副本。程序解码、文件指纹及用户播放确认见 [验证记录](docs/inference_comparison_checks.json)。

## 项目结构与提交

```text
export.py                 导出、来源记录与 ONNX 校验
infer.py                  双格式视频推理、绘制及输出检查
benchmark.py              预热、三轮测速与报告生成
requirements.txt          固定依赖版本
models/                   .pt、ONNX、导出报告
videos/armor_original.mp4 同一段本人实拍测试视频
results/                  两种模型的 H.264 交付视频
docs/                     实验记录、截图和上周说明归档
comparison.md             完整性能对比
outputs/                  本地自动输出，不进入 Git
```

本周提交 [个人 GitHub 仓库链接](https://github.com/Oliver000-1/yolo-armor-inference)，具体位置和截止时间以算法组本周通知为准。GitHub 文件发布与课程提交是独立步骤，应在指定收集渠道确认收到仓库链接。

上周内容保留：[说明归档](docs/previous_assignment.md)、[原截图说明](docs/README.md)、[已合并 PR #1](https://github.com/Oliver000-1/yolo-armor-inference/pull/1)。

接口参考：[Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/)、[ONNX 导出与推理](https://docs.ultralytics.com/integrations/onnx/)。
