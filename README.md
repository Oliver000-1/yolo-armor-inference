# 装甲板视频推理

使用课程提供的 `best.pt` 对视频逐帧推理，显示检测框、类别、置信度和每个目标的四个预测关键点。无需重新训练。

**当前阶段：代码准备；尚未提供本人实拍视频，实际识别表现待验证。**

2026-10-13 本地准备验证：权重成功加载，架构配置为 `yolo26n-pose.yaml`，类别为 B1/B2/B3/R1/R2/R3，关键点形状为 `[4, 2]`。使用 10 帧空白程序测试视频执行 CPU 推理，结果 10 帧均成功解码；没有检测属于此空白测试的预期结果，不代表实拍表现。

## 环境与文件

- Python 3.12，默认使用 CPU，不要求 NVIDIA 显卡。
- 主要依赖：Ultralytics、OpenCV、NumPy，版本见 `requirements.txt`。
- 模型由课程/学长提供，不包含在仓库中。将原下载文件 `best (1).pt` 复制为 `models/best.pt`。
- 本人原始视频放在 `videos/armor_original.mp4`；MOV 等格式也可通过参数指定。
- 权重、原始视频、结果视频、虚拟环境和自动输出已由 `.gitignore` 排除。

```text
infer.py                推理、绘制、保存及输出验证
requirements.txt        主要依赖
models/best.pt          自行准备，不进入 Git
videos/armor_original.mp4 本人实拍，不进入 Git
outputs/                自动生成的视频、截图与统计，不进入 Git
docs/                   手动挑选的真实代表截图，可进入 Git
```

## 安装（Windows PowerShell）

在项目文件夹内执行；命令直接使用虚拟环境的 Python，无需激活环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

若创建虚拟环境时 `ensurepip` 失败，但 `.venv/Scripts/python.exe` 已生成，可以使用系统 Python 的 pip 安装：

```powershell
python -m pip --python .\.venv\Scripts\python.exe install pip -r requirements.txt
```

## 运行

先检查模型类别和关键点数：

```powershell
.\.venv\Scripts\python.exe infer.py --model models/best.pt --check-model
```

实拍后执行：

```powershell
.\.venv\Scripts\python.exe infer.py --model models/best.pt --source videos/armor_original.mp4 --output outputs/armor_result.mp4 --device cpu --conf 0.25 --imgsz 640
```

如使用其他路径，将参数替换为实际路径，包含空格时加双引号。
有可用 CUDA 环境时可将 `--device cpu` 改为 `--device 0`；当前准备的是 CPU 环境。

输出文件：

- `outputs/armor_result.mp4`：与输入逐帧对应的静音结果视频，使用输入 FPS，不加速播放。保留原始视频作为作业附件。
- `outputs/armor_result.json`：输入/输出帧数、时长、检测统计、运行参数和警告。
- `outputs/armor_result.jpg`：存在检测时，自动保存最高框置信度帧；需要人工确认其能清晰展示关键点。

已有同名输出时脚本拒绝覆盖；重新运行请换名字，例如 `outputs/armor_result_v2.mp4`。
脚本会完整解码输出视频检查帧数；提交前仍应在播放器中人工检查画面与关键点。
若网页播放器不支持生成的 MP4 编码，可用视频转码工具转为 H.264 MP4，并重新检查。

## 主要代码如何工作

1. `load_model()` 加载权重，检查 pose 任务、六类名称以及每个目标的四关键点结构。
2. `run()` 用 OpenCV 逐帧读取视频，调用 `model.predict()`。
3. `annotate()` 从 `result.boxes` 取框、类别和置信度，从 `result.keypoints.data` 取模型预测的坐标，不用框角替代。
4. 目标框与其关键点使用相同颜色，`#1` 对应 `1:1` 至 `1:4`。目标编号仅在单帧内有效，不是跨帧追踪 ID。
5. `run()` 保存视频、检测代表帧和 JSON，并检查结果视频能否完整解码。

关键点未设额外置信度过滤，以便如实观察四点位置；非有限值或画面外坐标无法绘制，不人为修正。点号是模型输出顺序，未确认训练标注前不宣称它一定是某个方向的角。

## 本人拍摄信息（拍摄后填写）

| 项目 | 实际记录 |
|---|---|
| 地点 | 待拍摄：水建学院 D414 实验室 |
| 拍摄日期 | 待填写 |
| 时长 | 待填写，要求 10～30 秒 |
| 设备、分辨率及方向 | 待填写，建议横屏 |
| 场景 | 待填写目标类别、光照、距离和角度变化 |

## 真实结果与问题记录（实拍推理后填写）

- 实际运行命令：待填写。
- 输入时长、输出帧数、有检测帧数：待填写，可参考 JSON。
- 识别表现：待填写；分别观察类别、检测框和四点位置。
- 漏检、误检、类别错误或点位不准：记录具体时间段和现象，保留真实结果。
- 全片无检测时：核对模型检查结果，确认视频包含对应目标并且可播放；检查光照、目标大小；可用 `--conf 0.10` 另存一次结果，记录调整前后差异。不要将“有检测帧比例”写成识别准确率。
- 截图：实拍推理后，将能清晰展示框和四点的图片复制到 `docs/result.jpg`，再在这里加入 `![代表性结果](docs/result.jpg)`。

## GitHub 与 PR 流程

本地已初始化 `main`，当前工作分支是 `feat/yolo-inference`。初始化提交由 Codex 工具署名；后续提交请设置自己的 Git 身份。本地 origin 已预配置为 `https://github.com/Oliver000-1/yolo-armor-inference.git`，远端仓库是否已创建需在登录后确认。

1. 在自己的 GitHub 账号创建 **Public** 仓库 `yolo-armor-inference`。本地已有 README，网页创建时不要重复初始化文件。
2. 设置自己的 Git 提交身份（使用 GitHub 设置里提供的 noreply 邮箱也可以），并关联仓库：

```powershell
git config user.name "你的 GitHub 用户名"
git config user.email "你的提交邮箱"
git remote -v
# 本地已有 origin；若使用不同仓库，可执行：
# git remote set-url origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

3. 拍摄、运行验证并补全 README 后，检查将提交的内容：

```powershell
git status
git add infer.py requirements.txt README.md .gitignore .github docs
git diff --cached --stat
git commit -m "完成装甲板视频推理与关键点显示"
git push -u origin feat/yolo-inference
```

4. 在 GitHub 创建 PR：base 为 `main`，compare 为 `feat/yolo-inference`，双方均属于自己的仓库。补全 PR 模板中的实际验证结果。
5. 检查差异包含推理代码、依赖和 README，然后本人合并；确认显示 **Merged**，且 main 有完整代码。

## 提交前清单

- [ ] 视频是本人在 D414 拍摄的 10～30 秒原始素材。
- [ ] README 拍摄信息、真实结果和代表截图已补齐。
- [ ] 公开仓库链接有效，PR 状态是 Merged。
- [ ] 原始视频和结果视频均可播放，每个不超过 100 MiB。
- [ ] 作业说明包含仓库链接和 PR 链接，附件包含两个视频。
- [ ] 已点击“提交作业”并确认；重交时上传完整的一组附件。

截止：北京时间 **2026-10-18 00:00**，即 10 月 17 日周六结束时。

API 参考：[Ultralytics Pose 文档](https://docs.ultralytics.com/tasks/pose/)。
