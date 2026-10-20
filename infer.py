"""Run the supplied four-keypoint armor model on a locally recorded video."""
# 项目导读（第一次看 Python 可以先读这里）
# 视频可以理解为连续播放的很多张图片，每张图片叫“一帧”。
# 本程序的流程：读取参数 → 加载学长训练好的模型 → 逐帧识别 → 画标记 → 保存视频。
# 这里做的是“推理”（使用模型），不是“训练”（让模型学习）。
# 模型对每个目标提供两类位置：检测框，以及单独预测的四个关键点。
# 关键点必须用模型预测坐标，不能用检测框四角代替，这是本作业的重要要求。
#
# 阅读顺序建议：先看文件末尾的入口，再看 run()，最后看其他三个函数。
# def 表示定义函数；调用函数时，括号里放输入；return 表示把结果交还给调用者。
# args 保存运行参数；frame 是一帧图片；result 是模型对这一帧的识别结果。
# “#”开头的内容是注释，只帮助阅读，不参与程序执行。

# 下面是导入工具：argparse 读取命令行参数，json 保存统计，math 检查数值，
# os 设置环境变量，Path 处理文件路径，time 用来计时。
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time

# 把模型工具的配置存到项目的 .local 文件夹；若已有配置位置则保留原设置。
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# cv2（OpenCV）负责读写视频、画框、画点；numpy 处理数组；YOLO 负责模型推理。
import cv2
import numpy as np
from ultralytics import YOLO


def arguments():
    # 第一步：读取用户在命令后面输入的选项。
    # 例如 --source videos/armor_original.mp4 就是在指定输入视频。
    # default 是不填写时的默认值；type 是把输入转换成哪种数据类型。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--source", type=Path, help="Your original D414 video")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/armor_result.mp4")
    # cpu 使用处理器；0 表示第一块可用的 CUDA 显卡（需要对应环境）。
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device, e.g. 0")
    # conf 是检测置信度门槛：越低通常保留越多候选目标，也可能增加误检。
    # 置信度是模型给出的分数，不等于整个项目的准确率。
    parser.add_argument("--conf", type=float, default=0.25)
    # IoU 用于去掉高度重叠的重复候选框；它不是模型识别准确率。
    parser.add_argument("--iou", type=float, default=0.7)
    # imgsz 控制模型内部处理的输入尺寸；保存的视频仍使用原视频分辨率。
    parser.add_argument("--imgsz", type=int, default=640)
    # 加上 --check-model 时只检查权重，不处理视频。
    parser.add_argument("--check-model", action="store_true", help="Inspect model without a video")
    args = parser.parse_args()
    # 提前检查参数，避免运行一段时间后才发现输入不合理。
    if not 0 < args.conf <= 1 or not 0 <= args.iou <= 1 or args.imgsz <= 0 or args.imgsz % 32:
        parser.error("conf must be in (0, 1], iou in [0, 1], and imgsz a positive multiple of 32")
    if not args.check_model and args.source is None:
        parser.error("--source is required unless using --check-model")
    return args


def file_sha256(path):
    # 文件指纹用于证明两次实验使用同一段原视频、同一份权重来源。
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prediction_options(conf=0.25, iou=0.7, imgsz=640, device="cpu", end2end=False):
    """Shared preprocessing and prediction settings for saved videos and benchmarks."""
    # 正式推理与测速共用这里，防止修改一处后另一处仍使用不同设置。
    return {"conf": conf, "iou": iou, "imgsz": imgsz, "device": device,
            "rect": False, "quantize": 32, "batch": 1, "max_det": 300,
            "dnn": False, "nms": False if end2end else None, "verbose": False}


def load_model(path):
    # 第二步：检查权重文件存在，再加载模型。best.pt 保存了训练得到的参数。
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.suffix.lower() == ".pt":
        model = YOLO(str(path))
        # .pt 可直接读取网络最后一层；ONNX 没有这种 Python 网络对象。
        shape = list(getattr(model.model.model[-1], "kpt_shape", []))
        names, task = model.names, model.task
        input_shape = None
        end2end = bool(getattr(model.model, "end2end", False))
    elif path.suffix.lower() == ".onnx":
        import onnx
        graph = onnx.load(str(path), load_external_data=False)
        metadata = {item.key: item.value for item in graph.metadata_props}
        # literal_eval 只解析字典、列表等数据，不把模型元数据当作代码执行。
        names = ast.literal_eval(metadata.get("names", "{}"))
        shape = list(ast.literal_eval(metadata.get("kpt_shape", "[]")))
        task = metadata.get("task")
        end2end = ast.literal_eval(metadata.get("end2end", "False"))
        if len(graph.graph.input) != 1:
            raise ValueError("Expected an ONNX model with one image input")
        image_input = graph.graph.input[0].type.tensor_type
        input_shape = [dim.dim_value for dim in image_input.shape.dim]
        if (len(input_shape) != 4 or input_shape[:2] != [1, 3]
                or input_shape[2] <= 0 or input_shape[2] != input_shape[3]
                or image_input.elem_type != onnx.TensorProto.FLOAT):
            raise ValueError("Expected a fixed square FP32 ONNX input [1, 3, size, size]")
        model = YOLO(str(path), task="pose")
    else:
        raise ValueError("Supported model formats are .pt and .onnx")
    # [4, 2] 表示每个目标有四个点，每个点包含 x、y 两个坐标。
    info = {"task": task, "names": names, "keypoint_shape": shape,
            "format": path.suffix.lower(), "path": str(path),
            "sha256": file_sha256(path), "file_bytes": path.stat().st_size,
            "input_shape": input_shape, "end2end": end2end}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    # pose 是带关键点预测的任务；普通目标检测模型不一定提供关键点。
    if task != "pose" or shape != [4, 2]:
        raise ValueError("Expected a pose model with four XY keypoints per target")
    expected = {"B1", "B2", "B3", "R1", "R2", "R3"}
    if not isinstance(names, dict) or set(names.values()) != expected:
        raise ValueError("Model class names do not match the assignment's six classes")
    return model, info


def annotate(frame, result):
    """Read keypoints from model output; box corners are never used as keypoints."""
    # 第三步：把这一帧的识别结果画到图片上，并返回图片和目标数量。
    # 没有检测到目标时保留原画面，不删除这一帧，以保持视频内容和时长。
    count = len(result.boxes)
    if count == 0:
        return frame, 0
    if result.keypoints is None or len(result.keypoints.data) != count:
        raise ValueError("Missing keypoints or mismatched box/keypoint instances")
    # 转到 CPU 并转成 NumPy 数组，便于后面逐个读取数据、用 OpenCV 绘制。
    # boxes[i] 与 points[i] 属于同一个目标，所以框和点不会随意混配。
    boxes = result.boxes.cpu().numpy().data
    points = result.keypoints.cpu().numpy().data
    if points.shape[1] != 4:
        raise ValueError("Expected four predicted keypoints")
    # OpenCV 颜色顺序是 B、G、R；同一目标的框和点用同一种颜色。
    colors = [(0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 100, 240)]
    # zip 把每个框与它的四个点配对；enumerate 从 1 开始给目标编号。
    for index, (box, target_points) in enumerate(zip(boxes, points), start=1):
        # box 前四项是检测框左上角、右下角坐标；这不是四个关键点。
        # 图像左上角是 (0, 0)，x 向右增大，y 向下增大。
        x1, y1, x2, y2 = map(int, box[:4])
        # 最后两项分别是置信度和类别编号；names 把编号转换为 B3 等名称。
        confidence, class_id = float(box[-2]), int(box[-1])
        # % 是取余，让多个目标循环使用颜色；编号仍可区分同色目标。
        color = colors[(index - 1) % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 例如显示“#1 B3 0.89”；:.2f 表示分数保留两位小数。
        label = f"#{index} {result.names[class_id]} {confidence:.2f}"
        cv2.putText(frame, label, (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        # 四个点来自 result.keypoints，独立于上面的框坐标。
        # 点号沿用模型输出顺序，不擅自解释成“左上角”等方向。
        for number, point in enumerate(target_points, start=1):
            # 非正常数值或落在画面外的点不画，也不会伪造一个替代坐标。
            if not np.isfinite(point[:2]).all():
                continue
            # 模型坐标可能带小数；绘图时四舍五入到整数像素。
            px, py = map(lambda value: int(round(float(value))), point[:2])
            if not (0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]):
                continue
            # 先画黑色大圆，再画彩色小圆，形成黑边，便于看清。
            # 标记“1:3”表示这一帧第 1 个目标的第 3 个点。
            # 编号每帧重新分配，不表示在追踪同一个物体。
            cv2.circle(frame, (px, py), 6, (0, 0, 0), -1)
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.putText(frame, f"{index}:{number}", (px + 7, max(15, py - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame, count


def run(args):
    # 主流程：把加载模型、读取视频、推理绘图、保存和检查串起来。
    model, info = load_model(args.model)
    if args.check_model:
        return
    # 开始前检查路径；raise 会停止程序并告诉用户哪里有问题。
    if not args.source.is_file():
        raise FileNotFoundError(f"Input video not found: {args.source}")
    if info["input_shape"] is not None and info["input_shape"][2:] != [args.imgsz, args.imgsz]:
        raise ValueError(f"ONNX input is fixed at {info['input_shape'][2:]}; use the matching --imgsz")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    # with_suffix 只换扩展名：同名的 .mp4、.json、.jpg 对应视频、统计、截图。
    report_path = args.output.with_suffix(".json")
    screenshot_path = args.output.with_suffix(".jpg")
    # 防止覆盖以前的结果。再次运行时可以更换 --output 文件名。
    for path in (args.output, report_path, screenshot_path):
        if path.exists():
            raise FileExistsError(f"Use a new output name; already exists: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # capture 是视频读取器；writer 是视频写入器，拿到第一帧尺寸后再创建。
    capture = cv2.VideoCapture(str(args.source))
    writer = None
    # frames：已处理帧数；detected_frames：至少检出一个目标的帧数；
    # detections：各帧目标数量之和，同一装甲板在不同帧里会重复计数。
    frames = detected_frames = detections = 0
    frame_detection_counts = []
    class_detection_counts = {name: 0 for name in info["names"].values()}
    sample_screenshots = []
    best_score = -1.0
    source_hash = file_sha256(args.source)
    # 两种格式共享相同的预处理和阈值，rect=False 强制补边到完整正方形。
    # 不量化，使用 FP32；普通分支在推理工具中执行 NMS，不在导出时硬编码阈值。
    predict_options = prediction_options(args.conf, args.iou, args.imgsz, args.device, info["end2end"])
    start = time.perf_counter()
    # try/finally 保证即使中途出错，最后也会关闭视频读写器。
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open input video")
        # FPS 是每秒播放多少帧。沿用原 FPS，避免结果视频被加速或减速。
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input has invalid FPS; convert it to a constant-frame-rate MP4")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # 在同样的 0/3/6/9/12 秒取截图，便于之后并排比较，不能各挑最好的一帧。
        sample_paths = {int(round(seconds * fps)): args.output.with_name(
            f"{args.output.stem}_frame{int(round(seconds * fps)):06d}.jpg")
            for seconds in (0, 3, 6, 9, 12)}
        for path in sample_paths.values():
            if path.exists():
                raise FileExistsError(f"Use a new output name; already exists: {path}")
        # 不断读取下一帧，直到视频结束或无法继续读取。
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # 仅第一帧需要创建写入器，以这一帧的宽高作为输出视频尺寸。
            if writer is None:
                height, width = frame.shape[:2]
                # 使用偶数宽高，避免当前编码方式悄悄裁掉边缘像素。
                if width % 2 or height % 2:
                    raise ValueError("Use an input video with even width and height")
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Cannot create MP4 writer")
            # 真正的模型推理发生在这里。一次输入一张图，返回结果列表，
            # 所以用 [0] 取这一张图的结果。verbose=False 减少逐帧日志。
            result = model.predict(frame, **predict_options)[0]
            # 画好框、文字和四点后，立即写入输出视频，不把全部帧堆在内存中。
            annotated, count = annotate(frame, result)
            writer.write(annotated)
            frame_detection_counts.append(count)
            for class_id in result.boxes.cls.cpu().tolist():
                class_detection_counts[result.names[int(class_id)]] += 1
            if frames in sample_paths:
                sample_path = sample_paths[frames]
                encoded_ok, encoded = cv2.imencode(".jpg", annotated)
                if not encoded_ok:
                    raise RuntimeError("Sample screenshot encoding failed")
                sample_path.write_bytes(encoded.tobytes())
                sample_screenshots.append({"frame_index": frames, "seconds": frames / fps,
                                           "path": str(sample_path), "detections": count})
            frames += 1
            detections += count
            # count > 0 为真时 int(...) 是 1，否则是 0。
            detected_frames += int(count > 0)
            if count:
                # 找这一帧最高的检测框置信度；超过历史最高值就更新代表截图。
                # 最高置信度不一定等于最清晰，最终截图仍需人工检查。
                score = float(result.boxes.conf.max().item())
                if score > best_score:
                    # 先编码成 JPG 数据，再写入文件，对中文文件路径也更友好。
                    encoded_ok, encoded = cv2.imencode(".jpg", annotated)
                    if not encoded_ok:
                        raise RuntimeError("Screenshot encoding failed")
                    screenshot_path.write_bytes(encoded.tobytes())
                    best_score = score
            # 每处理 30 帧提示一次进度，不是每隔 30 秒。
            if frames % 30 == 0:
                print(f"Processed {frames} frames; {detected_frames} contain detections")
    finally:
        # release 释放文件资源；写入器关闭后，视频文件才能完整保存。
        capture.release()
        if writer is not None:
            writer.release()
    if not frames:
        raise RuntimeError("Input video contains no decodable frames")
    # 保存后重新从头读取完整视频，核对可读帧数，避免只检查第一帧。
    # 这能检查文件读写，但不能证明模型识别完全正确。
    saved = cv2.VideoCapture(str(args.output))
    verified_frames = 0
    output_fps = saved.get(cv2.CAP_PROP_FPS)
    output_width = int(saved.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(saved.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while saved.read()[0]:
            verified_frames += 1
    finally:
        saved.release()
    if verified_frames != frames:
        raise RuntimeError(f"Output validation failed: {verified_frames}/{frames} frames")
    if not math.isclose(output_fps, fps, rel_tol=1e-4) or (output_width, output_height) != (width, height):
        raise RuntimeError("Output validation failed: FPS or resolution changed")
    # 收集需要注意的情况：读入帧数异常、全片无检测、视频超过附件大小限制。
    warnings = []
    if expected_frames > 0 and frames != expected_frames:
        warnings.append("Decoded frame count differs from input metadata; check input integrity")
    if detected_frames == 0:
        warnings.append("No detections: check the target, lighting, model and confidence threshold")
    if args.output.stat().st_size > 100 * 1024 * 1024:
        warnings.append("Result exceeds the assignment's 100 MiB video limit")
    # 把本次输入、参数、帧数、耗时等保存为 JSON（结构化的文本文件）。
    # 时长 = 帧数 / FPS；耗时包括推理、画图、写文件和输出视频检查。
    # 这些是运行统计，没有人工真值标注，不能据此计算识别准确率。
    report = {"model": info, "source": str(args.source), "source_sha256": source_hash,
              "output": str(args.output), "predict_options": predict_options,
              "device": args.device, "confidence_threshold": args.conf, "image_size": args.imgsz,
              "iou_threshold": args.iou, "rect": False, "precision": "FP32",
              "frames": frames, "input_metadata_frames": expected_frames, "fps": fps,
              "width": width, "height": height, "verified_output_fps": output_fps,
              "duration_seconds": frames / fps, "detected_frames": detected_frames,
              "total_detections": detections, "verified_output_frames": verified_frames,
              "frame_detection_counts": frame_detection_counts,
              "class_detection_counts": class_detection_counts,
              "elapsed_seconds": round(time.perf_counter() - start, 2),
              "timing_scope": "Full video processing and output validation; not a model benchmark",
              "sample_screenshots": sample_screenshots,
              "screenshot": str(screenshot_path) if best_score >= 0 else None,
              "warnings": warnings}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # 终端只显示摘要；410 帧逐帧计数等完整细节保存在 JSON 中。
    summary = {key: report[key] for key in ("output", "frames", "fps", "duration_seconds",
               "detected_frames", "total_detections", "verified_output_frames", "warnings")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print("Finished. Output is silent; original video remains unchanged.")


# 程序入口：直接运行 python infer.py 时，从这里开始。
# 先 arguments() 收集参数，再把参数交给 run() 执行整个任务。
# 如果其他 Python 文件只是导入本文件，就不会自动开始处理视频。
if __name__ == "__main__":
    run(arguments())
