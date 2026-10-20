"""Benchmark PyTorch and ONNX on the same video after warmup, on CPU only."""
# 文件导读：检查模型来源 → 各预热 10 帧 → 交替各测 3 轮 → 保存原始记录和对比报告。
# 只计 model.predict(...) 的时间，包含预处理、网络前向和后处理。
# 视频解码、计数、日志、模型加载、绘图、文件保存均不计入；CPU 调用会同步返回。
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from infer import ROOT, file_sha256, load_model, prediction_options
import cv2
import torch
from ultralytics.utils.torch_utils import get_cpu_info


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--onnx-model", type=Path, default=ROOT / "models/best.onnx")
    parser.add_argument("--source", type=Path, default=ROOT / "videos/armor_original.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmark_cpu.json")
    parser.add_argument("--report", type=Path, default=ROOT / "comparison.md")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    if args.imgsz <= 0 or args.imgsz % 32 or not 0 < args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("invalid image size, confidence or IoU")
    if args.pt_model.suffix.lower() != ".pt" or args.onnx_model.suffix.lower() != ".onnx":
        parser.error("provide a .pt baseline and an .onnx export")
    if args.output.suffix.lower() != ".json" or args.report.suffix.lower() != ".md":
        parser.error("--output must end with .json and --report with .md")
    for path in (args.output, args.report):
        if path.exists():
            parser.error(f"refusing to overwrite {path}; use new output/report paths")
    return args


def relative_path(path):
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if frames <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video must have valid frame-count and FPS metadata")
        return {"path": relative_path(path), "sha256": file_sha256(path), "frames": frames,
                "fps": fps, "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        capture.release()


def measure_video(model, path, options, expected_frames, warmup=False, label=""):
    # 流式解码，避免把整个视频放入内存。warmup=True 时只处理前 expected_frames 帧。
    capture = cv2.VideoCapture(str(path))
    durations_ns, counts = [], []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        while True:
            ok, frame = capture.read()  # 在计时区间外读图和解码。
            if not ok:
                break
            started_ns = time.perf_counter_ns()
            result = model.predict(frame, **options)[0]
            elapsed_ns = time.perf_counter_ns() - started_ns
            # 计数与检查也在计时区间外；不画框、不保存视频。
            count = len(result.boxes)
            if count and (result.keypoints is None or tuple(result.keypoints.data.shape) != (count, 4, 2)):
                raise ValueError("Prediction lost the four XY keypoints")
            durations_ns.append(elapsed_ns)
            counts.append(count)
            if warmup and len(counts) == expected_frames:
                break
            if not warmup and len(counts) % 100 == 0:
                print(f"{label}: {len(counts)}/{expected_frames} frames", flush=True)
    finally:
        capture.release()
    if len(counts) != expected_frames:
        raise ValueError(f"Decoded {len(counts)} frames, expected {expected_frames}")
    seconds = sum(durations_ns) / 1e9
    if seconds <= 0:
        raise ValueError("Invalid measured duration")
    return {"frames": len(counts), "prediction_seconds": seconds,
            "mean_ms": seconds * 1000 / len(counts), "prediction_fps": len(counts) / seconds,
            "total_detections": sum(counts), "frame_detection_counts": counts,
            "per_frame_prediction_ns": durations_ns}


def runtime_configuration(model):
    # 保留两种运行时的默认线程策略，并记录实际配置，不宣称它们线程数完全相同。
    config = {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "opencv_threads": cv2.getNumThreads()}
    backend = model.predictor.model.backend
    if hasattr(backend, "session"):
        session = backend.session
        options = session.get_session_options()
        config.update({"providers": session.get_providers(), "onnx_intra_op_threads": options.intra_op_num_threads,
                       "onnx_inter_op_threads": options.inter_op_num_threads,
                       "onnx_execution_mode": str(options.execution_mode),
                       "onnx_graph_optimization": str(options.graph_optimization_level)})
    return config


def summarize(runs):
    frames = sum(run["frames"] for run in runs)
    seconds = sum(run["prediction_seconds"] for run in runs)
    means = [run["mean_ms"] for run in runs]
    return {"measured_frames": frames, "prediction_seconds": seconds,
            "mean_ms": seconds * 1000 / frames, "prediction_fps": frames / seconds,
            "round_mean_ms_stdev": statistics.stdev(means) if len(means) > 1 else 0.0,
            "detections_each_round": [run["total_detections"] for run in runs]}


def saved_video_evidence(data):
    # 参数可变时，只有来源与参数都匹配，才把以前的截图作为本次实验的证据。
    matches = True
    try:
        for key in ("pt", "onnx"):
            saved = json.loads((ROOT / f"docs/{key}_640_report.json").read_text(encoding="utf-8"))
            matches &= (saved["source_sha256"] == data["source"]["sha256"]
                        and saved["model"]["sha256"] == data["models"][key]["info"]["sha256"]
                        and saved["predict_options"] == data["protocol"]["predict_options"])
        manifest = json.loads((ROOT / "docs/inference_comparison_checks.json").read_text(encoding="utf-8"))
        review = manifest["user_playback_review"]
        return {"matches_benchmark": bool(matches),
                "onnx_user_playback_confirmed": isinstance(review, dict) and review.get("onnx", "").startswith("confirmed by user"),
                "saved_playback_sha256_equal": manifest["playback_videos"]["pt"]["sha256"] == manifest["playback_videos"]["onnx"]["sha256"]}
    except (FileNotFoundError, KeyError):
        return {"matches_benchmark": False}


def markdown_report(data, raw_path):
    pt, onnx = (data["models"][key] for key in ("pt", "onnx"))
    a, b = pt["summary"], onnx["summary"]
    time_change = (b["mean_ms"] / a["mean_ms"] - 1) * 100
    comparison = (f"ONNX 平均每帧预测耗时降低 {-time_change:.1f}%" if time_change <= 0
                  else f"ONNX 平均每帧预测耗时增加 {time_change:.1f}%")
    env, source, protocol = data["environment"], data["source"], data["protocol"]
    lines = ["# YOLO PyTorch 与 ONNX 性能对比", "",
             "## 实验条件", "",
             f"- CPU：{env['cpu']}；操作系统：{env['os']}；Python：{env['python']}。",
             f"- 视频：`{source['path']}`，{source['frames']} 帧，{source['fps']:g} FPS，{source['width']}×{source['height']}。",
             f"- 同一权重来源：已校验导出报告中的源模型/ONNX SHA-256；视频 SHA-256：`{source['sha256']}`。",
             f"- 统一 CPU、FP32、{protocol['predict_options']['imgsz']}×{protocol['predict_options']['imgsz']}、rect=False、conf={protocol['predict_options']['conf']}、IoU={protocol['predict_options']['iou']}、batch=1、max_det=300。",
             f"- 各预热 {protocol['warmup_frames']} 帧；随后按 PT→ONNX 顺序交替测量，各完整运行 {protocol['repeats']} 轮。",
             "- 计时范围：逐帧 `model.predict(...)` 调用，包含预处理、网络前向及后处理；排除模型加载、预热、视频解码、绘图、日志和文件保存。",
             "- 使用各运行时的默认线程策略；ONNX 线程配置 0 表示运行时自动决定，不是零线程。线程和库版本详见原始 JSON。", "",
             "## 测量结果", "",
             "| 项目 | PyTorch `.pt` | ONNX |", "|---|---|---|",
             f"| 模型大小 | {pt['info']['file_bytes']:,} 字节（{pt['info']['file_bytes']/1024**2:.2f} MiB） | {onnx['info']['file_bytes']:,} 字节（{onnx['info']['file_bytes']/1024**2:.2f} MiB） |",
             f"| 平均每帧预测耗时 | {a['mean_ms']:.3f} ms | {b['mean_ms']:.3f} ms |",
             f"| 预测吞吐量 | {a['prediction_fps']:.2f} FPS | {b['prediction_fps']:.2f} FPS |",
             f"| 轮均耗时的样本标准差 | {a['round_mean_ms_stdev']:.3f} ms | {b['round_mean_ms_stdev']:.3f} ms |",
             f"| 正式测量总帧数 | {a['measured_frames']} | {b['measured_frames']} |",
             f"| 每轮逐帧检测总次数 | {a['detections_each_round']} | {b['detections_each_round']} |", "",
             "平均耗时 = 所有正式预测调用的总时间 / 总帧数；FPS = 总帧数 / 总时间。该 FPS 衡量预测调用吞吐量，视频播放仍为 30 FPS。", "",
             "| 轮次 | PyTorch 平均 ms/帧 | ONNX 平均 ms/帧 | PyTorch FPS | ONNX FPS |", "|---|---|---|---|---|"]
    for index, (p, o) in enumerate(zip(pt["runs"], onnx["runs"]), 1):
        lines.append(f"| {index} | {p['mean_ms']:.3f} | {o['mean_ms']:.3f} | {p['prediction_fps']:.2f} | {o['prediction_fps']:.2f} |")
    count_text = "全部一致" if data["checks"]["all_frame_counts_equal"] else "存在差异"
    lines += ["", "## 检测效果与解释", "",
              f"- 各轮逐帧检测数量{count_text}。统计为跨帧累计检测次数，不是独立物体数。"]
    evidence = data.get("saved_video_evidence", {})
    if evidence.get("matches_benchmark"):
        review_text = "已由用户确认可以播放" if evidence.get("onnx_user_playback_confirmed") else "尚待用户确认播放"
        hash_text = "相同" if evidence.get("saved_playback_sha256_equal") else "不同"
        lines += [f"- 第 3 步视频实验与本次来源/参数匹配；ONNX 视频{review_text}。两份 H.264 文件指纹{hash_text}。",
                  "- 同一第 180 帧可见两个 B3 目标和各自四个预测关键点，抽查图如下。侧面漏检、运动模糊等问题仍应如实保留。", "",
                  "![PyTorch，第180帧](docs/pt_640_frame180.jpg)", "",
                  "![ONNX，第180帧](docs/onnx_640_frame180.jpg)", ""]
    else:
        lines += ["- 本次实验与已归档视频证据不匹配或证据缺失，需要按本次来源和参数另行保存效果对照。"]
    lines += [
              "- 没有人工真值标注，未计算准确率、召回率或关键点误差；相同框数和可视化不表示所有浮点坐标完全相同。",
              f"- 本机本次实验：{comparison}；耗时比 PyTorch/ONNX 为 {a['mean_ms']/b['mean_ms']:.3f}。结果适用于这里记录的模型、硬件及默认运行时配置。",
              "- 速度差异可能受运行时算子优化、图优化、线程策略及系统负载影响；本实验未逐项消融这些因素。",
              f"- 原始 checkpoint 参数存储类型：{data['checkpoint_parameter_dtypes']}；两种模型本次都使用 FP32 进行 CPU 推理。ONNX 文件大小还受权重精度、图结构和文件组织影响，导出不保证文件更小。", "",
              "## 复现与原始记录", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -u benchmark.py --warmup 10 --repeats 3 --imgsz 640 --conf 0.25 --iou 0.7",
              "```", "",
              "已有结果时必须指定新的 `--output docs/benchmark_cpu_v2.json --report comparison_v2.md`，避免覆盖本次证据。",
              f"完整逐帧时间、每轮计数、模型指纹、实际执行顺序及环境见 [{raw_path}]({raw_path})。", ""]
    return "\n".join(lines)


def main():
    args = arguments()
    source = video_metadata(args.source)
    if args.warmup > source["frames"]:
        raise ValueError("Warmup frame count exceeds the video length")
    export_path = args.onnx_model.with_suffix(".export.json")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if file_sha256(args.pt_model) != export["source_sha256"] or file_sha256(args.onnx_model) != export["output_sha256"]:
        raise ValueError("Models do not match the supplied export provenance report")
    models = {}
    for key, path in (("pt", args.pt_model), ("onnx", args.onnx_model)):
        model, info = load_model(path)
        if info["input_shape"] and info["input_shape"][2:] != [args.imgsz, args.imgsz]:
            raise ValueError("Requested image size differs from the fixed ONNX input")
        models[key] = {"model": model, "info": info, "runs": []}
    if any(models['pt']['info'][field] != models['onnx']['info'][field] for field in ("names", "keypoint_shape", "end2end")):
        raise ValueError("Model tasks or prediction heads differ")
    options = prediction_options(args.conf, args.iou, args.imgsz, "cpu", models["pt"]["info"]["end2end"])
    for key, entry in models.items():
        print(f"Warmup {key}: {args.warmup} frames", flush=True)
        measure_video(entry["model"], args.source, options, args.warmup, warmup=True)
        entry["runtime"] = runtime_configuration(entry["model"])
    order = []
    for repeat in range(1, args.repeats + 1):
        for key, entry in models.items():
            label = f"Round {repeat}/{args.repeats} {key}"
            print(label, flush=True)
            run = measure_video(entry["model"], args.source, options, source["frames"], label=label)
            entry["runs"].append(run)
            order.append({"round": repeat, "model": key})
            print(f"{label} done: {run['mean_ms']:.3f} ms/frame, {run['prediction_fps']:.2f} FPS, {run['total_detections']} detections", flush=True)
    # 检查测量期间输入文件未被其他程序替换；报告与原始记录只在完整成功后保存。
    if file_sha256(args.source) != source["sha256"]:
        raise ValueError("Source video changed while benchmarking")
    for entry, path in ((models["pt"], args.pt_model), (models["onnx"], args.onnx_model)):
        if file_sha256(path) != entry["info"]["sha256"]:
            raise ValueError("Model file changed while benchmarking")
        entry["summary"] = summarize(entry["runs"])
        del entry["model"]
    baseline_counts = models["pt"]["runs"][0]["frame_detection_counts"]
    count_match = all(run["frame_detection_counts"] == baseline_counts for entry in models.values() for run in entry["runs"])
    # 从 checkpoint 文件读取存储类型；YOLO 加载后会转为浮点推理，不能反推保存精度。
    checkpoint = torch.load(args.pt_model, map_location="cpu", weights_only=False)
    saved_model = checkpoint.get("ema") or checkpoint.get("model")
    stored_dtypes = sorted({str(parameter.dtype) for parameter in saved_model.parameters()})
    data = {"measured_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
            "protocol": {"warmup_frames": args.warmup, "repeats": args.repeats, "predict_options": options,
                         "timing_scope": "Synchronous model.predict call including preprocess/forward/postprocess; excludes decode/render/save/load/warmup",
                         "execution_order": order},
            "environment": {"python": platform.python_version(), "os": platform.platform(), "cpu": get_cpu_info(),
                            "logical_cpu_count": os.cpu_count(), "torch_runtime": torch.__version__,
                            "packages": {name: importlib.metadata.version(name) for name in (
                                "ultralytics", "torch", "torchvision", "opencv-python", "numpy", "onnx", "onnxruntime", "onnxslim")}},
            "checkpoint_parameter_dtypes": stored_dtypes, "models": models,
            "checks": {"export_provenance_verified": True, "all_frame_counts_equal": count_match}}
    data["saved_video_evidence"] = saved_video_evidence(data)
    text = markdown_report(data, relative_path(args.output))
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with args.report.open("x", encoding="utf-8") as file:
        file.write(text)
    print(f"Raw benchmark: {args.output}\nComparison: {args.report}", flush=True)


if __name__ == "__main__":
    main()
