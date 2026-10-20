"""Export the supplied armor pose model to ONNX and validate it on CPU."""
# 阅读顺序：main() 检查文件 → 导出 → 检查结构 → CPU 试运行 → 保存报告。
# 导出不会重新训练，也不修改原始 .pt；ONNX 是供其他推理工具读取的模型格式。
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local"))
# 依赖由 requirements.txt 安装，避免导出时自动联网修改环境。
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def sha256(path):
    # 文件指纹：相同内容产生相同指纹，用于记录权重来源及检查原文件未变化。
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "models/best.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")
    return args


def validate(path, imgsz, expected_names):
    # checker 检查图结构；这一步成功不等于真实视频检测效果已经通过。
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    metadata = {item.key: item.value for item in graph.metadata_props}
    names = ast.literal_eval(metadata.get("names", "{}"))
    shape = ast.literal_eval(metadata.get("kpt_shape", "[]"))
    if metadata.get("task") != "pose" or list(shape) != [4, 2] or names != expected_names:
        raise ValueError("ONNX metadata lost the pose task, class names or four keypoints")

    # 明确选择 CPU，避免在同一台电脑上误用不同硬件进行比较。
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    expected_shape = [1, 3, imgsz, imgsz]
    if len(inputs) != 1 or inputs[0].shape != expected_shape or inputs[0].type != "tensor(float)":
        raise ValueError("Expected a fixed FP32 input of shape [1, 3, imgsz, imgsz]")
    if any(item.data_type == onnx.TensorProto.FLOAT16 for item in graph.graph.initializer):
        raise ValueError("Unexpected FP16 weights in this FP32 export")

    # 全零图片只验证模型能执行，不用于评价装甲板识别效果或推理速度。
    outputs = session.run(None, {inputs[0].name: np.zeros(expected_shape, dtype=np.float32)})
    if not outputs or any(array.size == 0 or not np.isfinite(array).all() for array in outputs):
        raise ValueError("ONNX Runtime returned empty or non-finite output")
    return {
        "checker_passed": True,
        "cpu_smoke_test_passed": True,
        "input_shape": inputs[0].shape,
        "input_type": inputs[0].type,
        "output_shapes": [list(array.shape) for array in outputs],
        "providers": session.get_providers(),
        "task": metadata["task"],
        "names": names,
        "keypoint_shape": shape,
        "end2end": ast.literal_eval(metadata.get("end2end", "False")),
        "opset": [{"domain": item.domain, "version": item.version} for item in graph.opset_import],
    }


def main():
    args = arguments()
    source, output = args.model.resolve(), args.output.resolve()
    report_path = output.with_suffix(".export.json")
    if not source.is_file() or source.suffix.lower() != ".pt":
        raise ValueError("--model must point to an existing .pt file")
    if output.suffix.lower() != ".onnx":
        raise ValueError("--output must end with .onnx")
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new --output")

    original_hash = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 在临时目录导出并校验，成功后才放到目标位置，避免误用未校验的模型。
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as staging:
        working_source = Path(staging) / source.name
        shutil.copyfile(source, working_source)
        model = YOLO(str(working_source))
        keypoint_shape = list(getattr(model.model.model[-1], "kpt_shape", []))
        expected_classes = {"B1", "B2", "B3", "R1", "R2", "R3"}
        if model.task != "pose" or keypoint_shape != [4, 2] or set(model.names.values()) != expected_classes:
            raise ValueError("Expected the supplied six-class, four-keypoint armor pose model")
        architecture = model.model.yaml.get("yaml_file")
        # nms=None 保留本次模型的普通预测分支；若权重采用端到端分支则对应使用 False。
        # 当前 best.pt 的 end2end=False，ONNX 输出原始预测，后续由推理工具做 NMS。
        original_end2end = bool(getattr(model.model, "end2end", False))
        settings = {
            "format": "onnx", "device": "cpu", "quantize": 32,
            "imgsz": args.imgsz, "batch": 1, "dynamic": False,
            "simplify": True, "opset": 18, "nms": False if original_end2end else None,
        }
        exported = Path(model.export(**settings))
        checks = validate(exported, args.imgsz, model.names)
        if checks["end2end"] != original_end2end:
            raise ValueError("The export unexpectedly changed the prediction head")
        if sha256(source) != original_hash or sha256(working_source) != original_hash:
            raise ValueError("Source checkpoint changed during export")

        report = {
            "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "source_sha256": original_hash,
            "source_bytes": source.stat().st_size,
            "output": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
            "output_sha256": sha256(exported),
            "output_bytes": exported.stat().st_size,
            "architecture": architecture,
            "settings": settings,
            "validation": checks,
            "environment": {"python": platform.python_version(), **{
                name: importlib.metadata.version(name)
                for name in ("ultralytics", "torch", "numpy", "onnx", "onnxruntime", "onnxslim")
            }},
        }
        # xb/x 表示只创建新文件，已有同名文件时拒绝覆盖。
        with exported.open("rb") as reader, output.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        with report_path.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")

    print(f"Export OK: {output}")
    print(f"Report: {report_path}")
    print(f"Input: {checks['input_shape']} (FP32), output shapes: {checks['output_shapes']}")
    print("ONNX checker and CPU smoke test passed; real-video inference is the next step.")


if __name__ == "__main__":
    main()
