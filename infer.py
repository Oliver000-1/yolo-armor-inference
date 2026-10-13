"""Run the supplied four-keypoint armor model on a locally recorded video."""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".local"))

import cv2
import numpy as np
from ultralytics import YOLO


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--source", type=Path, help="Your original D414 video")
    parser.add_argument("--output", type=Path, default=Path("outputs/armor_result.mp4"))
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device, e.g. 0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--check-model", action="store_true", help="Inspect model without a video")
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.imgsz <= 0:
        parser.error("conf must be in (0, 1] and imgsz must be positive")
    if not args.check_model and args.source is None:
        parser.error("--source is required unless using --check-model")
    return args


def load_model(path):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    model = YOLO(str(path))
    shape = list(getattr(model.model.model[-1], "kpt_shape", []))
    info = {"task": model.task, "names": model.names, "keypoint_shape": shape}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if model.task != "pose" or not shape or shape[0] != 4:
        raise ValueError("Expected a pose model with exactly four keypoints per target")
    expected = {"B1", "B2", "B3", "R1", "R2", "R3"}
    if set(model.names.values()) != expected:
        raise ValueError("Model class names do not match the assignment's six classes")
    return model, info


def annotate(frame, result):
    """Read keypoints from model output; box corners are never used as keypoints."""
    count = len(result.boxes)
    if count == 0:
        return frame, 0
    if result.keypoints is None or len(result.keypoints.data) != count:
        raise ValueError("Missing keypoints or mismatched box/keypoint instances")
    boxes = result.boxes.cpu().numpy().data
    points = result.keypoints.cpu().numpy().data
    if points.shape[1] != 4:
        raise ValueError("Expected four predicted keypoints")
    colors = [(0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 100, 240)]
    for index, (box, target_points) in enumerate(zip(boxes, points), start=1):
        x1, y1, x2, y2 = map(int, box[:4])
        confidence, class_id = float(box[-2]), int(box[-1])
        color = colors[(index - 1) % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{index} {result.names[class_id]} {confidence:.2f}"
        cv2.putText(frame, label, (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        for number, point in enumerate(target_points, start=1):
            if not np.isfinite(point[:2]).all():
                continue
            px, py = map(lambda value: int(round(float(value))), point[:2])
            if not (0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]):
                continue
            # Same instance color and matching #id make ownership explicit.
            cv2.circle(frame, (px, py), 6, (0, 0, 0), -1)
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.putText(frame, f"{index}:{number}", (px + 7, max(15, py - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame, count


def run(args):
    model, info = load_model(args.model)
    if args.check_model:
        return
    if not args.source.is_file():
        raise FileNotFoundError(f"Input video not found: {args.source}")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    report_path = args.output.with_suffix(".json")
    screenshot_path = args.output.with_suffix(".jpg")
    for path in (args.output, report_path, screenshot_path):
        if path.exists():
            raise FileExistsError(f"Use a new output name; already exists: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.source))
    writer = None
    frames = detected_frames = detections = 0
    best_score = -1.0
    start = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open input video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input has invalid FPS; convert it to a constant-frame-rate MP4")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                if width % 2 or height % 2:
                    raise ValueError("Use an input video with even width and height")
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Cannot create MP4 writer")
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   device=args.device, verbose=False)[0]
            annotated, count = annotate(frame, result)
            writer.write(annotated)
            frames += 1
            detections += count
            detected_frames += int(count > 0)
            if count:
                score = float(result.boxes.conf.max().item())
                if score > best_score:
                    encoded_ok, encoded = cv2.imencode(".jpg", annotated)
                    if not encoded_ok:
                        raise RuntimeError("Screenshot encoding failed")
                    screenshot_path.write_bytes(encoded.tobytes())
                    best_score = score
            if frames % 30 == 0:
                print(f"Processed {frames} frames; {detected_frames} contain detections")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not frames:
        raise RuntimeError("Input video contains no decodable frames")
    # Decode the entire saved video, not just its first frame.
    saved = cv2.VideoCapture(str(args.output))
    verified_frames = 0
    try:
        while saved.read()[0]:
            verified_frames += 1
    finally:
        saved.release()
    if verified_frames != frames:
        raise RuntimeError(f"Output validation failed: {verified_frames}/{frames} frames")
    warnings = []
    if expected_frames > 0 and frames != expected_frames:
        warnings.append("Decoded frame count differs from input metadata; check input integrity")
    if detected_frames == 0:
        warnings.append("No detections: check the target, lighting, model and confidence threshold")
    if args.output.stat().st_size > 100 * 1024 * 1024:
        warnings.append("Result exceeds the assignment's 100 MiB video limit")
    report = {"model": info, "source": str(args.source), "output": str(args.output),
              "device": args.device, "confidence_threshold": args.conf, "image_size": args.imgsz,
              "frames": frames, "input_metadata_frames": expected_frames, "fps": fps,
              "duration_seconds": frames / fps, "detected_frames": detected_frames,
              "total_detections": detections, "verified_output_frames": verified_frames,
              "elapsed_seconds": round(time.perf_counter() - start, 2),
              "screenshot": str(screenshot_path) if best_score >= 0 else None,
              "warnings": warnings}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Finished. Output is silent; original video remains unchanged.")


if __name__ == "__main__":
    run(arguments())
