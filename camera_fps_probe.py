import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


DEFAULT_RESOLUTIONS = (
    (1920, 1080),
    (1280, 720),
    (640, 480),
)
DEFAULT_FPS_TARGETS = (30, 60)


@dataclass
class ProbeResult:
    device: int
    backend: str
    width_req: int
    height_req: int
    fps_req: int
    width_actual: int
    height_actual: int
    fps_reported: float
    frames: int
    elapsed_s: float
    read_fps: float
    avg_interval_ms: float
    p95_interval_ms: float
    low_diff_ratio: float
    avg_frame_diff: float
    verdict: str


def parse_resolutions(raw: Optional[str]) -> Iterable[tuple[int, int]]:
    if not raw:
        return DEFAULT_RESOLUTIONS

    result = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Bad resolution '{item}', expected WIDTHxHEIGHT")
        w, h = item.split("x", 1)
        result.append((int(w), int(h)))
    return result


def parse_fps_targets(raw: Optional[str]) -> Iterable[int]:
    if not raw:
        return DEFAULT_FPS_TARGETS
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def open_capture(device: int, backend: str) -> cv2.VideoCapture:
    if backend == "dshow":
        return cv2.VideoCapture(device, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(device, cv2.CAP_MSMF)
    return cv2.VideoCapture(device)


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def frame_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return small.astype(np.int16)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def make_verdict(
    fps_req: int,
    read_fps: float,
    low_diff_ratio: float,
    avg_frame_diff: float,
) -> str:
    fps_ok = read_fps >= fps_req * 0.9
    motion_enough = avg_frame_diff >= 2.0
    duplicates_low = low_diff_ratio <= 0.20

    if fps_ok and duplicates_low:
        return "PASS"
    if fps_ok and not motion_enough:
        return "MOTION_TOO_LOW"
    if fps_ok:
        return "FPS_OK_BUT_MAY_REPEAT"
    return "LOW_FPS"


def probe_once(
    device: int,
    backend: str,
    width: int,
    height: int,
    fps: int,
    duration_s: float,
    warmup_s: float,
    diff_threshold: float,
    preview: bool,
) -> ProbeResult:
    cap = open_capture(device, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device} with backend {backend}")

    try:
        configure_capture(cap, width, height, fps)
        time.sleep(0.3)

        warmup_end = time.perf_counter() + warmup_s
        while time.perf_counter() < warmup_end:
            cap.read()

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))

        frames = 0
        low_diff_count = 0
        diff_sum = 0.0
        diff_count = 0
        intervals_ms: list[float] = []
        previous_sig: Optional[np.ndarray] = None
        previous_time: Optional[float] = None

        start = time.perf_counter()
        end = start + duration_s
        while time.perf_counter() < end:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                continue

            frames += 1
            if previous_time is not None:
                intervals_ms.append((now - previous_time) * 1000.0)
            previous_time = now

            sig = frame_signature(frame)
            if previous_sig is not None:
                diff = float(np.mean(np.abs(sig - previous_sig)))
                diff_sum += diff
                diff_count += 1
                if diff < diff_threshold:
                    low_diff_count += 1
            previous_sig = sig

            if preview:
                cv2.putText(
                    frame,
                    f"{actual_w}x{actual_h} target {fps}fps",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("camera_fps_probe - press q to stop preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        elapsed = time.perf_counter() - start
        read_fps = frames / elapsed if elapsed > 0 else 0.0
        low_diff_ratio = low_diff_count / diff_count if diff_count else 0.0
        avg_frame_diff = diff_sum / diff_count if diff_count else 0.0
        verdict = make_verdict(fps, read_fps, low_diff_ratio, avg_frame_diff)

        return ProbeResult(
            device=device,
            backend=backend,
            width_req=width,
            height_req=height,
            fps_req=fps,
            width_actual=actual_w,
            height_actual=actual_h,
            fps_reported=reported_fps,
            frames=frames,
            elapsed_s=elapsed,
            read_fps=read_fps,
            avg_interval_ms=float(np.mean(intervals_ms)) if intervals_ms else 0.0,
            p95_interval_ms=percentile(intervals_ms, 95),
            low_diff_ratio=low_diff_ratio,
            avg_frame_diff=avg_frame_diff,
            verdict=verdict,
        )
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()


def print_result(result: ProbeResult) -> None:
    print(
        f"{result.backend:5s} "
        f"req={result.width_req}x{result.height_req}@{result.fps_req:<2d} "
        f"actual={result.width_actual}x{result.height_actual} "
        f"reported={result.fps_reported:5.1f} "
        f"read={result.read_fps:5.1f}fps "
        f"avg_dt={result.avg_interval_ms:5.1f}ms "
        f"p95_dt={result.p95_interval_ms:5.1f}ms "
        f"low_diff={result.low_diff_ratio * 100:5.1f}% "
        f"diff={result.avg_frame_diff:5.2f} "
        f"{result.verdict}"
    )


def save_csv(path: Path, results: list[ProbeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProbeResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether a USB camera can really deliver 30/60fps at common "
            "resolutions. Move a hand or tag in front of the lens during the test "
            "so duplicate-frame detection is meaningful."
        )
    )
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--backend",
        choices=("auto", "dshow", "msmf", "all"),
        default="dshow",
        help="Windows camera backend to test",
    )
    parser.add_argument(
        "--res",
        default=None,
        help="Comma-separated resolutions, for example 1920x1080,1280x720,640x480",
    )
    parser.add_argument("--fps", default=None, help="Comma-separated fps targets, for example 30,60")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds per mode")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup seconds per mode")
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=1.0,
        help="Mean pixel diff below this is counted as a near-duplicate frame",
    )
    parser.add_argument("--preview", action="store_true", help="Show camera preview while testing")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.backend == "all":
        backends = ("dshow", "msmf", "auto")
    else:
        backends = (args.backend,)

    resolutions = list(parse_resolutions(args.res))
    fps_targets = list(parse_fps_targets(args.fps))
    results: list[ProbeResult] = []

    print("Move a hand or the tag continuously in front of the camera during this test.")
    print("Verdict guide: PASS = likely real mode, MOTION_TOO_LOW = scene too static, LOW_FPS = cannot keep target.\n")

    for backend in backends:
        for width, height in resolutions:
            for fps in fps_targets:
                try:
                    result = probe_once(
                        device=args.device,
                        backend=backend,
                        width=width,
                        height=height,
                        fps=fps,
                        duration_s=args.duration,
                        warmup_s=args.warmup,
                        diff_threshold=args.diff_threshold,
                        preview=args.preview,
                    )
                    results.append(result)
                    print_result(result)
                except RuntimeError as exc:
                    print(f"{backend:5s} req={width}x{height}@{fps:<2d} ERROR: {exc}")

    if args.csv:
        csv_path = Path(args.csv)
        save_csv(csv_path, results)
        print(f"\nSaved CSV: {csv_path}")

    print("\nBest 60fps candidates:")
    candidates = [r for r in results if r.fps_req == 60 and r.read_fps >= 54.0]
    candidates.sort(key=lambda r: (r.verdict != "PASS", -r.read_fps, r.p95_interval_ms))
    if not candidates:
        print("  None found. Try lower resolution, another backend, or another USB port/camera.")
    else:
        for result in candidates[:5]:
            print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
