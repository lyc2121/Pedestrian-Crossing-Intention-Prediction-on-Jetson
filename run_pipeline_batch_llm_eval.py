"""
run_pipeline_batch_llm_eval.py
================================
对一个"检测已经跑完"的输出根目录做批量 LLM标注 + 评价。

根目录下每个子文件夹对应一个视频的检测产物，结构与 run_pipeline.py /
run_pipeline_batch.py 单视频输出的 output/<video_name>/ 完全一致，
子文件夹里应当已经包含：

    <video_name>_track_only.mp4     纯bbox+ID视频（喂给LLM标注用）
    <video_name>_prediction.csv     逐帧模型预测结果

本脚本对每个子文件夹依次执行 run_pipeline.py 后两步的逻辑（跳过检测）：

    [1/2] 调用 LLM 对 track_only 视频逐帧标注
              -> <video_name>_annotations.csv
    [2/2] 对比 LLM 标注（ground truth）与模型预测，计算评价指标
              -> <video_name>_overall_metrics.csv
              -> <video_name>_per_id_metrics.csv

每个子文件夹互不影响，某一个失败（缺文件/LLM报错/评价报错）不会中断整个
批量任务，最后汇总成功/失败/跳过情况到 batch_summary.csv。

用法：
    python run_pipeline_batch_llm_eval.py --input_dir ./evalution_output
    python run_pipeline_batch_llm_eval.py --input_dir ./evalution_output --force_llm
    python run_pipeline_batch_llm_eval.py --input_dir ./evalution_output --skip_existing
    python run_pipeline_batch_llm_eval.py --input_dir ./evalution_output --stop_on_error
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_SCRIPT = SCRIPT_DIR / "LLM for pedestrian state recognition.py"
EVAL_SCRIPT = SCRIPT_DIR / "evaluate_crossing_intention.py"


def detect_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if fps and fps > 0 else 25.0


def run_step(cmd, step_name):
    print(f"\n{'=' * 60}\n{step_name}\n{'=' * 60}")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def find_video_dirs(input_dir: Path):
    """
    根目录下，凡是子文件夹里同时存在
    <子文件夹名>_track_only.mp4 和 <子文件夹名>_prediction.csv 的，
    都认为是一个待处理的检测产物目录。
    """
    video_dirs = []
    skipped = []
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        track_only_video = sub / f"{name}_track_only.mp4"
        prediction_csv = sub / f"{name}_prediction.csv"
        if track_only_video.exists() and prediction_csv.exists():
            video_dirs.append((name, sub, track_only_video, prediction_csv))
        else:
            skipped.append(name)
    return video_dirs, skipped


def process_one(name, sub_dir, track_only_video, prediction_csv, args):
    """对单个子文件夹执行 LLM标注 + 评价，与 run_pipeline.py 的第2/3步逻辑一致。"""

    annotation_csv = sub_dir / f"{name}_annotations.csv"

    # ── 第1步：LLM 对 track_only 视频逐帧标注（已存在则跳过，节省API调用）──
    if annotation_csv.exists() and not args.force_llm:
        print(f"\n[1/2] LLM标注文件已存在，跳过调用API：{annotation_csv}")
    else:
        llm_cmd = [
            sys.executable, str(LLM_SCRIPT),
            "--video", str(track_only_video),
            "--output", str(sub_dir),
            "--sample-fps", str(args.llm_sample_fps),
            "--context-gap-sec", str(args.llm_context_gap_sec),
            "--model", args.llm_model,
            "--base-url", args.llm_base_url,
        ]
        if args.llm_api_key:
            llm_cmd += ["--api-key", args.llm_api_key]
        if args.force_llm:
            llm_cmd += ["--overwrite"]
        run_step(llm_cmd, f"[1/2] 调用 Qwen 对 {name} 的 track_only 视频逐帧标注")

        # LLM脚本按 <video.stem>_annotations.csv 命名，
        # track_only_video.stem 是 "<name>_track_only"，需要重命名成 "<name>_annotations.csv"
        generated = sub_dir / f"{track_only_video.stem}_annotations.csv"
        if generated != annotation_csv and generated.exists():
            generated.replace(annotation_csv)  # replace()在Windows上也会覆盖已存在的目标文件

    if not annotation_csv.exists():
        raise RuntimeError(f"LLM标注阶段未产出预期文件：{annotation_csv}")

    # ── 第2步：评价 ──────────────────────────────────────────────────
    fps = detect_fps(track_only_video)
    run_step(
        [
            sys.executable, str(EVAL_SCRIPT),
            "--annotations", str(annotation_csv),
            "--predictions", str(prediction_csv),
            "--output-dir", str(sub_dir),
            "--video-name", name,
            "--fps", str(fps),
            "--sample-interval", str(args.sample_interval),
            "--future-start-sec", str(args.future_start_sec),
            "--future-end-sec", str(args.future_end_sec),
        ],
        f"[2/2] 计算 {name} 的过街意图预测评价指标",
    )


def main():
    parser = argparse.ArgumentParser(
        description="对已完成检测的输出根目录批量做 LLM标注 + 评价（不重新跑检测）"
    )

    parser.add_argument("--input-dir", "--input_dir", "--input", dest="input_dir",
                         type=Path, required=True,
                         help="*_folder.py 打印的 results root；其下每个子文件夹是一个视频的检测产物")
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true",
                         help="若某子文件夹的 <name>_overall_metrics.csv 已存在，则跳过该子文件夹")
    parser.add_argument("--stop-on-error", "--stop_on_error", dest="stop_on_error", action="store_true",
                         help="某个子文件夹处理失败时立即终止批量任务（默认：记录失败后继续处理下一个）")

    # ---- LLM标注阶段 ----
    parser.add_argument("--llm_sample_fps", type=float, default=10)
    parser.add_argument("--llm_context_gap_sec", type=float, default=0.33)
    parser.add_argument("--llm_model", type=str, default=os.getenv("QWEN_MODEL", "qwen3.7-plus"))
    parser.add_argument("--llm_api_key", type=str, default=os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--llm_base_url", type=str,
                         default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--force-llm", "--force_llm", dest="force_llm", action="store_true", help="强制重新调用LLM标注")

    # ---- 评价阶段 ----
    parser.add_argument("--sample_interval", type=int, default=3)
    parser.add_argument("--future_start_sec", type=float, default=1.0)
    parser.add_argument("--future_end_sec", type=float, default=2.0)

    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"输入文件夹不存在：{input_dir}")

    video_dirs, skipped_dirs = find_video_dirs(input_dir)
    if not video_dirs:
        raise RuntimeError(
            f"文件夹中没有找到任何符合结构的子文件夹（需要包含 <name>_track_only.mp4 和 "
            f"<name>_prediction.csv）：{input_dir}"
        )

    print(f"=> 共找到 {len(video_dirs)} 个已检测子文件夹待处理：")
    for name, *_ in video_dirs:
        print(f"   - {name}")
    if skipped_dirs:
        print(f"=> 以下子文件夹缺少检测产物，已忽略：{skipped_dirs}")

    results = []  # {video, status, elapsed_sec, output_dir, error}
    batch_start = time.time()

    for idx, (name, sub_dir, track_only_video, prediction_csv) in enumerate(video_dirs, start=1):
        overall_metrics_csv = sub_dir / f"{name}_overall_metrics.csv"

        print(f"\n{'#' * 70}\n# [{idx}/{len(video_dirs)}] {name}\n{'#' * 70}")

        if args.skip_existing and overall_metrics_csv.exists():
            print(f"=> 已存在评价结果，跳过：{overall_metrics_csv}")
            results.append({
                "video": name, "status": "skipped",
                "elapsed_sec": 0.0, "output_dir": str(sub_dir), "error": "",
            })
            continue

        start = time.time()
        try:
            process_one(name, sub_dir, track_only_video, prediction_csv, args)
            elapsed = time.time() - start
            results.append({
                "video": name, "status": "success",
                "elapsed_sec": round(elapsed, 2), "output_dir": str(sub_dir), "error": "",
            })
            print(f"=> [{idx}/{len(video_dirs)}] 完成，耗时 {elapsed:.2f}s")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            results.append({
                "video": name, "status": "failed",
                "elapsed_sec": round(elapsed, 2), "output_dir": str(sub_dir),
                "error": f"exit code {e.returncode}",
            })
            print(f"=> [{idx}/{len(video_dirs)}] 失败（exit code {e.returncode}），继续处理下一个" if not args.stop_on_error
                  else f"=> [{idx}/{len(video_dirs)}] 失败（exit code {e.returncode}），已按--stop_on_error终止")
            if args.stop_on_error:
                break
        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "video": name, "status": "failed",
                "elapsed_sec": round(elapsed, 2), "output_dir": str(sub_dir),
                "error": str(e),
            })
            print(f"=> [{idx}/{len(video_dirs)}] 失败（{e}），继续处理下一个" if not args.stop_on_error
                  else f"=> [{idx}/{len(video_dirs)}] 失败（{e}），已按--stop_on_error终止")
            if args.stop_on_error:
                break

    total_elapsed = time.time() - batch_start

    # ── 汇总 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n批量处理汇总（总耗时 {total_elapsed:.2f}s）\n{'=' * 70}")
    for r in results:
        print(f"  [{r['status']:>7}] {r['video']:<40} {r['elapsed_sec']:>8.2f}s  {r['error']}")

    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n成功: {n_success}  失败: {n_failed}  跳过: {n_skipped}  共: {len(results)}")

    summary_csv = input_dir / "batch_llm_eval_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "status", "elapsed_sec", "output_dir", "error"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n批量汇总已保存到：{summary_csv.resolve()}")

    if n_failed and args.stop_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
