"""
aggregate_crossing_intention_metrics.py
========================================
对一个文件夹下所有已经跑完 run_pipeline.py 的视频（每个视频一个子文件夹，
里面有 <name>_prediction.csv 和 <name>_annotations.csv），把所有视频、
所有帧的预测样本合并在一起，计算一个整体的 Accuracy/Precision/Recall/F1。

口径：先把所有视频的逐帧评价样本（frame_id, person_id, prediction, future_gt）
拼到一张表里，再统一算一次混淆矩阵 —— 是"所有帧的整体指标"（微平均），
不是把每个视频已经算好的指标取平均。

复用 evaluate_crossing_intention.py 里已经验证过的核心逻辑（标签归一化、
person_id归一化、按sample-interval采样、future_gt窗口构造），保证口径
与单视频评价完全一致。

用法：
    python aggregate_crossing_intention_metrics.py --input_dir ./evalution_output
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from evaluate_crossing_intention import (
    normalize_label,
    normalize_person_id,
    calculate_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并文件夹下所有视频的逐帧预测结果，计算整体Accuracy/Precision/Recall/F1。"
    )
    parser.add_argument("--input-dir", "--input_dir", "--input", dest="input_dir",
                         type=Path, required=True,
                         help="*_folder.py 打印的 results root")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=None,
                         help="结果CSV的输出目录，默认与--input_dir相同")
    parser.add_argument("--sample-interval", type=int, default=3,
                         help="每个行人ID的预测结果，每隔多少帧采样一次（需与run_pipeline.py评价阶段一致）")
    parser.add_argument("--future-start-sec", type=float, default=1.0,
                         help="未来真实意图窗口起点（秒）")
    parser.add_argument("--future-end-sec", type=float, default=2.0,
                         help="未来真实意图窗口终点（秒）")
    parser.add_argument("--default-fps", type=float, default=25.0,
                         help="当无法从视频文件探测到fps时使用的兜底值")
    return parser.parse_args()


def detect_fps(video_path: Path, default_fps: float) -> float:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if fps and fps > 0 else default_fps


def find_one(folder: Path, suffix: str):
    matches = sorted(folder.glob(f"*{suffix}"))
    return matches[0] if matches else None


def build_evaluation_df(prediction_csv: Path, annotation_csv: Path, fps: float, args) -> pd.DataFrame:
    """对单个视频重建逐帧评价样本表：frame_id, person_id, prediction, future_gt。

    与 evaluate_crossing_intention.py 第4~6步的采样/未来窗口逻辑完全一致，
    只是这里把中间结果（原脚本算完即丢）保留下来，供多视频合并使用。
    """
    annotations = pd.read_csv(annotation_csv)
    predictions = pd.read_csv(prediction_csv)

    annotation_required_columns = {"frame_id", "person_id", "state"}
    prediction_required_columns = {"frame_id", "person_id", "pred_intent"}

    missing_annotation_columns = annotation_required_columns - set(annotations.columns)
    missing_prediction_columns = prediction_required_columns - set(predictions.columns)
    if missing_annotation_columns:
        raise ValueError(f"{annotation_csv.name} 缺少列：{sorted(missing_annotation_columns)}")
    if missing_prediction_columns:
        raise ValueError(f"{prediction_csv.name} 缺少列：{sorted(missing_prediction_columns)}")

    annotations = annotations[["frame_id", "person_id", "state"]].copy()
    predictions = predictions[["frame_id", "person_id", "pred_intent"]].copy()

    annotations["frame_id"] = pd.to_numeric(annotations["frame_id"], errors="raise").astype(int)
    predictions["frame_id"] = pd.to_numeric(predictions["frame_id"], errors="raise").astype(int)

    annotations["person_id"] = annotations["person_id"].map(normalize_person_id)
    predictions["person_id"] = predictions["person_id"].map(normalize_person_id)

    annotations["state_binary"] = annotations["state"].map(normalize_label)
    predictions["prediction_binary"] = predictions["pred_intent"].map(normalize_label)

    if annotations.duplicated(subset=["frame_id", "person_id"], keep=False).any():
        raise ValueError(f"{annotation_csv.name} 中存在重复的 frame_id + person_id")

    if predictions.duplicated(subset=["frame_id", "person_id"], keep=False).any():
        raise ValueError(f"{prediction_csv.name} 中存在重复的 frame_id + person_id")

    sampled_prediction_groups = []
    for person_id, person_predictions in predictions.groupby("person_id", sort=False):
        person_predictions = person_predictions.sort_values("frame_id").copy()
        first_frame = int(person_predictions["frame_id"].min())
        sampled = person_predictions[
            (person_predictions["frame_id"] - first_frame) % args.sample_interval == 0
        ].copy()
        sampled_prediction_groups.append(sampled)

    if not sampled_prediction_groups:
        return pd.DataFrame(columns=["frame_id", "person_id", "prediction", "future_gt"])

    sampled_predictions = pd.concat(sampled_prediction_groups, ignore_index=True)

    annotations_by_id = {
        person_id: person_annotations.sort_values("frame_id").copy()
        for person_id, person_annotations in annotations.groupby("person_id", sort=False)
    }

    future_start_offset = int(round(fps * args.future_start_sec))
    future_end_offset = int(round(fps * args.future_end_sec))

    evaluation_rows = []
    for row in sampled_predictions.itertuples(index=False):
        current_frame = int(row.frame_id)
        person_id = row.person_id
        future_start_frame = current_frame + future_start_offset
        future_end_frame = current_frame + future_end_offset

        person_annotations = annotations_by_id.get(person_id)
        if person_annotations is None:
            future_gt = 0
        else:
            future_annotations = person_annotations[
                (person_annotations["frame_id"] >= future_start_frame)
                & (person_annotations["frame_id"] <= future_end_frame)
            ]
            future_gt = int((future_annotations["state_binary"] == 1).any())

        evaluation_rows.append({
            "frame_id": current_frame,
            "person_id": person_id,
            "prediction": int(row.prediction_binary),
            "future_gt": future_gt,
        })

    return pd.DataFrame(evaluation_rows)


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"输入文件夹不存在：{input_dir}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not video_dirs:
        raise RuntimeError(f"{input_dir} 下没有找到任何子文件夹")

    all_rows = []
    per_video_summary = []

    for video_dir in video_dirs:
        prediction_csv = find_one(video_dir, "_prediction.csv")
        annotation_csv = find_one(video_dir, "_annotations.csv")

        if prediction_csv is None or annotation_csv is None:
            print(f"[跳过] {video_dir.name}：缺少 prediction.csv 或 annotations.csv")
            continue

        video_file = find_one(video_dir, "_track_only.mp4") or find_one(video_dir, "_full.mp4")
        fps = detect_fps(video_file, args.default_fps) if video_file else args.default_fps

        try:
            evaluation_df = build_evaluation_df(prediction_csv, annotation_csv, fps, args)
        except ValueError as e:
            print(f"[跳过] {video_dir.name}：{e}")
            continue

        if evaluation_df.empty:
            print(f"[跳过] {video_dir.name}：没有可用于评价的样本")
            continue

        evaluation_df["video"] = video_dir.name
        all_rows.append(evaluation_df)

        video_metrics = calculate_metrics(evaluation_df)
        video_metrics["video"] = video_dir.name
        video_metrics["num_samples"] = len(evaluation_df)
        video_metrics["fps_used"] = round(fps, 2)
        per_video_summary.append(video_metrics)

        print(f"[完成] {video_dir.name}：{len(evaluation_df)} 条帧级样本，fps={fps:.2f}")

    if not all_rows:
        raise RuntimeError("没有任何视频产出可用的评价样本，无法计算整体指标。")

    combined_df = pd.concat(all_rows, ignore_index=True)

    overall_metrics = calculate_metrics(combined_df)
    overall_metrics["num_videos"] = len(per_video_summary)
    overall_metrics["num_samples"] = len(combined_df)

    metric_columns = ["Accuracy", "Precision", "Recall", "F1"]

    overall_df = pd.DataFrame([overall_metrics])
    overall_df[metric_columns] = overall_df[metric_columns].round(6)

    per_video_df = pd.DataFrame(per_video_summary)
    per_video_df[metric_columns] = per_video_df[metric_columns].round(6)
    per_video_df = per_video_df[["video", "num_samples", "fps_used"] + metric_columns]

    print(f"\n{'=' * 70}\n所有视频、所有帧合并后的整体指标"
          f"（{len(per_video_summary)}个视频，{len(combined_df)}条样本）\n{'=' * 70}")
    print(overall_df.to_string(index=False))

    print("\n各视频单独的指标（供对比参考）：")
    print(per_video_df.to_string(index=False))

    overall_output = output_dir / "aggregate_overall_metrics.csv"
    per_video_output = output_dir / "aggregate_per_video_metrics.csv"

    overall_df.to_csv(overall_output, index=False, encoding="utf-8-sig")
    per_video_df.to_csv(per_video_output, index=False, encoding="utf-8-sig")

    print(f"\n整体指标已保存到：{overall_output.resolve()}")
    print(f"各视频指标已保存到：{per_video_output.resolve()}")


if __name__ == "__main__":
    main()
