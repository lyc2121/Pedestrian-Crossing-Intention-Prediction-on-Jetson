import argparse
from pathlib import Path
import pandas as pd


# =========================
# 1. 参数配置（默认值与旧版硬编码一致，便于单独运行）
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比LLM标注(ground truth)与模型预测(prediction.csv)，计算过街意图预测的Accuracy/Precision/Recall/F1。"
    )

    parser.add_argument(
        "--annotations", type=Path,
        default=Path("Pedestrain_crossing_evalution/clip_track_only_annotations.csv"),
        help="LLM生成的annotations.csv路径。",
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("Pedestrain_crossing_evalution/clip_prediction.csv"),
        help="main_onnx.py生成的prediction.csv路径。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="overall_metrics.csv / per_id_metrics.csv的输出目录。",
    )
    parser.add_argument(
        "--video-name", type=str, default="",
        help="输出文件名前缀，例如传入clip则输出clip_overall_metrics.csv。默认不加前缀（与旧版行为一致）。",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="prediction.csv生成时使用的视频帧率，用于换算未来时间窗口对应的帧号。",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=3,
        help="每个行人ID的预测结果，每隔多少帧采样一次。",
    )
    parser.add_argument(
        "--future-start-sec", type=float, default=1.0,
        help="未来真实意图窗口起点（秒）。",
    )
    parser.add_argument(
        "--future-end-sec", type=float, default=2.0,
        help="未来真实意图窗口终点（秒）。",
    )

    return parser.parse_args()


def normalize_label(value) -> int:
    """
    将标签统一转换为二进制：

    crossing = 1
    not crossing / not_crossing = 0
    """
    if pd.isna(value):
        raise ValueError("发现空标签。")

    label = str(value).strip().lower()
    label = label.replace("_", " ").replace("-", " ")
    label = " ".join(label.split())

    if label == "crossing":
        return 1

    if label in {
        "not crossing",
        "notcrossing",
        "non crossing",
    }:
        return 0

    raise ValueError(f"无法识别的标签：{value!r}")


def normalize_person_id(value) -> str:
    """
    统一 person_id。

    避免一个文件中 ID 是 1，
    另一个文件中 ID 是 1.0。
    """
    if pd.isna(value):
        raise ValueError("发现空 person_id。")

    text = str(value).strip()

    try:
        number = float(text)

        if number.is_integer():
            return str(int(number))

    except ValueError:
        pass

    return text


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    计算 Accuracy、Precision、Recall 和 F1。

    crossing 为正类。
    """
    ground_truth = df["future_gt"]
    prediction = df["prediction"]

    tp = int(
        (
            (ground_truth == 1)
            & (prediction == 1)
        ).sum()
    )

    tn = int(
        (
            (ground_truth == 0)
            & (prediction == 0)
        ).sum()
    )

    fp = int(
        (
            (ground_truth == 0)
            & (prediction == 1)
        ).sum()
    )

    fn = int(
        (
            (ground_truth == 1)
            & (prediction == 0)
        ).sum()
    )

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main():
    args = parse_args()

    # =========================
    # 2. 读取 CSV
    # =========================
    annotations = pd.read_csv(args.annotations)
    predictions = pd.read_csv(args.predictions)

    annotation_required_columns = {
        "frame_id",
        "person_id",
        "state",
    }

    prediction_required_columns = {
        "frame_id",
        "person_id",
        "pred_intent",
    }

    missing_annotation_columns = (
        annotation_required_columns
        - set(annotations.columns)
    )

    missing_prediction_columns = (
        prediction_required_columns
        - set(predictions.columns)
    )

    if missing_annotation_columns:
        raise ValueError(
            f"{args.annotations.name} 缺少列："
            f"{sorted(missing_annotation_columns)}"
        )

    if missing_prediction_columns:
        raise ValueError(
            f"{args.predictions.name} 缺少列："
            f"{sorted(missing_prediction_columns)}"
        )

    # 只保留评价所需列
    annotations = annotations[
        [
            "frame_id",
            "person_id",
            "state",
        ]
    ].copy()

    predictions = predictions[
        [
            "frame_id",
            "person_id",
            "pred_intent",
        ]
    ].copy()

    # =========================
    # 3. 统一数据格式
    # =========================
    annotations["frame_id"] = pd.to_numeric(
        annotations["frame_id"],
        errors="raise",
    ).astype(int)

    predictions["frame_id"] = pd.to_numeric(
        predictions["frame_id"],
        errors="raise",
    ).astype(int)

    annotations["person_id"] = (
        annotations["person_id"]
        .map(normalize_person_id)
    )

    predictions["person_id"] = (
        predictions["person_id"]
        .map(normalize_person_id)
    )

    annotations["state_binary"] = (
        annotations["state"]
        .map(normalize_label)
    )

    predictions["prediction_binary"] = (
        predictions["pred_intent"]
        .map(normalize_label)
    )

    # 同一个 ID、同一帧只能有一条标注
    annotation_duplicates = annotations.duplicated(
        subset=[
            "frame_id",
            "person_id",
        ],
        keep=False,
    )

    if annotation_duplicates.any():
        duplicate_rows = annotations.loc[
            annotation_duplicates,
            [
                "frame_id",
                "person_id",
            ],
        ]

        raise ValueError(
            "annotations.csv 中存在重复的 "
            "frame_id + person_id：\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    # 同一个 ID、同一帧只能有一条预测
    prediction_duplicates = predictions.duplicated(
        subset=[
            "frame_id",
            "person_id",
        ],
        keep=False,
    )

    if prediction_duplicates.any():
        duplicate_rows = predictions.loc[
            prediction_duplicates,
            [
                "frame_id",
                "person_id",
            ],
        ]

        raise ValueError(
            "预测文件中存在重复的 "
            "frame_id + person_id：\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    # =========================
    # 4. 每个 ID 独立按 3 帧采样
    # =========================
    sampled_prediction_groups = []

    for person_id, person_predictions in predictions.groupby(
        "person_id",
        sort=False,
    ):
        person_predictions = (
            person_predictions
            .sort_values("frame_id")
            .copy()
        )

        # 每个 ID 都从自己的第一条预测帧开始采样
        first_frame = int(
            person_predictions["frame_id"].min()
        )

        sampled = person_predictions[
            (
                person_predictions["frame_id"]
                - first_frame
            )
            % args.sample_interval
            == 0
        ].copy()

        sampled_prediction_groups.append(sampled)

    if not sampled_prediction_groups:
        raise RuntimeError(
            "预测文件中没有可用于评价的数据。"
        )

    sampled_predictions = pd.concat(
        sampled_prediction_groups,
        ignore_index=True,
    )

    # =========================
    # 5. 为每个 ID 建立独立标注表
    # =========================
    annotations_by_id = {
        person_id: person_annotations
        .sort_values("frame_id")
        .copy()

        for person_id, person_annotations
        in annotations.groupby(
            "person_id",
            sort=False,
        )
    }

    future_start_offset = int(
        round(args.fps * args.future_start_sec)
    )

    future_end_offset = int(
        round(args.fps * args.future_end_sec)
    )

    evaluation_rows = []

    # =========================
    # 6. 构造未来真实意图
    # =========================
    for row in sampled_predictions.itertuples(
        index=False
    ):
        current_frame = int(row.frame_id)
        person_id = row.person_id

        future_start_frame = (
            current_frame
            + future_start_offset
        )

        future_end_frame = (
            current_frame
            + future_end_offset
        )

        # 只查询相同 person_id 的标注
        person_annotations = annotations_by_id.get(
            person_id
        )

        if person_annotations is None:
            # 标注文件中完全没有该 ID
            future_gt = 0

        else:
            future_annotations = person_annotations[
                (
                    person_annotations["frame_id"]
                    >= future_start_frame
                )
                & (
                    person_annotations["frame_id"]
                    <= future_end_frame
                )
            ]

            # 未来 1～2 秒内出现任意 crossing
            # 则未来真实意图为 crossing
            has_crossing = (
                future_annotations["state_binary"]
                == 1
            ).any()

            future_gt = int(has_crossing)

            # 当没有 crossing 时，包括以下情况：
            # 1. 未来全部为 not crossing；
            # 2. 未来没有该 ID；
            # 3. 未来超过该 ID 的标注范围；
            # 均自动得到 future_gt = 0

        evaluation_rows.append(
            {
                "frame_id": current_frame,
                "person_id": person_id,
                "prediction": int(
                    row.prediction_binary
                ),
                "future_gt": future_gt,
            }
        )

    evaluation_df = pd.DataFrame(
        evaluation_rows
    )

    if evaluation_df.empty:
        raise RuntimeError(
            "没有生成任何评价样本。"
        )

    # =========================
    # 7. 所有 ID 合并后的总体指标
    # =========================
    overall_metrics = calculate_metrics(
        evaluation_df
    )

    overall_df = pd.DataFrame(
        [overall_metrics]
    )

    # =========================
    # 8. 每个 ID 单独计算指标
    # =========================
    per_id_rows = []

    for person_id, person_evaluation in (
        evaluation_df.groupby(
            "person_id",
            sort=True,
        )
    ):
        person_metrics = calculate_metrics(
            person_evaluation
        )

        per_id_rows.append(
            {
                "person_id": person_id,
                "Accuracy": person_metrics[
                    "Accuracy"
                ],
                "Precision": person_metrics[
                    "Precision"
                ],
                "Recall": person_metrics[
                    "Recall"
                ],
                "F1": person_metrics["F1"],
            }
        )

    per_id_df = pd.DataFrame(
        per_id_rows
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    overall_df[metric_columns] = (
        overall_df[metric_columns]
        .round(6)
    )

    per_id_df[metric_columns] = (
        per_id_df[metric_columns]
        .round(6)
    )

    # =========================
    # 9. 输出结果
    # =========================
    print(
        "\n所有 ID 合并后的总体指标："
    )

    print(
        overall_df.to_string(
            index=False
        )
    )

    print(
        "\n每个 ID 单独的指标："
    )

    print(
        per_id_df.to_string(
            index=False
        )
    )

    prefix = f"{args.video_name}_" if args.video_name else ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_output = args.output_dir / f"{prefix}overall_metrics.csv"
    per_id_output = args.output_dir / f"{prefix}per_id_metrics.csv"

    overall_df.to_csv(
        overall_output,
        index=False,
        encoding="utf-8-sig",
    )

    per_id_df.to_csv(
        per_id_output,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n总体指标已保存到："
        f"{overall_output.resolve()}"
    )

    print(
        f"每个 ID 指标已保存到："
        f"{per_id_output.resolve()}"
    )


if __name__ == "__main__":
    main()