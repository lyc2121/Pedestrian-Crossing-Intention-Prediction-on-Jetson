#!/usr/bin/env python3
"""
使用 Qwen3.7-Plus 对视频中的行人进行 PIE 风格的逐帧过街动作标注。

输入视频要求：
1. 视频中已经绘制了行人 bounding box。
2. 每个 bounding box 附近具有可见的行人 ID，例如 person,ID:12。
3. 视频中不能显示待评估模型预测的 crossing/not crossing 结果，
   否则会产生标签泄漏。

标注类别：
- crossing:
    当前行人正在进入、穿越或横跨自车预期行驶路径。
- not_crossing:
    当前没有穿越自车预期行驶路径，包括等待、站立、沿道路行走、
    靠近路边但尚未过街，以及在与自车无关的区域横穿。

输出：
annotations.csv

CSV字段：
frame_id,time_sec,person_id,state,confidence,review_required,reason
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
from tqdm import tqdm


# ============================================================
# 基本配置
# ============================================================

VALID_STATES = {
    "crossing",
    "not_crossing",
}

CSV_FIELDS = [
    "frame_id",
    "time_sec",
    "person_id",
    "state",
    "confidence",
    "review_required",
    "reason",
]


@dataclass(frozen=True)
class VideoInfo:
    """视频基本信息。"""

    fps: float
    total_frames: int


@dataclass(frozen=True)
class TargetFrame:
    """一次待标注的目标帧及其前后文帧。"""

    frame_id: int
    time_sec: float
    previous_frame_id: int
    next_frame_id: int


# ============================================================
# Qwen提示词
# ============================================================

SYSTEM_PROMPT = """
You are a rigorous traffic-scene annotation assistant.

Your task is to generate PIE-style CURRENT pedestrian crossing-action labels.

The annotation is about the pedestrian's CURRENT ACTION in the TARGET frame,
not the pedestrian's future intention.

State definitions:

1. crossing
The pedestrian is currently entering, moving through, or traversing the ego
vehicle's expected driving path or relevant drivable corridor.

2. not_crossing
The pedestrian is not currently crossing the ego vehicle's expected path.

This includes:
- waiting near the road;
- standing on the sidewalk or roadside;
- walking along a sidewalk;
- walking approximately parallel to the road;
- approaching the curb but not yet entering the ego path;
- standing inside the road without observable crossing motion;
- crossing an irrelevant road area that does not intersect the ego path.

Important rules:

- You will receive three images:
  1. a previous context frame;
  2. the TARGET frame;
  3. a next context frame.

- Use the previous and next frames only as motion context.
- Label only the pedestrians visible inside red bounding boxes in the TARGET frame.
- Read and copy the integer appearing after "ID:" exactly.
- Never invent a pedestrian ID.
- Return each pedestrian ID in the TARGET frame exactly once.
- Being located inside a road is not sufficient to classify a pedestrian as crossing.
- The pedestrian must be entering or traversing the ego vehicle's relevant path.
- If the action is difficult to determine, still choose one of the two states,
  assign a low confidence score, and set review_required to true.
- Do not use the crossing/not-crossing text from an evaluated model, if any appears.
- Return strict JSON only.
- Do not return Markdown.
- Do not return explanations outside the JSON object.
""".strip()


USER_PROMPT = """
The three images are ordered as follows:

1. PREVIOUS context frame
2. TARGET frame to annotate
3. NEXT context frame

TARGET frame_id: {frame_id}
TARGET time: {time_sec:.3f} seconds

Scene assumption:
{scene_note}

Inspect every red bounding box in the TARGET frame.

Read the integer pedestrian ID displayed after "ID:".

Return JSON using exactly this structure:

{{
  "pedestrians": [
    {{
      "person_id": 12,
      "state": "crossing",
      "confidence": 0.90,
      "review_required": false,
      "reason": "The pedestrian is moving laterally across the ego driving path."
    }},
    {{
      "person_id": 18,
      "state": "not_crossing",
      "confidence": 0.88,
      "review_required": false,
      "reason": "The pedestrian is walking parallel to the road."
    }}
  ]
}}

Requirements:

- state must be either "crossing" or "not_crossing".
- confidence must be between 0.0 and 1.0.
- reason must be brief and based on visible evidence.
- Return every target-frame pedestrian ID exactly once.
- If no pedestrian bounding box is visible in the TARGET frame, return:

{{"pedestrians": []}}
""".strip()


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Qwen3.7-Plus to generate PIE-style current pedestrian "
            "crossing-action annotations from a video containing bbox and IDs."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        default='/output/clip/clip_track_only.mp4',
        help="输入视频路径。",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default='C:/Users/yancheng/Pedestrain_crossing_evalution',
        help="输出目录。生成的文件名为 <视频名>_annotations.csv。",
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10,
        help=(
            "每秒评估多少帧。默认3 FPS；"
            "设置为0表示评估视频的每一个原始帧。"
        ),
    )

    parser.add_argument(
        "--context-gap-sec",
        type=float,
        default=0.33,
        help=(
            "目标帧与前后文帧之间的时间间隔，单位为秒。"
            "默认0.33秒。"
        ),
    )

    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="从视频第几秒开始处理。",
    )

    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="处理到视频第几秒。默认处理到视频结束。",
    )

    parser.add_argument(
        "--max-target-frames",
        type=int,
        default=None,
        help="最多处理多少个目标帧，适合测试API。",
    )

    parser.add_argument(
        "--scene-note",
        type=str,
        default=(
            "The ego vehicle is the dash-camera vehicle. "
            "Unless the road geometry clearly indicates otherwise, "
            "the ego vehicle is expected to continue through the visible "
            "forward drivable corridor."
        ),
        help="提供给Qwen的自车路径说明。",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN_MODEL", "qwen3.7-plus"),
        help="Qwen模型名称。",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY"),
        help="百炼API Key，也可以通过DASHSCOPE_API_KEY设置。",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        help="百炼OpenAI兼容接口地址。",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="单次API请求超时时间，单位为秒。",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="单个目标帧API调用失败后的最大重试次数。",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="发送给Qwen的临时JPEG图片质量。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "强制重新生成，即使annotations.csv已经存在。"
            "默认会跳过已存在的标注文件，避免重复消耗API。"
        ),
    )

    return parser.parse_args()


# ============================================================
# 视频处理
# ============================================================

def inspect_video(video_path: Path) -> VideoInfo:
    """读取视频FPS和总帧数。"""

    if not video_path.exists():
        raise FileNotFoundError(f"找不到输入视频：{video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"视频FPS无效：{fps}")

    if total_frames <= 0:
        raise RuntimeError(f"视频总帧数无效：{total_frames}")

    return VideoInfo(
        fps=fps,
        total_frames=total_frames,
    )


def build_targets(
    video_info: VideoInfo,
    sample_fps: float,
    context_gap_sec: float,
    start_sec: float,
    end_sec: Optional[float],
    max_target_frames: Optional[int],
) -> List[TargetFrame]:
    """建立所有需要提交给Qwen的目标帧。"""

    if sample_fps < 0:
        raise ValueError("--sample-fps不能小于0。")

    if context_gap_sec <= 0:
        raise ValueError("--context-gap-sec必须大于0。")

    if start_sec < 0:
        raise ValueError("--start-sec不能小于0。")

    start_frame_id = max(
        0,
        int(math.floor(start_sec * video_info.fps)),
    )

    if end_sec is None:
        end_frame_id = video_info.total_frames - 1
    else:
        end_frame_id = min(
            video_info.total_frames - 1,
            int(math.floor(end_sec * video_info.fps)),
        )

    if end_frame_id < start_frame_id:
        raise ValueError("--end-sec必须大于--start-sec。")

    # sample_fps=0：处理全部原始帧
    if sample_fps == 0 or sample_fps >= video_info.fps:
        frame_step = 1
    else:
        frame_step = max(
            1,
            int(round(video_info.fps / sample_fps)),
        )

    context_gap_frames = max(
        1,
        int(round(context_gap_sec * video_info.fps)),
    )

    targets: List[TargetFrame] = []

    for frame_id in range(
        start_frame_id,
        end_frame_id + 1,
        frame_step,
    ):
        previous_frame_id = max(
            0,
            frame_id - context_gap_frames,
        )

        next_frame_id = min(
            video_info.total_frames - 1,
            frame_id + context_gap_frames,
        )

        targets.append(
            TargetFrame(
                frame_id=frame_id,
                time_sec=frame_id / video_info.fps,
                previous_frame_id=previous_frame_id,
                next_frame_id=next_frame_id,
            )
        )

    if max_target_frames is not None:
        if max_target_frames < 0:
            raise ValueError("--max-target-frames不能小于0。")

        targets = targets[:max_target_frames]

    return targets


def extract_frames(
    video_path: Path,
    temp_dir: Path,
    required_frame_ids: Sequence[int],
    jpeg_quality: int,
) -> Dict[int, Path]:
    """
    将需要的帧提取到临时目录。

    程序结束后，TemporaryDirectory会自动删除所有临时图片。
    """

    required_ids = sorted(
        set(int(frame_id) for frame_id in required_frame_ids)
    )

    if not required_ids:
        return {}

    output_paths: Dict[int, Path] = {
        frame_id: temp_dir / f"frame_{frame_id:08d}.jpg"
        for frame_id in required_ids
    }

    required_id_set = set(required_ids)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV无法打开视频：{video_path}")

    maximum_required_frame = max(required_ids)

    current_frame_id = 0

    progress = tqdm(
        total=len(required_ids),
        desc="提取临时视频帧",
    )

    while current_frame_id <= maximum_required_frame:
        success, frame = capture.read()

        if not success:
            break

        if current_frame_id in required_id_set:
            saved = cv2.imwrite(
                str(output_paths[current_frame_id]),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    int(jpeg_quality),
                ],
            )

            if not saved:
                progress.close()
                capture.release()

                raise RuntimeError(
                    f"无法保存临时帧：{current_frame_id}"
                )

            progress.update(1)

        current_frame_id += 1

    progress.close()
    capture.release()

    missing_frame_ids = [
        frame_id
        for frame_id, frame_path in output_paths.items()
        if not frame_path.exists()
    ]

    if missing_frame_ids:
        raise RuntimeError(
            "以下视频帧提取失败："
            f"{missing_frame_ids[:10]}"
        )

    return output_paths


# ============================================================
# 图片编码和API消息
# ============================================================

def image_to_data_url(image_path: Path) -> str:
    """将本地JPEG图片转换为Base64 Data URL。"""

    image_bytes = image_path.read_bytes()

    encoded = base64.b64encode(
        image_bytes
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def build_messages(
    target: TargetFrame,
    frame_paths: Dict[int, Path],
    scene_note: str,
) -> List[Dict[str, Any]]:
    """为某个目标帧建立多模态API消息。"""

    user_prompt = USER_PROMPT.format(
        frame_id=target.frame_id,
        time_sec=target.time_sec,
        scene_note=scene_note,
    )

    previous_path = frame_paths[target.previous_frame_id]
    target_path = frame_paths[target.frame_id]
    next_path = frame_paths[target.next_frame_id]

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "PREVIOUS context frame:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(previous_path),
                    },
                },
                {
                    "type": "text",
                    "text": "TARGET frame:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(target_path),
                    },
                },
                {
                    "type": "text",
                    "text": "NEXT context frame:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(next_path),
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]


# ============================================================
# Qwen API调用
# ============================================================

def call_qwen(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    retries: int,
) -> str:
    """调用Qwen API，并在失败时自动重试。"""

    last_error: Optional[Exception] = None
    total_attempts = max(1, retries)

    for attempt in range(total_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_completion_tokens=2000,
                response_format={
                    "type": "json_object",
                },
                extra_body={
                    "enable_thinking": False,
                    "vl_high_resolution_images": True,
                },
            )

            content = response.choices[0].message.content

            if content is None:
                return ""

            return content

        except Exception as error:
            last_error = error

            if attempt + 1 < total_attempts:
                wait_seconds = min(
                    30.0,
                    (2 ** attempt) + random.random(),
                )

                print(
                    f"API调用失败，{wait_seconds:.1f}秒后重试："
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Qwen API调用失败：{last_error}"
    )


# ============================================================
# API结果解析
# ============================================================

def parse_json_response(response_text: str) -> Dict[str, Any]:
    """解析Qwen返回的JSON。"""

    cleaned_text = response_text.strip()

    if not cleaned_text:
        raise ValueError("Qwen返回了空内容。")

    try:
        parsed = json.loads(cleaned_text)

    except json.JSONDecodeError:
        # 兼容模型在JSON前后输出少量其他文本的情况
        match = re.search(
            r"\{.*\}",
            cleaned_text,
            flags=re.DOTALL,
        )

        if match is None:
            raise ValueError(
                f"无法从模型返回内容中找到JSON：{cleaned_text[:500]}"
            )

        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Qwen返回JSON的根节点必须是对象。")

    return parsed


def parse_boolean(value: Any) -> bool:
    """更稳健地解析布尔值。"""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()

    return normalized in {
        "true",
        "1",
        "yes",
        "y",
    }


def normalize_people(
    payload: Dict[str, Any],
    target: TargetFrame,
) -> List[Dict[str, Any]]:
    """将Qwen结果转换为CSV行。"""

    raw_pedestrians = payload.get(
        "pedestrians",
        [],
    )

    if not isinstance(raw_pedestrians, list):
        raise ValueError(
            "Qwen结果中的pedestrians必须是JSON数组。"
        )

    rows: List[Dict[str, Any]] = []
    seen_person_ids: set[int] = set()

    for raw_person in raw_pedestrians:
        if not isinstance(raw_person, dict):
            continue

        # ----------------------------
        # person_id
        # ----------------------------
        try:
            person_id = int(
                str(
                    raw_person.get(
                        "person_id",
                        "",
                    )
                ).strip()
            )

        except (TypeError, ValueError):
            continue

        # 删除同一帧中重复出现的ID
        if person_id in seen_person_ids:
            continue

        seen_person_ids.add(person_id)

        # ----------------------------
        # state
        # ----------------------------
        state = (
            str(
                raw_person.get(
                    "state",
                    "",
                )
            )
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        invalid_state = state not in VALID_STATES

        if invalid_state:
            # 无效类别默认转换为not_crossing，并强制人工检查
            state = "not_crossing"

        # ----------------------------
        # confidence
        # ----------------------------
        try:
            confidence = float(
                raw_person.get(
                    "confidence",
                    0.0,
                )
            )

        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        # ----------------------------
        # review_required
        # ----------------------------
        review_required = parse_boolean(
            raw_person.get(
                "review_required",
                False,
            )
        )

        # 类别无效或置信度低于0.65时强制复核
        if invalid_state or confidence < 0.65:
            review_required = True

        # ----------------------------
        # reason
        # ----------------------------
        reason = str(
            raw_person.get(
                "reason",
                "",
            )
        ).strip()

        reason = reason.replace(
            "\n",
            " ",
        ).replace(
            "\r",
            " ",
        )

        rows.append(
            {
                "frame_id": target.frame_id,
                "time_sec": f"{target.time_sec:.6f}",
                "person_id": person_id,
                "state": state,
                "confidence": f"{confidence:.4f}",
                "review_required": str(
                    review_required
                ).lower(),
                "reason": reason,
            }
        )

    rows.sort(
        key=lambda row: int(row["person_id"])
    )

    return rows


# ============================================================
# CSV输出
# ============================================================

def prepare_output(
    output_dir: Path,
    video_path: Path,
    overwrite: bool,
) -> Path:
    """创建只包含表头的<视频名>_annotations.csv。"""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / f"{video_path.stem}_annotations.csv"

    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"{csv_path}已经存在。"
            "如需覆盖，请添加--overwrite参数。"
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

    return csv_path


def append_rows(
    csv_path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    """将一个目标帧的标注结果追加到CSV。"""

    if not rows:
        return

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
        )

        writer.writerows(rows)


# ============================================================
# 主程序
# ============================================================

def main() -> int:
    args = parse_args()

    existing_csv_path = args.output / f"{args.video.stem}_annotations.csv"

    if existing_csv_path.exists() and not args.overwrite:
        print(
            "标注文件已存在，跳过LLM标注调用（避免重复消耗API）："
            f"{existing_csv_path}\n"
            "如需重新生成，请添加 --overwrite 参数。"
        )
        return 0

    if not args.api_key:
        raise ValueError(
            "没有找到API Key。请设置环境变量DASHSCOPE_API_KEY，"
            "或者使用--api-key参数。"
        )

    if not args.base_url:
        raise ValueError(
            "没有找到API Base URL。请设置环境变量DASHSCOPE_BASE_URL，"
            "或者使用--base-url参数。"
        )

    try:
        from openai import OpenAI

    except ImportError as error:
        raise RuntimeError(
            "缺少依赖。请运行：\n"
            "pip install openai opencv-python tqdm"
        ) from error

    # 读取视频信息
    video_info = inspect_video(
        args.video
    )

    print(
        f"视频FPS：{video_info.fps:.3f}"
    )

    print(
        f"视频总帧数：{video_info.total_frames}"
    )

    print(
        "视频时长："
        f"{video_info.total_frames / video_info.fps:.3f}秒"
    )

    # 建立待处理目标帧
    targets = build_targets(
        video_info=video_info,
        sample_fps=args.sample_fps,
        context_gap_sec=args.context_gap_sec,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_target_frames=args.max_target_frames,
    )

    if not targets:
        raise RuntimeError(
            "没有选择任何待处理目标帧。"
        )

    print(
        f"需要调用Qwen处理的目标帧数量：{len(targets)}"
    )

    # 创建CSV（前面已经判断过是否需要跳过，这里总是允许写入）
    csv_path = prepare_output(
        output_dir=args.output,
        video_path=args.video,
        overwrite=True,
    )

    # 收集所有需要提取的前帧、目标帧和后帧
    required_frame_ids: List[int] = []

    for target in targets:
        required_frame_ids.extend(
            [
                target.previous_frame_id,
                target.frame_id,
                target.next_frame_id,
            ]
        )

    # 初始化OpenAI兼容客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    failed_frame_count = 0
    successful_frame_count = 0
    total_annotation_count = 0

    # 所有帧只保存到临时目录
    # 程序结束后自动删除
    with tempfile.TemporaryDirectory(
        prefix="qwen_pie_frames_"
    ) as temporary_directory:
        temporary_path = Path(
            temporary_directory
        )

        frame_paths = extract_frames(
            video_path=args.video,
            temp_dir=temporary_path,
            required_frame_ids=required_frame_ids,
            jpeg_quality=args.jpeg_quality,
        )

        progress = tqdm(
            targets,
            desc="Qwen3.7逐帧标注",
        )

        for target in progress:
            try:
                messages = build_messages(
                    target=target,
                    frame_paths=frame_paths,
                    scene_note=args.scene_note,
                )

                raw_response = call_qwen(
                    client=client,
                    model=args.model,
                    messages=messages,
                    retries=args.retries,
                )

                parsed_payload = parse_json_response(
                    raw_response
                )

                rows = normalize_people(
                    payload=parsed_payload,
                    target=target,
                )

                append_rows(
                    csv_path=csv_path,
                    rows=rows,
                )

                successful_frame_count += 1
                total_annotation_count += len(rows)

                progress.set_postfix(
                    {
                        "frame": target.frame_id,
                        "persons": len(rows),
                        "failed": failed_frame_count,
                    }
                )

            except Exception as error:
                failed_frame_count += 1

                print(
                    f"\n[警告] 第{target.frame_id}帧处理失败："
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )

    print()
    print("处理完成。")
    print(f"成功处理目标帧：{successful_frame_count}")
    print(f"失败目标帧：{failed_frame_count}")
    print(f"行人标注总数：{total_annotation_count}")
    print(f"输出文件：{csv_path}")

    if failed_frame_count > 0:
        print(
            "注意：处理失败的目标帧不会在annotations.csv中产生记录。"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\n程序被用户中断。",
            file=sys.stderr,
        )

        raise SystemExit(130)

    except Exception as error:
        print(
            f"\n程序运行失败：{type(error).__name__}: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1)
