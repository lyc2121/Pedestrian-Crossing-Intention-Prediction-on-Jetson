"""Independent YOLO-Pose + DeepSORT + PedAST-GCN processing pipeline."""

import time
import traceback

import cv2
import numpy as np
import torch

from .globals_val import Global
from .post_processing import draw_skeleton
from utils.utils import plot_one_box


def _sync_cuda(device):
    if str(device) != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize(int(device))


def _record(profile, key, started):
    if profile is not None:
        profile[key].append((time.perf_counter() - started) * 1000.0)


def _iou_matrix(a, b):
    """Pairwise IoU for xyxy arrays a=(M,4), b=(N,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(br - tl, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def _associate_tracks_to_poses(track_boxes, detection_boxes, threshold=0.25):
    """Greedy one-to-one IoU association: track index -> detection index."""
    ious = _iou_matrix(track_boxes, detection_boxes)
    pairs = []
    for track_idx in range(ious.shape[0]):
        for detection_idx in range(ious.shape[1]):
            pairs.append((float(ious[track_idx, detection_idx]), track_idx, detection_idx))
    matches = {}
    used_tracks, used_detections = set(), set()
    for iou, track_idx, detection_idx in sorted(pairs, reverse=True):
        if iou < threshold:
            break
        if track_idx in used_tracks or detection_idx in used_detections:
            continue
        matches[track_idx] = detection_idx
        used_tracks.add(track_idx)
        used_detections.add(detection_idx)
    return matches


def _skeleton18(keypoints_xy, keypoints_conf, bbox):
    """Convert absolute COCO-17 keypoints to bbox-relative PedAST 18x3."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    skeleton = np.zeros((18, 3), dtype=np.float32)
    skeleton[:17, 0] = (keypoints_xy[:17, 0] - x1) / width
    skeleton[:17, 1] = (keypoints_xy[:17, 1] - y1) / height
    skeleton[:17, 2] = keypoints_conf[:17]
    # Keep the same virtual node used by the existing RTMPose pipeline.
    skeleton[17] = (skeleton[5] + skeleton[6]) / 2.0
    valid = np.isfinite(skeleton).all() and np.any(skeleton[:17, 2] > 0)
    return skeleton, bool(valid)


def process_yolo_pose_frame(
        image, config, model, class_names, tracker, object_counter,
        track_buffer, intention_predictor, predictions, fps=25.0,
        frame_records=None, return_track_only=True, latency_profile=None):
    """Process one frame without using RTMPose."""
    try:
        if latency_profile is not None:
            _sync_cuda(config.device)
        started = time.perf_counter()
        results = model.predict(
            source=image,
            imgsz=config.img_size,
            conf=config.conf_thres,
            iou=config.iou_thres,
            classes=[config.classes],
            device=config.device,
            verbose=False,
        )
        if latency_profile is not None:
            _sync_cuda(config.device)
            _record(latency_profile, 'YOLO11s-pose TensorRT', started)

        result = results[0]
        if result.keypoints is None:
            raise RuntimeError(
                'The loaded TensorRT model has no pose output. Use yolo11s-pose.engine, '
                'not a detection-only engine.')

        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            det_boxes = np.empty((0, 4), dtype=np.float32)
            det_conf = np.empty((0,), dtype=np.float32)
            det_cls = np.empty((0,), dtype=np.int64)
            keypoints_xy = np.empty((0, 17, 2), dtype=np.float32)
            keypoints_conf = np.empty((0, 17), dtype=np.float32)
        else:
            det_boxes = boxes_obj.xyxy.detach().cpu().numpy().astype(np.float32)
            det_conf = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
            det_cls = boxes_obj.cls.detach().cpu().numpy().astype(np.int64)
            keypoints_xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
            if result.keypoints.conf is None:
                keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
            else:
                keypoints_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)

        # Relative small-box filter. Ratios are configurable from the entry
        # point so thresholds scale consistently with the source resolution.
        img_h, img_w = image.shape[:2]
        if len(det_boxes):
            min_box_w_ratio = float(config.min_box_w_ratio)
            min_box_h_ratio = float(config.min_box_h_ratio)
            keep = ((det_boxes[:, 2] - det_boxes[:, 0]) >= min_box_w_ratio * img_w) & \
                   ((det_boxes[:, 3] - det_boxes[:, 1]) >= min_box_h_ratio * img_h)
            det_boxes, det_conf, det_cls = det_boxes[keep], det_conf[keep], det_cls[keep]
            keypoints_xy, keypoints_conf = keypoints_xy[keep], keypoints_conf[keep]

        if len(det_boxes):
            xywh = np.column_stack((
                (det_boxes[:, 0] + det_boxes[:, 2]) / 2,
                (det_boxes[:, 1] + det_boxes[:, 3]) / 2,
                det_boxes[:, 2] - det_boxes[:, 0],
                det_boxes[:, 3] - det_boxes[:, 1],
            )).astype(np.float32)
            confidences = det_conf.reshape(-1, 1)
            labels = det_cls.tolist()
        else:
            xywh = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0, 1), dtype=np.float32)
            labels = []

        if latency_profile is not None:
            _sync_cuda(config.device)
        started = time.perf_counter()
        outputs = tracker.update(
            torch.from_numpy(xywh), torch.from_numpy(confidences), labels, image)
        if latency_profile is not None:
            _sync_cuda(config.device)
            _record(latency_profile, 'DeepSORT', started)

        full_image = image
        track_only_image = image.copy() if return_track_only else None
        active_ids = set()
        present_num = 0

        if len(outputs) > 0:
            track_boxes = outputs[:, :4].astype(np.float32)
            identities = outputs[:, 5].astype(np.int64)
            # getattr keeps this processing module compatible with entry points
            # created before --pose_match_iou was added.
            pose_match_iou = float(getattr(config, 'pose_match_iou', 0.25))
            matches = _associate_tracks_to_poses(
                track_boxes, det_boxes, threshold=pose_match_iou)
            present_num = len(identities)
            active_ids = set(int(v) for v in identities)
            Global.total_person = Global.total_person | active_ids

            if return_track_only:
                for box, track_id in zip(track_boxes, identities):
                    plot_one_box(box, track_only_image,
                                 text_info='{},ID:{}'.format(class_names[0], int(track_id)),
                                 color=(0, 0, 255))

            infer_ids, infer_skeletons, infer_boxes = [], [], []
            temporal_started = time.perf_counter()
            frame_skeletons = {}
            for track_idx, (box, track_id) in enumerate(zip(track_boxes, identities)):
                track_id = int(track_id)
                detection_idx = matches.get(track_idx)
                if detection_idx is None:
                    track_buffer.mark_miss(track_id)
                    continue
                skeleton, valid = _skeleton18(
                    keypoints_xy[detection_idx], keypoints_conf[detection_idx], box)
                frame_skeletons[track_id] = (skeleton, valid)
                if valid:
                    ready = track_buffer.update(track_id, box.tolist(), skeleton)
                else:
                    track_buffer.mark_miss(track_id)
                    ready = False
                if ready:
                    skeleton_t, box_t = track_buffer.get_sample(track_id, img_w, img_h)
                    if skeleton_t is not None:
                        infer_ids.append(track_id)
                        infer_skeletons.append(skeleton_t)
                        infer_boxes.append(box_t)
            if latency_profile is not None:
                latency_profile['_temporal_ms'] += \
                    (time.perf_counter() - temporal_started) * 1000.0

            if infer_ids:
                started = time.perf_counter()
                labels_batch, probabilities = intention_predictor.predict(
                    torch.cat(infer_skeletons, dim=0), torch.cat(infer_boxes, dim=0))
                _record(latency_profile, 'PedAST-GCN', started)
                for track_id, label, probability in zip(
                        infer_ids, labels_batch, probabilities):
                    predictions[track_id] = (label, probability)

            for box, track_id in zip(track_boxes, identities):
                track_id = int(track_id)
                if track_id not in predictions:
                    continue
                label, probability = predictions[track_id]
                if frame_records is not None:
                    frame_records.append((track_id, label, probability))
                skeleton_info = frame_skeletons.get(track_id)
                if skeleton_info is not None and skeleton_info[1]:
                    draw_skeleton(full_image, skeleton_info[0], box)
                color = (0, 0, 255) if label == 'crossing' else (0, 255, 0)
                plot_one_box(
                    box, full_image,
                    text_info='{},ID:{} | {}({:.2f})'.format(
                        class_names[0], track_id, label, probability),
                    color=color)

        temporal_started = time.perf_counter()
        track_buffer.pad_miss(active_ids)
        track_buffer.prune_all_small(
            img_w, img_h,
            min_w_ratio=float(config.min_box_w_ratio),
            min_h_ratio=float(config.min_box_h_ratio))
        track_buffer.cleanup(active_ids)
        if latency_profile is not None:
            latency_profile['_temporal_ms'] += \
                (time.perf_counter() - temporal_started) * 1000.0
            latency_profile['Temporal buffer'].append(latency_profile['_temporal_ms'])
            latency_profile['_temporal_ms'] = 0.0

        text = 'present person' if False else 'total person'
        total_num = len(Global.total_person)
        full_image = object_counter.draw_counter(
            full_image, present_num, total_num, text, False)
        if return_track_only:
            track_only_image = object_counter.draw_counter(
                track_only_image, present_num, total_num, text, False)
            return full_image, track_only_image
        return full_image
    except Exception as error:
        traceback.print_exc()
        return error
