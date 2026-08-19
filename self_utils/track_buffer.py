from collections import deque
import numpy as np
import torch


class TrackBuffer:
    """
    Maintains a sliding window of T=16 frames per tracked person.
    Stores bbox and skeleton keypoints for each frame.
    Missing frames are filled by linear interpolation (if surrounded by real frames)
    or forward-fill (if only preceding frames exist).
    """

    def __init__(self, window_size=16, min_frames=3):
        self.window_size = window_size
        self.min_frames  = min_frames
        self._buffers = {}  # track_id (int) -> deque of frame-data dicts

    def update(self, track_id, bbox, skeleton):
        """
        Append one real detection frame for a track.

        Args:
            track_id (int): unique DeepSORT track identifier
            bbox (list): [x1, y1, x2, y2] pixel coordinates
            skeleton (np.ndarray): (18, 3) [x_rel, y_rel, conf] normalised by bbox size

        Returns:
            bool: True when the buffer has accumulated >= window_size frames
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)

        self._buffers[track_id].append({
            'type': 'real',
            'bbox': list(bbox),
            'skeleton': skeleton.copy(),
        })

        return len(self._buffers[track_id]) >= self.min_frames

    def mark_miss(self, track_id):
        """
        为仍在被跟踪、但本帧姿态估计失败/退化（如 bbox 太小）的 track
        追加一个 miss 占位符。

        与 pad_miss 的区别：pad_miss 用于本帧完全未检测到的 track，
        这里用于本帧检测到了人、但骨架数据不可靠、不应作为 'real' 帧
        存入滑窗参与意图预测的情况。
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)

        self._buffers[track_id].append({'type': 'miss'})

    def pad_miss(self, active_ids):
        """
        For every track in the buffer NOT currently detected,
        append a 'miss' placeholder to keep the timeline aligned.
        Called once per frame after processing all detected tracks.
        """
        for tid in list(self._buffers.keys()):
            if tid not in active_ids:
                self._buffers[tid].append({'type': 'miss'})

    def get_sample(self, track_id, img_w, img_h):
        """
        Build model-ready tensors from the last window_size frames.
        Missing frames are filled before building tensors.

        Returns:
            skeleton_t : Tensor (1, 16, 18, 3), or None if no real data exists
            box_t      : Tensor (1, 16,  4, 2), or None
        """
        frames = list(self._buffers[track_id])[-self.window_size:]

        # 不足 window_size 帧时，在开头补 miss 占位符，由 _fill_gaps
        # 用第一帧向后回填（backward-fill）。
        # 补在开头是为了让真实的最近一帧始终留在窗口末尾——
        # 如果补在末尾，会用最后一帧反复向后 forward-fill，
        # 导致真实的最新一帧被埋在窗口中间、窗口后半段变成人为的
        # "静止不动"伪造序列，容易被模型误判为行人在路口停下准备过街。
        while len(frames) < self.window_size:
            frames.insert(0, {'type': 'miss'})

        frames = self._fill_gaps(frames)
        if frames is None:
            return None, None

        skeletons = []
        boxes = []

        for frame in frames:
            skeletons.append(frame['skeleton'].astype(np.float32))

            x1, y1, x2, y2 = frame['bbox']
            corners = np.array([
                [x1 / img_w, y1 / img_h],   # top-left
                [x2 / img_w, y1 / img_h],   # top-right
                [x2 / img_w, y2 / img_h],   # bottom-right
                [x1 / img_w, y2 / img_h],   # bottom-left
            ], dtype=np.float32)
            boxes.append(corners)

        skeleton_arr = np.stack(skeletons, axis=0)   # (16, 18, 3)
        box_arr      = np.stack(boxes,     axis=0)   # (16,  4, 2)

        skeleton_t = torch.from_numpy(skeleton_arr).float().unsqueeze(0)  # (1,16,18,3)
        box_t      = torch.from_numpy(box_arr     ).float().unsqueeze(0)  # (1,16, 4,2)

        return skeleton_t, box_t

    def _fill_gaps(self, frames):
        """
        Fill 'miss' entries in a frame list:
          - Both neighbours real  →  linear interpolation
          - Only preceding real   →  forward-fill (copy that frame)
          - Only following real   →  backward-fill (copy that frame, for leading miss)
          - All miss              →  return None (no real data to fill from)
        """
        if all(f['type'] == 'miss' for f in frames):
            return None

        n = len(frames)
        result = list(frames)

        for i, f in enumerate(frames):
            if f['type'] != 'miss':
                continue

            # nearest real frame before i
            prev_idx = next(
                (j for j in range(i - 1, -1, -1) if frames[j]['type'] == 'real'),
                None
            )
            # nearest real frame after i
            next_idx = next(
                (j for j in range(i + 1, n) if frames[j]['type'] == 'real'),
                None
            )

            if prev_idx is not None and next_idx is not None:
                # linear interpolation between prev and next
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                pf, nf = frames[prev_idx], frames[next_idx]
                result[i] = {
                    'type': 'interp',
                    'bbox': [
                        pf['bbox'][k] * (1 - alpha) + nf['bbox'][k] * alpha
                        for k in range(4)
                    ],
                    'skeleton': pf['skeleton'] * (1 - alpha) + nf['skeleton'] * alpha,
                }
            elif prev_idx is not None:
                # only preceding real frame: forward-fill
                result[i] = frames[prev_idx]
            else:
                # only following real frame: backward-fill (leading miss)
                result[i] = frames[next_idx]

        return result

    # min_w_ratio=0.030, min_h_ratio=0.11
    # min_w_ratio=0.020, min_h_ratio=0.090
    # min_w_ratio=0.013, min_h_ratio=0.074
    # min_w_ratio=0.007, min_h_ratio=0.035

    def prune_all_small(self, img_w, img_h, min_w_ratio=0.03, min_h_ratio=0.11):
        """
        删除 buffer：只有当窗口已经攒满（len(buf) >= window_size）之后，
        如果整个窗口里所有 real 帧的最大宽度仍 < min_w_ratio*img_w 或
        最大高度仍 < min_h_ratio*img_h，才认为这个人在整个观察期间
        始终太远，丢弃 buffer。用相对图像宽高的比例而非绝对像素值，
        避免不同分辨率下同样大小的行人因像素数不同而被区别对待。

        必须先等窗口攒满再评估——如果对还没攒够帧数、buffer 很短的
        新目标也做这个判断，一个正在走近、还没长到阈值的人会在每帧
        被误判为"太小"而清空 buffer，导致 buffer 永远长不过 1 帧，
        is_full 永远无法触发（这曾经是意图预测迟迟不触发的真正原因）。
        """
        min_w = min_w_ratio * img_w
        min_h = min_h_ratio * img_h
        to_remove = []
        for tid, buf in self._buffers.items():
            if len(buf) < self.window_size:
                continue
            real_frames = [f for f in buf if f['type'] == 'real']
            if not real_frames:
                continue
            max_w = max(f['bbox'][2] - f['bbox'][0] for f in real_frames)
            max_h = max(f['bbox'][3] - f['bbox'][1] for f in real_frames)
            if max_w < min_w or max_h < min_h:
                to_remove.append(tid)
        for tid in to_remove:
            del self._buffers[tid]

    def cleanup(self, active_ids):
        """
        Remove a track's buffer only when more than half (>8) of the window
        consists of misses, meaning the person has been gone long enough.
        """
        to_remove = [
            tid for tid, buf in self._buffers.items()
            if tid not in active_ids
            and sum(1 for f in buf if f['type'] == 'miss') > self.window_size // 2
               # and sum(1 for f in buf if f['type'] == 'miss') > 15
        ]
        for tid in to_remove:
            del self._buffers[tid]