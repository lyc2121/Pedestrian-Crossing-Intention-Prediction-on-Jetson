import sys
import os
import numpy as np
import torch


def _add_rtmlib_to_path():
    """
    如果当前 Python 环境找不到 rtmlib，则自动在常见 conda 环境目录中搜索
    并将含有 rtmlib 的 site-packages 加入 sys.path。
    """
    try:
        import rtmlib  # noqa: F401
        return  # 已经可以找到，无需操作
    except ImportError:
        pass

    search_roots = [
        os.path.expanduser(r"~\.conda\envs"),
        os.path.expanduser(r"~\anaconda3\envs"),
        os.path.expanduser(r"~\miniconda3\envs"),
        r"C:\ProgramData\Anaconda3\envs",
        r"C:\ProgramData\miniconda3\envs",
    ]

    for envs_root in search_roots:
        if not os.path.isdir(envs_root):
            continue
        for env_name in os.listdir(envs_root):
            site_pkg = os.path.join(envs_root, env_name, "lib", "site-packages")
            if os.path.isdir(os.path.join(site_pkg, "rtmlib")):
                if site_pkg not in sys.path:
                    sys.path.insert(0, site_pkg)
                    print(f"[PoseExtractor] rtmlib found in conda env '{env_name}', added to sys.path")
                return  # 找到第一个即可

    raise ImportError(
        "rtmlib 未找到。请在运行脚本的 Python 环境中安装：\n"
        "  conda activate yolov5-deepsort-pedestraintracking-master\n"
        "  pip install rtmlib onnxruntime-gpu"
    )


_add_rtmlib_to_path()


class PoseExtractor:
    """
    RTMPose-s 单人姿态估计封装（基于 rtmlib + ONNX Runtime）。

    输出格式: np.ndarray (18, 3) → [x_rel, y_rel, confidence]
      - COCO-17 关节 + 虚拟第 17 号节点（髋关节中心，关节 11 和 12 均值）
      - 坐标已相对原始检测框 (x1,y1,x2,y2) 归一化

    依赖:
        pip install rtmlib onnxruntime-gpu
    """

    def __init__(self, checkpoint, device):
        """
        Args:
            checkpoint (str): RTMPose-s ONNX 模型文件路径
                              例: 'rtmpose/weights/rtmpose-s_256x192.onnx'
            device    : GPU 索引字符串 ('0','1',...) 或 'cpu'，或 torch.device
        """
        from rtmlib import RTMPose

        # rtmlib 使用 'cuda' / 'cpu' 字符串
        if isinstance(device, torch.device):
            device_str = 'cpu' if device.type == 'cpu' else 'cuda'
        elif isinstance(device, str):
            device_str = 'cpu' if device == 'cpu' else 'cuda'
        else:
            device_str = 'cuda'

        # RTMPose-s 标准输入尺寸: width=192, height=256
        self.rtmpose = RTMPose(
            onnx_model=checkpoint,
            model_input_size=(192, 256),
            device=device_str,
            backend='onnxruntime',
        )

        session = getattr(self.rtmpose, 'session', None)
        if session is not None and hasattr(session, 'get_providers'):
            providers = session.get_providers()
            print(f"[RTMPose] Provider in use: {providers[0]}")
            if device_str == 'cuda' and providers[0] != 'CUDAExecutionProvider':
                raise RuntimeError(
                    'RTMPose requested CUDA, but ONNX Runtime fell back to '
                    f'{providers[0]}. Check the CUDA/cuDNN installation.')

    def extract(self, frame, bbox):
        """单人提取（内部复用 extract_batch）。"""
        skeletons, _ = self.extract_batch(frame, [bbox])
        return skeletons[0]

    def extract_batch(self, frame, bboxes, track_ids=None):
        """
        批量提取多人骨架，只调用一次 RTMPose inference。

        Args:
            frame  (np.ndarray): 完整 BGR 图像 (H, W, 3)
            bboxes (array-like): (N, 4) 或 list of [x1,y1,x2,y2]

        Returns:
            np.ndarray (N, 18, 3): 每人 [x_rel, y_rel, confidence]，坐标按各自 bbox 归一化
        """
        bboxes_arr = np.array([[float(v) for v in b] for b in bboxes], dtype=np.float32)
        N = len(bboxes_arr)
        empty = np.zeros((N, 18, 3), dtype=np.float32)
        invalid = np.zeros(N, dtype=bool)
        if N == 0:
            return empty, invalid

        try:
            keypoints, scores = self.rtmpose(frame, bboxes=bboxes_arr)
        except Exception:
            return empty, invalid

        if keypoints is None or len(keypoints) == 0:
            return empty, invalid

        results = np.zeros((N, 18, 3), dtype=np.float32)
        pose_valid = np.zeros(N, dtype=bool)
        result_count = min(N, len(keypoints), len(scores))
        for n in range(result_count):
            x1, y1, x2, y2 = bboxes_arr[n]
            bbox_w = max(x2 - x1, 1.0)
            bbox_h = max(y2 - y1, 1.0)
            kpts = keypoints[n]   # (17, 2)
            conf = scores[n]      # (17,)

            skel17 = np.empty((17, 3), dtype=np.float32)
            for j in range(17):
                skel17[j, 0] = (float(kpts[j, 0]) - x1) / bbox_w
                skel17[j, 1] = (float(kpts[j, 1]) - y1) / bbox_h
                skel17[j, 2] = float(conf[j])

            hip_center = (skel17[5] + skel17[6]) / 2.0
            results[n] = np.vstack([skel17, hip_center])
            pose_valid[n] = np.isfinite(results[n]).all() and np.any(results[n, :, 2] > 0)

        return results, pose_valid
