import os
import sys
import numpy as np


class IntentionPredictorONNX:
    """
    ONNX Runtime wrapper for the PedAST-GCN crossing-intention model.
    Drop-in replacement for IntentionPredictor — same predict() interface.

    Advantages over the PyTorch version:
      - No st_gcn / graph import gymnastics needed
      - Runs on CPU or GPU via onnxruntime / onnxruntime-gpu
      - Easier deployment (Jetson, edge devices, no PyTorch required)

    Required:
        pip install onnxruntime-gpu   # GPU
        pip install onnxruntime       # CPU only
    """

    def __init__(self, onnx_path: str, device='0'):
        """
        Args:
            onnx_path (str): path to best.onnx
            device: GPU index string ('0', '1', ...) or 'cpu'
        """
        use_gpu = (device != 'cpu')

        # Import PedestrianIntentPredictor from PedAST-GCN/inference_onnx.py
        # (inference_onnx.py has no conflicting imports, no sys.modules trick needed)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pedast_dir   = os.path.join(project_root, 'PedAST-GCN')
        sys.path.insert(0, pedast_dir)
        try:
            from inference_onnx import PedestrianIntentPredictor
        finally:
            if sys.path and sys.path[0] == pedast_dir:
                sys.path.pop(0)

        self.predictor = PedestrianIntentPredictor(onnx_path, use_gpu=use_gpu)

    def predict(self, skeleton_t, box_t):
        """
        Run crossing-intention inference (supports batch).
        Same interface as IntentionPredictor.predict().

        Args:
            skeleton_t: torch.Tensor or np.ndarray, shape (B, 16, 18, 3)
            box_t:      torch.Tensor or np.ndarray, shape (B, 16,  4, 2)

        Returns:
            labels : list[str]   — 'crossing' or 'not crossing', length B
            probs  : list[float] — sigmoid probability in [0, 1],  length B
        """
        # 支持 torch.Tensor 输入，自动转换为 numpy
        if hasattr(skeleton_t, 'cpu'):
            skeleton_t = skeleton_t.cpu().numpy()
        if hasattr(box_t, 'cpu'):
            box_t = box_t.cpu().numpy()

        probs_np = self.predictor.predict(
            skeleton_t.astype(np.float32),
            box_t.astype(np.float32),
        )  # (B, 1)

        probs  = probs_np[:, 0].tolist()
        labels = ['crossing' if p > 0.5 else 'not crossing' for p in probs]
        return labels, probs
