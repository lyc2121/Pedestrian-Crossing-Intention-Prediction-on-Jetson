"""
Pedestrian Crossing Intention Prediction - ONNX Inference Script
=================================================================
Installation (works on both Jetson and PC):
    pip install onnxruntime          # CPU inference
    pip install onnxruntime-gpu      # GPU inference (Jetson requires a dedicated wheel, see below)

Jetson installation:
    https://elinux.org/Jetson_Zoo#ONNX_Runtime
    Download the onnxruntime wheel that matches your JetPack version.

Input format:
    skeleton : numpy float32, shape (N, 16, 18, 3)
               N=batch, 16=time frames, 18=skeleton joints, 3=(x, y, conf)
    box      : numpy float32, shape (N, 16,  4, 2)
               N=batch, 16=time frames,  4=bounding box nodes, 2=(x, y)

Output:
    prob     : numpy float32, shape (N, 1)  range 0~1, >=0.5 means predicted to cross
"""

import numpy as np
import onnxruntime as ort


class PedestrianIntentPredictor:
    def __init__(self, onnx_path: str, use_gpu: bool = False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu \
                    else ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=session_options, providers=providers)
        print(f"[ONNX] Loaded: {onnx_path}")
        print(f"[ONNX] Provider in use: {self.sess.get_providers()[0]}")

    def predict(self, skeleton: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        skeleton : float32 array, shape (N, 16, 18, 3)
        box      : float32 array, shape (N, 16,  4, 2)

        Returns
        -------
        prob : float32 array, shape (N, 1)  crossing probability
        """
        skeleton = skeleton.astype(np.float32)
        box      = box.astype(np.float32)

        result = self.sess.run(
            output_names=['prob'],
            input_feed={'skeleton': skeleton, 'box': box}
        )
        return result[0]   # shape (N, 1)

    def predict_label(self, skeleton: np.ndarray, box: np.ndarray,
                      threshold: float = 0.5) -> np.ndarray:
        """Returns 0/1 label"""
        prob = self.predict(skeleton, box)
        return (prob >= threshold).astype(np.int32)


# -----------------------------------------------------------------------
# Usage example
# -----------------------------------------------------------------------
if __name__ == '__main__':
    # Replace with your .onnx file path
    ONNX_PATH = 'C:/Users/yancheng/yolov5-deepsort-pedestraintracking-pedestrain-crossing-intention-skeleton-boundingbox/PedAST-GCN/best.onnx'

    predictor = PedestrianIntentPredictor(ONNX_PATH, use_gpu=False)

    # Random test data (replace with real data in practice)
    skeleton_demo = np.random.rand(1, 16, 18, 3).astype(np.float32)
    box_demo      = np.random.rand(1, 16,  4, 2).astype(np.float32)

    prob  = predictor.predict(skeleton_demo, box_demo)
    label = predictor.predict_label(skeleton_demo, box_demo)

    print(f"Crossing probability: {prob[0, 0]:.4f}")
    print(f"Predicted label: {'crossing' if label[0, 0] == 1 else 'not crossing'}")
