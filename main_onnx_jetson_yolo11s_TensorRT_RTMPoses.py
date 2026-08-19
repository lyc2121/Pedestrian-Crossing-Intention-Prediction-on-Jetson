"""YOLO11s PyTorch CUDA + RTMPose-s TensorRT pipeline for Jetson.

YOLO11s retains the original .pt implementation. RTMPose-s uses a TensorRT
engine. DeepSORT, temporal buffering, PedAST-GCN, Flask streaming, video/CSV
output and latency profiling reuse the existing Jetson pipeline.
"""

import torch

import main_onnx_jetson_yolo11s as jetson_app
from self_utils.pose_extractor_tensorrt import PoseExtractorTensorRT


_base_video_name = jetson_app.get_video_name


def _output_name(input_path):
    """Use <original_name>_yolo11s_TensorRT_RTMPoses for all outputs."""
    return '{}_yolo11s_TensorRT_RTMPoses'.format(
        _base_video_name(input_path))


class _ConfiguredPoseExtractor(PoseExtractorTensorRT):
    """Inject the selected RTMPose TensorRT engine into the shared pipeline."""

    engine_path = 'rtmpose/weights/rtmpose-s_256x192-fp32.engine'

    def __init__(self, checkpoint=None, device='0'):
        super().__init__(checkpoint=self.engine_path, device=device)


def main():
    torch.multiprocessing.set_start_method('spawn')
    parser = jetson_app.build_parser()
    parser.set_defaults(
        weights='weights/yolo11s.pt',
        device='0',
        onnx_device='0',
    )
    parser.add_argument(
        '--pose_engine',
        default='rtmpose/weights/rtmpose-s_256x192-fp32.engine',
        help='RTMPose-s TensorRT .engine (FP32 recommended for stable keypoints)')
    config = parser.parse_args()
    print(config)

    # Keep the shared PyTorch YOLO loader and replace only RTMPose with TRT.
    _ConfiguredPoseExtractor.engine_path = config.pose_engine
    jetson_app.PoseExtractor = _ConfiguredPoseExtractor
    jetson_app.get_video_name = _output_name
    jetson_app.output_name_suffix = ''
    jetson_app.main(config)
    print('结果保存在：', config.output)


if __name__ == '__main__':
    main()
