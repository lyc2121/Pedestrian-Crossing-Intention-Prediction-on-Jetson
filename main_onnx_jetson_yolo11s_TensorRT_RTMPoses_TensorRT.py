"""YOLO11s TensorRT + RTMPose-s TensorRT pipeline on Jetson.

This independent entry point runs both YOLO11s and RTMPose-s with TensorRT.
DeepSORT, temporal buffer, PedAST-GCN, Flask stream and output handling are
reused from the existing Jetson pipeline.
"""

import torch

import main_onnx_jetson_yolo11s as jetson_app
from main_onnx_jetson_yolo11s_TensorRT import load_tensorrt_yolo
from self_utils.pose_extractor_tensorrt import PoseExtractorTensorRT


_base_video_name = jetson_app.get_video_name


def _output_name(input_path):
    return '{}_yolo11s_RTMPose_TensorRT'.format(_base_video_name(input_path))


class _ConfiguredPoseExtractor(PoseExtractorTensorRT):
    """Inject the command-line engine path into the shared pipeline."""

    engine_path = 'rtmpose/weights/rtmpose-s_256x192-fp32.engine'

    def __init__(self, checkpoint=None, device='0'):
        super().__init__(checkpoint=self.engine_path, device=device)


def main():
    torch.multiprocessing.set_start_method('spawn')
    parser = jetson_app.build_parser()
    parser.set_defaults(
        weights='weights/yolo11s.engine', device='0', onnx_device='0')
    parser.add_argument(
        '--pose_engine',
        default='rtmpose/weights/rtmpose-s_256x192-fp32.engine',
        help='RTMPose-s TensorRT .engine (FP32 recommended for stable keypoints)')
    config = parser.parse_args()
    print(config)

    _ConfiguredPoseExtractor.engine_path = config.pose_engine
    jetson_app.load_yolo_model = load_tensorrt_yolo
    jetson_app.PoseExtractor = _ConfiguredPoseExtractor
    jetson_app.get_video_name = _output_name
    jetson_app.output_name_suffix = ''
    jetson_app.main(config)
    print('结果保存在：', config.output)


if __name__ == '__main__':
    main()
