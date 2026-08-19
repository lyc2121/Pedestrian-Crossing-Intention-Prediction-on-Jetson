"""Jetson pipeline using a TensorRT YOLO11s engine.

The tracking, pose, temporal-buffer, PedAST-GCN, Flask streaming and output
logic are shared with main_onnx_jetson.py. Only the YOLO model loader and the
default weights file differ.

Export the engine on the target Jetson before running this entry point:
    yolo export model=weights/yolo11s.pt format=engine imgsz=640 half=True device=0

Run:
    python main_onnx_jetson_yolo_TensorRT.py --input ./video.mp4
"""

import os
import torch
from ultralytics import YOLO

import main_onnx_jetson_yolo11s as jetson_app


_base_get_video_name = jetson_app.get_video_name


def get_tensorrt_output_name(input_path):
    return '{}_yolo_TensorRT'.format(_base_get_video_name(input_path))


def load_tensorrt_yolo(config):
    if config.device == 'cpu':
        raise ValueError('TensorRT requires a CUDA device; use --device 0')
    if not config.weights.lower().endswith('.engine'):
        raise ValueError(
            'TensorRT entry point requires a .engine model, got: {}'.format(
                config.weights))
    if not os.path.isfile(config.weights):
        raise FileNotFoundError(
            'TensorRT engine not found: {}\n'
            'Export it on this Jetson with:\n'
            '  yolo export model=weights/yolo11s.pt format=engine '
            'imgsz={} half=True device={}'.format(
                config.weights, config.img_size, config.device))

    # Ultralytics selects its TensorRT backend from the .engine suffix. Do not
    # call model.to(): a serialized TensorRT engine is already device-bound.
    print('=> YOLO backend: TensorRT ({})'.format(config.weights))
    return YOLO(config.weights, task='detect')


def main():
    torch.multiprocessing.set_start_method('spawn')
    parser = jetson_app.build_parser()
    parser.set_defaults(weights='weights/yolo11s.engine', device='0')
    config = parser.parse_args()
    print(config)

    jetson_app.load_yolo_model = load_tensorrt_yolo
    jetson_app.get_video_name = get_tensorrt_output_name
    jetson_app.output_name_suffix = ''
    jetson_app.main(config)
    print('结果保存在：', config.output)


if __name__ == '__main__':
    main()
