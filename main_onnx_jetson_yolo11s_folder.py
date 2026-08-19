"""Folder batch version of main_onnx_jetson_yolo11s.py."""

import os
import sys

from self_utils.video_folder_batch_runner import run_folder


if __name__ == '__main__':
    script = os.path.join(os.path.dirname(__file__), 'main_onnx_jetson_yolo11s.py')
    sys.exit(run_folder(script, 'Batch YOLO11s videos from a folder'))

