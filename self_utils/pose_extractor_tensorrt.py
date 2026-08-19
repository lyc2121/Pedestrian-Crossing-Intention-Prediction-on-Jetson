"""RTMPose-s TensorRT pose extractor for Jetson (TensorRT 10 API)."""

import os

import numpy as np
import torch


class _TensorRTRTMPose:
    """Reuse rtmlib preprocessing/postprocessing and replace only inference."""

    def __init__(self, engine_path, device_id=0):
        import tensorrt as trt
        from rtmlib import RTMPose

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as engine_file:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(
                engine_file.read())
        if self.engine is None:
            raise RuntimeError('Failed to deserialize RTMPose engine: ' + engine_path)

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Failed to create RTMPose TensorRT context')

        self.device = torch.device('cuda:{}'.format(device_id))
        self.stream = torch.cuda.Stream(device=self.device)
        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 1 or len(self.output_names) != 2:
            raise RuntimeError(
                'Expected RTMPose engine with 1 input and 2 outputs, got '
                '{} input(s), {} output(s)'.format(
                    len(self.input_names), len(self.output_names)))

        # Create a lightweight RTMPose object without its ONNX Runtime session.
        self.pose = RTMPose.__new__(RTMPose)
        self.pose.onnx_model = engine_path
        self.pose.model_input_size = (192, 256)
        self.pose.mean = (123.675, 116.28, 103.53)
        self.pose.std = (58.395, 57.12, 57.375)
        self.pose.to_openpose = False

    def _infer(self, image):
        # rtmlib preprocessing returns HWC float64; TensorRT expects NCHW float32.
        array = np.ascontiguousarray(
            image.transpose(2, 0, 1)[None], dtype=np.float32)
        with torch.cuda.stream(self.stream):
            input_tensor = torch.from_numpy(array).to(self.device)
            input_name = self.input_names[0]
            self.context.set_input_shape(input_name, tuple(input_tensor.shape))
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())

            output_tensors = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(
                        'Unresolved TensorRT output shape for {}: {}'.format(name, shape))
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
                self.context.set_tensor_address(name, tensor.data_ptr())
                output_tensors.append(tensor)

            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('RTMPose TensorRT execute_async_v3 failed')
        self.stream.synchronize()
        return [tensor.detach().cpu().numpy() for tensor in output_tensors]

    def __call__(self, image, bboxes):
        keypoints, scores = [], []
        for bbox in bboxes:
            crop, center, scale = self.pose.preprocess(image, bbox)
            outputs = self._infer(crop)
            kpts, score = self.pose.postprocess(outputs, center, scale)
            keypoints.append(kpts)
            scores.append(score)
        return np.concatenate(keypoints, axis=0), np.concatenate(scores, axis=0)


class PoseExtractorTensorRT:
    """Drop-in replacement for ``PoseExtractor`` using RTMPose TensorRT."""

    def __init__(self, checkpoint, device='0'):
        if str(device).lower() == 'cpu':
            raise ValueError('RTMPose TensorRT requires CUDA; use --onnx_device 0')
        if not checkpoint.lower().endswith('.engine'):
            raise ValueError('RTMPose TensorRT checkpoint must be a .engine file')
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                'RTMPose TensorRT engine not found: {}\n'
                'Build it on this Jetson with:\n'
                '  /usr/src/tensorrt/bin/trtexec '
                '--onnx=rtmpose/weights/rtmpose-s_256x192.onnx '
                '--minShapes=input:1x3x256x192 '
                '--optShapes=input:1x3x256x192 '
                '--maxShapes=input:1x3x256x192 '
                '--saveEngine={} --fp16'.format(checkpoint, checkpoint))

        device_id = int(str(device).split(':')[-1])
        torch.cuda.set_device(device_id)
        self.rtmpose = _TensorRTRTMPose(checkpoint, device_id)
        print('[RTMPose] Provider in use: TensorRT')
        print('[RTMPose] Engine: {}'.format(checkpoint))

    def extract(self, frame, bbox):
        skeletons, _ = self.extract_batch(frame, [bbox])
        return skeletons[0]

    def extract_batch(self, frame, bboxes, track_ids=None):
        boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        count = len(boxes)
        skeletons = np.zeros((count, 18, 3), dtype=np.float32)
        valid = np.zeros(count, dtype=bool)
        if count == 0:
            return skeletons, valid

        keypoints, scores = self.rtmpose(frame, boxes)
        result_count = min(count, len(keypoints), len(scores))
        for index in range(result_count):
            x1, y1, x2, y2 = boxes[index]
            width = max(float(x2 - x1), 1.0)
            height = max(float(y2 - y1), 1.0)
            skeletons[index, :17, 0] = (keypoints[index, :17, 0] - x1) / width
            skeletons[index, :17, 1] = (keypoints[index, :17, 1] - y1) / height
            skeletons[index, :17, 2] = scores[index, :17]
            skeletons[index, 17] = (
                skeletons[index, 5] + skeletons[index, 6]) / 2.0
            valid[index] = (
                np.isfinite(skeletons[index]).all()
                and np.any(skeletons[index, :17, 2] > 0))
        return skeletons, valid
