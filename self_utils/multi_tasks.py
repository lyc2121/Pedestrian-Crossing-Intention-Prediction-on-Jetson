import cv2
import traceback
import time
import numpy as np
import torch

from .inference import yolov5_prediction, img_preprocessing, yolov8_prediction
from .post_processing import count_post_processing, count_post_processing_with_intention

try:
    from ultralytics import YOLO as YOLOv8
except ImportError:
    YOLOv8 = None


def _cuda_sync(device):
    if str(device) != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize(int(device))


def _profile_add(profile, module, started):
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    profile[module].append(elapsed_ms)


def Counting_Processing(input_img, yolo5_config, model, class_names, Tracker, Obj_Counter,
                        isCountPresent,
                        pose_extractor=None, track_buffer=None, intention_predictor=None,
                        predictions=None, fps=25.0,
                        frame_records=None, return_track_only=False,
                        latency_profile=None):
    try:
        if latency_profile is not None:
            _cuda_sync(yolo5_config.device)
        yolo_started = time.perf_counter()
        if YOLOv8 is not None and isinstance(model, YOLOv8):
            # ── YOLOv8 推理：坐标已在原图空间，inference_shape 传 None ──
            pred = yolov8_prediction(model, input_img, yolo5_config.conf_thres,
                                     yolo5_config.iou_thres, yolo5_config.classes)
            inference_shape = None
        else:
            # ── YOLOv5 推理：需要预处理 + scale_coords ──────────────────
            tensor_img = img_preprocessing(input_img, yolo5_config.device, yolo5_config.img_size)
            pred = yolov5_prediction(model, tensor_img, yolo5_config.conf_thres,
                                     yolo5_config.iou_thres, yolo5_config.classes)
            inference_shape = tensor_img.shape
        if latency_profile is not None:
            _cuda_sync(yolo5_config.device)
            _profile_add(latency_profile, 'YOLOv5', yolo_started)

        if pose_extractor is not None and track_buffer is not None and intention_predictor is not None:
            result_img = count_post_processing_with_intention(
                input_img, pred, class_names, inference_shape, Tracker, Obj_Counter,
                isCountPresent,
                pose_extractor, track_buffer, intention_predictor,
                predictions, fps,
                frame_records=frame_records,
                return_track_only=return_track_only,
                latency_profile=latency_profile)
        else:
            result_img = count_post_processing(input_img, pred, class_names,
                                               inference_shape, Tracker, Obj_Counter,
                                               isCountPresent)
        return result_img

    except Exception as e:
        traceback.print_exc()
        return e


def Background_Modeling(myP,input_img,save_path,bg_model):
    try:
        fg_mask = bg_model.apply(input_img)
        bg_img = bg_model.getBackgroundImage()
        cv2.putText(input_img,"origin image",(5,80),cv2.FONT_HERSHEY_TRIPLEX, 1.6, [0,200,0],thickness=3)
        cv2.putText(bg_img,"background image",(5,80),cv2.FONT_HERSHEY_TRIPLEX, 1.6, [0,200,0],thickness=3)
        result_img=np.vstack([input_img, bg_img])
        if myP is not None:
            myP.apply_async(cv2.imwrite,(save_path,result_img,))
        else:
            cv2.imwrite(save_path,result_img)
        return True,save_path
    except Exception as e:
        print("Wrong:",e,save_path)
        return False,e
