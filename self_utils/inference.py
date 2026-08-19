import torch
import numpy as np
from utils.datasets import letterbox
from utils.utils import non_max_suppression


def img_preprocessing(np_img,device,newsize=640):
    np_img=letterbox(np_img,new_shape=newsize)[0]
    np_img = np_img[:, :, ::-1].transpose(2, 0, 1)
    np_img = np.ascontiguousarray(np_img)
    if device != "cpu":
        tensor_img=torch.from_numpy(np_img).to("cuda:{}".format(device))
    else:
        tensor_img = torch.from_numpy(np_img)
    tensor_img=tensor_img[np.newaxis,:].float()
    tensor_img /= 255.0
    return tensor_img


def yolov5_prediction(model,tensor_img,conf_thres,iou_thres,classes):
    # print(classes)
    with torch.no_grad():
        out=model(tensor_img)[0]
        pred = non_max_suppression(out, conf_thres, iou_thres, classes=classes)[0]
    return pred


def yolov8_prediction(model, np_img, conf_thres, iou_thres, classes):
    """
    YOLOv8 推理，返回与 yolov5_prediction 相同格式的 tensor (N, 6)：
    [x1, y1, x2, y2, conf, cls]，坐标已在原图空间，无需 scale_coords。
    """
    cls_filter = [classes] if isinstance(classes, int) else list(classes)
    results = model(np_img, conf=conf_thres, iou=iou_thres,
                    classes=cls_filter, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu()
    conf = boxes.conf.cpu().unsqueeze(1)
    cls  = boxes.cls.cpu().unsqueeze(1)
    return torch.cat([xyxy, conf, cls], dim=1)  # (N, 6)
