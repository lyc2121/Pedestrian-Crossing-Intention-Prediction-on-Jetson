import cv2,random,torch,time
import math
import numpy as np
from skimage import draw
from .globals_val import Global
from utils.utils import scale_coords,plot_one_box

np.set_printoptions(precision=3, suppress=True, linewidth=200)


# ── COCO-17 关节连线定义（含虚拟第 17 号节点：index5/6 均值）────────────────
# 关节索引:
#  0=鼻子  1=左眼  2=右眼  3=左耳  4=右耳
#  5=左肩  6=右肩  7=左肘  8=右肘  9=左腕  10=右腕
# 11=左髋 12=右髋 13=左膝 14=右膝 15=左踝  16=右踝
# 17=虚拟躯干中心（左肩[5]+右肩[6]均值）
#
# 连线与 graph.py neighbor_link_coco18 保持一致：
# (10,8),(8,6),(9,7),(7,5)           右臂、左臂
# (15,13),(13,11),(16,14),(14,12)    左腿、右腿
# (11,17),(12,17),(5,17),(6,17)      髋/肩 → 躯干中心
# (17,0),(0,1),(0,2),(2,4),(1,3)     躯干中心 → 头部
_SKELETON_PAIRS = [
    (10,  8), ( 8,  6),   # 右臂
    ( 9,  7), ( 7,  5),   # 左臂
    (15, 13), (13, 11),   # 左腿
    (16, 14), (14, 12),   # 右腿
    (11, 17), (12, 17),   # 髋 → 躯干中心
    ( 5, 17), ( 6, 17),   # 肩 → 躯干中心
    (17,  0),             # 躯干中心 → 鼻子
    ( 0,  1), ( 0,  2),   # 鼻子 → 眼
    ( 2,  4), ( 1,  3),   # 眼 → 耳
]

# 各肢体段颜色 (BGR)，顺序与 _SKELETON_PAIRS 一一对应
_PAIR_COLORS = [
    (255,  50, 200), (255,  50, 200),   # 右臂 - 粉
    ( 50, 200, 255), ( 50, 200, 255),   # 左臂 - 黄
    (255, 255,   0), (255, 255,   0),   # 左腿 - 黄
    (  0, 128, 255), (  0, 128, 255),   # 右腿 - 蓝
    (  0, 220, 220), (  0, 220, 220),   # 髋→中心 - 青
    (  0, 255, 128), (  0, 255, 128),   # 肩→中心 - 青绿
    (180, 180, 180),                    # 中心→鼻 - 灰
    (255, 128,   0), (255, 128,   0),   # 眼 - 橙
    (255, 128,   0), (255, 128,   0),   # 耳 - 橙
]


def _cuda_sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_add(profile, module, started):
    profile[module].append((time.perf_counter() - started) * 1000.0)


def draw_skeleton(img, skeleton, bbox, conf_thr=0.3, joint_r=3, line_w=2):
    """
    将 18 关节骨架绘制到图像上。

    Args:
        img      : BGR 图像（原地修改）
        skeleton : np.ndarray (18, 3) [x_rel, y_rel, conf]，坐标已按 bbox 归一化
        bbox     : [x1, y1, x2, y2] 像素坐标
        conf_thr : 低于此置信度的关节不绘制
        joint_r  : 关节圆点半径
        line_w   : 连线粗细
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)

    # 归一化坐标 → 像素坐标
    pts = []
    for j in range(18):
        px = int(skeleton[j, 0] * bbox_w + x1)
        py = int(skeleton[j, 1] * bbox_h + y1)
        pts.append((px, py, float(skeleton[j, 2])))

    # 画连线
    for idx, (i, j) in enumerate(_SKELETON_PAIRS):
        if pts[i][2] >= conf_thr and pts[j][2] >= conf_thr:
            cv2.line(img, pts[i][:2], pts[j][:2], _PAIR_COLORS[idx], line_w, cv2.LINE_AA)

    # 画关节点
    for px, py, conf in pts:
        if conf >= conf_thr:
            cv2.circle(img, (px, py), joint_r, (0, 0, 255), -1, cv2.LINE_AA)


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

# min_w_ratio=0.030, min_h_ratio=0.11
    # min_w_ratio=0.020, min_h_ratio=0.090
    # min_w_ratio=0.013, min_h_ratio=0.074
    # min_w_ratio=0.007, min_h_ratio=0.035
def deepsort_update(Tracker, pred, inference_shape, np_img, min_box_w_ratio=0.03, min_box_h_ratio=0.11):
    # YOLOv5 的框位于缩放后的推理图上；YOLOv8/11 返回的已经是原图坐标，
    # 后者由调用方用 inference_shape=None 标识。
    if inference_shape is not None:
        pred[:, :4] = scale_coords(inference_shape[2:], pred[:, :4], np_img.shape).round()
    else:
        pred[:, :4] = pred[:, :4].round()

    # 过滤掉宽/高太小的检测框（远处/噪声行人），不让它们进入跟踪器，
    # 因此也不会被分配 track id、不会被计数、不会被画出来。
    # 用相对图像宽高的比例而非绝对像素值，避免不同分辨率下同样大小的
    # 行人因像素数不同而被区别对待。
    img_h, img_w = np_img.shape[:2]
    min_box_w = min_box_w_ratio * img_w
    min_box_h = min_box_h_ratio * img_h
    w = pred[:, 2] - pred[:, 0]
    h = pred[:, 3] - pred[:, 1]
    pred = pred[(w >= min_box_w) & (h >= min_box_h)]

    bbox_xywh = []
    confs = []
    labels = []
    for *xyxy, conf, cls in pred:
        img_h, img_w, _ = np_img.shape
        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
        obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(obj)
        confs.append([conf.item()])
        labels.append(int(cls))
    # 显式指定形状，避免空列表被 torch.Tensor 转成 1 维张量
    # （所有框都被尺寸过滤掉时会发生），导致下游 [:, 0] 索引报错。
    xywhs = torch.Tensor(bbox_xywh).reshape(-1, 4)
    confss = torch.Tensor(confs).reshape(-1, 1)
    outputs = Tracker.update(xywhs, confss , labels, np_img)
    return outputs


def count_post_processing(np_img,pred,class_names,inference_shape,Tracker,Obj_Counter, isCountPresent):
    """
        isCountPresent:
            True：表示只显示当前人数
            False：表示显示总人数和当前人数
    """
    present_num = 0
    if isCountPresent:
        text = "present person"
    else:
        text = "total person"
    if pred is not None and len(pred):
        outputs = deepsort_update(Tracker, pred, inference_shape, np_img)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, 5]
            present_num = len(identities)
            Global.total_person = Global.total_person | set(identities)
            for i in range(len(outputs)):
                box = bbox_xyxy[i]
                trackid = identities[i]
                text_info = '%s,ID:%d' % (class_names[0], int(trackid))
                plot_one_box(box, np_img, text_info=text_info, color=(0, 0, 255))
    # 可视化计数结果
    total_num = len(Global.total_person)
    np_img = Obj_Counter.draw_counter(np_img, present_num, total_num, text, isCountPresent)
    return np_img

                
def count_post_processing_with_intention(
        np_img, pred, class_names, inference_shape, Tracker, Obj_Counter,
        isCountPresent,
        pose_extractor, track_buffer, intention_predictor,
        predictions, fps=25.0, frame_records=None, return_track_only=False,
        latency_profile=None):
    """
    Drop-in replacement for count_post_processing that additionally
    extracts skeleton keypoints per track, maintains a 16-frame sliding
    window, and overlays crossing-intention predictions on the video.

    Args:
        pose_extractor    : PoseExtractor instance
        track_buffer      : TrackBuffer instance
        intention_predictor: IntentionPredictor instance
        predictions (dict): {track_id: (label, prob)}  —  mutated in-place
        frame_records (list, optional): 若提供，本帧每个已产出预测结果的
            track 会以 (track_id, label, prob) 追加进该 list，供调用方
            写逐帧预测文档使用。
        return_track_only (bool): True 时额外返回一份只画 bbox+ID
            （不含骨架/意图文字）的图像，用于单独导出 track_only 视频。

    Returns:
        np_img，或 (np_img, track_only_img)（return_track_only=True 时）
    """
    present_num = 0
    active_set  = set()
    text = "present person" if isCountPresent else "total person"
    img_h, img_w = np_img.shape[:2]

    track_only_img = np_img.copy() if return_track_only else None

    if pred is not None and len(pred):
        if latency_profile is not None:
            _cuda_sync_if_needed()
        stage_started = time.perf_counter()
        outputs = deepsort_update(Tracker, pred, inference_shape, np_img)
        if latency_profile is not None:
            _cuda_sync_if_needed()
            _profile_add(latency_profile, 'DeepSORT', stage_started)

        if len(outputs) > 0:
            bbox_xyxy  = outputs[:, :4]
            identities = outputs[:, 5]
            present_num = len(identities)
            Global.total_person = Global.total_person | set(identities)

            n_persons = len(outputs)

            if return_track_only:
                for i in range(n_persons):
                    box = bbox_xyxy[i]
                    trackid = int(identities[i])
                    text_info = '%s,ID:%d' % (class_names[0], trackid)
                    plot_one_box(box, track_only_img, text_info=text_info, color=(0, 0, 255))

            # ── 批量姿态估计 ────────────────────────────────────────────
            # skeletons_all: (N,18,3)
            # pose_valid   : (N,) bool，False 表示该人本帧姿态估计被跳过/失败
            #                （如 bbox 太小退化），对应骨架为全 0
            stage_started = time.perf_counter()
            skeletons_all, pose_valid = pose_extractor.extract_batch(
                np_img, bbox_xyxy, track_ids=[int(tid) for tid in identities]
            )
            if latency_profile is not None:
                _profile_add(latency_profile, 'RTMPose-s', stage_started)

            # ── 更新滑窗 & 收集需要推理的 track ────────────────────────
            infer_ids   = []   # 需要运行 PedAST-GCN 的 track id 列表
            infer_sk    = []   # 对应的 skeleton tensors
            infer_box   = []   # 对应的 box tensors

            stage_started = time.perf_counter()
            for i in range(n_persons):
                box     = bbox_xyxy[i]
                trackid = int(identities[i])
                skeleton = skeletons_all[i]

                if pose_valid[i]:
                    is_full = track_buffer.update(trackid, box.tolist(), skeleton)
                else:
                    # 姿态估计不可靠：当作 miss 处理，不作为 'real' 帧
                    # 污染滑窗，也不提前凑够 min_frames 触发不可靠的预测。
                    track_buffer.mark_miss(trackid)
                    is_full = False

                if is_full:
                    sk_t, box_t = track_buffer.get_sample(trackid, img_w, img_h)
                    if sk_t is not None:
                        infer_ids.append(trackid)
                        infer_sk.append(sk_t)
                        infer_box.append(box_t)
            if latency_profile is not None:
                latency_profile['_temporal_ms'] += (time.perf_counter() - stage_started) * 1000.0

            # ── 批量 PedAST-GCN 推理（一次 forward 处理所有待推理 track）─
            if infer_ids:
                stage_started = time.perf_counter()
                import torch
                sk_batch  = torch.cat(infer_sk,  dim=0)  # (B,16,18,3)
                box_batch = torch.cat(infer_box, dim=0)  # (B,16, 4,2)
                labels_batch, probs_batch = intention_predictor.predict(sk_batch, box_batch)
                if latency_profile is not None:
                    _profile_add(latency_profile, 'PedAST-GCN', stage_started)
                for tid, lbl, prob, sk_t in zip(infer_ids, labels_batch, probs_batch, infer_sk):
                    predictions[tid] = (lbl, prob)
                    print(f"[GCN] ID:{tid} | {lbl} | prob={prob:.4f}")

            # ── 可视化 ──────────────────────────────────────────────────
            # 只有意图预测结果出来之后才画框/骨架，避免过早显示无意图信息的框
            for i in range(n_persons):
                box     = bbox_xyxy[i]
                trackid = int(identities[i])
                if trackid not in predictions:
                    continue

                label, prob = predictions[trackid]
                if frame_records is not None:
                    frame_records.append((trackid, label, prob))
                color     = (0, 0, 255) if label == 'crossing' else (0, 255, 0)
                text_info = '%s,ID:%d | %s(%.2f)' % (class_names[0], trackid, label, prob)
                if pose_valid[i]:
                    draw_skeleton(np_img, skeletons_all[i], box)

                plot_one_box(box, np_img, text_info=text_info, color=color)

            active_set = set(int(tid) for tid in identities)

    # 补全本帧未检测到的 track 的缺失帧，再清理彻底消失的 track
    stage_started = time.perf_counter()
    track_buffer.pad_miss(active_set)
    # 序列中所有 real 帧都太小（一直很远）的 track 直接丢弃 buffer，
    # 不让其进入意图预测。
    track_buffer.prune_all_small(img_w, img_h)
    track_buffer.cleanup(active_set)
    if latency_profile is not None:
        latency_profile['_temporal_ms'] += (time.perf_counter() - stage_started) * 1000.0
        latency_profile['Temporal buffer'].append(latency_profile['_temporal_ms'])
        latency_profile['_temporal_ms'] = 0.0

    total_num = len(Global.total_person)
    np_img = Obj_Counter.draw_counter(np_img, present_num, total_num, text, isCountPresent)
    if return_track_only:
        track_only_img = Obj_Counter.draw_counter(track_only_img, present_num, total_num, text, isCountPresent)
        return np_img, track_only_img
    return np_img


def draw_obj_dense(img,box_list,k_size=281,beta=1.5):
    value=np.ones((img.shape[0],img.shape[1])).astype('uint8')
    value=value*10
    value=fill_box(box_list,value)
    value=cv2.GaussianBlur(value, ksize=(k_size,k_size),sigmaX=0,sigmaY=0)
    color=value_to_color(value)
    color=cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    value[value<=20]=0.9
    value[value>20]=1.0
    mask=np.ones_like(img)
    mask[:,:,0]=value
    mask[:,:,1]=value
    mask[:,:,2]=value
    mask_color=mask*color
    mask_color=cv2.GaussianBlur(mask_color, ksize=(7,7),sigmaX=0,sigmaY=0)
    result = cv2.addWeighted(img, 1, mask_color, beta, 0)
    info='Total number: {}'.format(len(box_list))
    W_size,H_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.8 , 2)[0]
    cv2.putText(result, info, (3, 1+H_size+9), cv2.FONT_HERSHEY_TRIPLEX, 0.8, [0,255,0], 2)
    return result


def between(x,x_min,x_max):
    return min(x_max,max(x,x_min))


def fill_box(box_list,mask,fill_size=25):
    for box in box_list:
        cenXY=[(box[0]+box[2])/2,(box[1]+box[3])/2]
        cenXY=[between(cenXY[0],0+fill_size,mask.shape[1]-fill_size),between(cenXY[1],0+fill_size,mask.shape[0]-fill_size)]
        Y=np.array([cenXY[1]-fill_size,cenXY[1]-fill_size,cenXY[1]+fill_size,cenXY[1]+fill_size])
        X=np.array([cenXY[0]-fill_size,cenXY[0]+fill_size,cenXY[0]+fill_size,cenXY[0]-fill_size])
        yy, xx=draw.polygon(Y,X)
        mask[yy, xx] = 255
    return mask


def value_to_color(grayimg,low_value=15,high_value=220,low_color=[10,10,10],high_color=[255,10,10]):
    r=low_color[0]+((grayimg-low_value)/(high_value-low_value))*(high_color[0]-low_color[0])
    g=low_color[1]+((grayimg-low_value)/(high_value-low_value))*(high_color[1]-low_color[1])
    b=low_color[2]+((grayimg-low_value)/(high_value-low_value))*(high_color[2]-low_color[2])
    rgb=np.ones((grayimg.shape[0],grayimg.shape[1],3))
    rgb[:,:,0]=r
    rgb[:,:,1]=g
    rgb[:,:,2]=b
    return rgb.astype('uint8')
