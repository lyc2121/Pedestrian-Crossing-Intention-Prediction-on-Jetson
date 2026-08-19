"""
main_onnx_jetson.py
===================
Jetson 专用版本：去掉 cv2.imshow，改用 Flask MJPEG 串流。
在本地浏览器打开 http://<jetson-ip>:5000 即可实时查看。

用法：
    python main_onnx_jetson.py --input ./video_0001.mp4
    python main_onnx_jetson.py --input 0          # 摄像头
"""

import torch, sys, argparse, cv2, os, time, threading, csv
import numpy as np
from datetime import datetime
from flask import Flask, Response
from ultralytics import YOLO
from self_utils.multi_tasks import Counting_Processing
from self_utils.overall_method import Object_Counter, Image_Capture
from self_utils.pose_extractor import PoseExtractor
from self_utils.track_buffer import TrackBuffer
from self_utils.intention_predictor_onnx import IntentionPredictorONNX
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort

# ── Flask 串流 ────────────────────────────────────────────────────────────────
app = Flask(__name__)
_frame_lock = threading.Lock()
_current_frame = None   # 最新处理帧（JPEG bytes）
output_name_suffix = '_yolo11s'


def _set_frame(bgr_img):
    global _current_frame
    _, buf = cv2.imencode('.jpg', bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with _frame_lock:
        _current_frame = buf.tobytes()


def _generate():
    while True:
        with _frame_lock:
            frame = _current_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.02)   # ~50fps 上限，实际受推理速度限制


@app.route('/video_feed')
def video_feed():
    return Response(_generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return '<html><body style="background:#000;margin:0">' \
           '<img src="/video_feed" style="max-width:100%;height:auto"></body></html>'


def _start_flask(port=5000):
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)   # 屏蔽每帧请求日志
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


def get_video_name(input_path):
    if os.path.isdir(input_path):
        return os.path.basename(os.path.normpath(input_path)) or "images"
    if str(input_path).isdigit():
        return "camera{}".format(input_path)
    if str(input_path).startswith(('rtsp', 'rtmp')):
        return "stream"
    return os.path.splitext(os.path.basename(input_path))[0]


def save_latency_profile(path, samples, frame_count):
    modules = [
        ('YOLO11', 'YOLOv5'),
        ('DeepSORT', 'DeepSORT'),
        ('RTMPose-s', 'RTMPose-s'),
        ('Temporal buffer', 'Temporal buffer'),
        ('PedAST-GCN', 'PedAST-GCN'),
    ]
    e2e = np.asarray(samples['End-to-end'], dtype=np.float64)
    e2e_total = float(e2e.sum()) if e2e.size else 0.0
    with open(path, 'w', newline='', encoding='utf-8') as profile_file:
        writer = csv.writer(profile_file)
        writer.writerow([
            'Module', 'Calls', 'Mean/call (ms)', 'P50 (ms)', 'P95 (ms)',
            'Per-frame (ms)', 'Runtime proportion',
        ])
        for display_name, key in modules:
            values = np.asarray(samples[key], dtype=np.float64)
            total = float(values.sum()) if values.size else 0.0
            writer.writerow([
                display_name, int(values.size),
                '{:.3f}'.format(float(values.mean()) if values.size else 0.0),
                '{:.3f}'.format(float(np.percentile(values, 50)) if values.size else 0.0),
                '{:.3f}'.format(float(np.percentile(values, 95)) if values.size else 0.0),
                '{:.3f}'.format(total / max(frame_count, 1)),
                '{:.2f}%'.format(total * 100.0 / e2e_total if e2e_total else 0.0),
            ])
        writer.writerow([
            'End-to-end', int(e2e.size),
            '{:.3f}'.format(float(e2e.mean()) if e2e.size else 0.0),
            '{:.3f}'.format(float(np.percentile(e2e, 50)) if e2e.size else 0.0),
            '{:.3f}'.format(float(np.percentile(e2e, 95)) if e2e.size else 0.0),
            '{:.3f}'.format(float(e2e.mean()) if e2e.size else 0.0), '100.00%',
        ])


def load_yolo_model(config):
    """Load the regular PyTorch Ultralytics model."""
    model = YOLO(config.weights)
    if config.device != "cpu":
        model.to("cuda:{}".format(config.device))
    return model


# ── 主推理函数 ────────────────────────────────────────────────────────────────
def main(yolo5_config):
    print("=> main task started: {}".format(datetime.now().strftime('%H:%M:%S')))
    print("=> Flask stream: http://0.0.0.0:{} (用 Jetson IP 访问)".format(yolo5_config.port))

    # 启动 Flask 后台线程
    t = threading.Thread(target=_start_flask, args=(yolo5_config.port,), daemon=True)
    t.start()

    a = time.time()

    # ── YOLOv8/11 加载 ───────────────────────────────────────────────────────
    Model = load_yolo_model(yolo5_config)

    class_names = [Model.names[0]]
    b = time.time()
    print("==> class names: ", class_names)
    print("=> load YOLO, cost:{:.2f}s".format(b - a))

    c = time.time()

    # ── DeepSORT 初始化 ───────────────────────────────────────────────────────
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort_tracker = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=(yolo5_config.device != 'cpu'),
        use_appearence=True,
    )

    # ── 意图预测组件 ──────────────────────────────────────────────────────────
    print("=> loading PoseExtractor (RTMPose-s) ...")
    pose_extractor = PoseExtractor(
        checkpoint="rtmpose/weights/rtmpose-s_256x192.onnx",
        device=yolo5_config.onnx_device,
    )
    track_buffer = TrackBuffer(window_size=16)

    print("=> loading IntentionPredictor (PedAST-GCN ONNX) ...")
    intention_predictor = IntentionPredictorONNX(
        onnx_path=yolo5_config.onnx_path,
        device=yolo5_config.onnx_device,
    )
    predictions = {}

    # ── 输出目录（与 main_onnx.py 一致）──────────────────────────────────────
    video_name = get_video_name(yolo5_config.input) + output_name_suffix
    out_dir = os.path.join(yolo5_config.output, video_name)
    os.makedirs(out_dir, exist_ok=True)
    full_video_path = os.path.join(out_dir, '{}_full.mp4'.format(video_name))
    track_only_video_path = os.path.join(out_dir, '{}_track_only.mp4'.format(video_name))
    csv_path = os.path.join(out_dir, '{}_prediction.csv'.format(video_name))
    latency_path = os.path.join(out_dir, '{}_latency_profile.csv'.format(video_name))
    latency_profile = {
        'YOLOv5': [], 'DeepSORT': [], 'RTMPose-s': [],
        'Temporal buffer': [], 'PedAST-GCN': [], 'End-to-end': [],
        '_temporal_ms': 0.0,
    }

    # ── 视频输入 ──────────────────────────────────────────────────────────────
    mycap       = Image_Capture(yolo5_config.input)
    Obj_Counter = Object_Counter(class_names)
    total_num   = mycap.get_length()
    fps = int(mycap.get(5))
    if fps == 0:
        fps = 25
    frame_id = 0
    full_videowriter = None
    track_only_videowriter = None
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_id', 'time_sec', 'person_id', 'pred_intent', 'crossing_prob'])

    _t_prev = time.time()
    while mycap.ifcontinue():
        ret, img, *_ = mycap.read()
        if ret:
            profile_this_frame = frame_id >= yolo5_config.profile_warmup
            active_profile = latency_profile if profile_this_frame else None
            if profile_this_frame and yolo5_config.device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize(int(yolo5_config.device))
            frame_started = time.perf_counter()
            frame_records = []
            result = Counting_Processing(
                img, yolo5_config, Model, class_names, deepsort_tracker, Obj_Counter,
                isCountPresent=False,
                pose_extractor=pose_extractor,
                track_buffer=track_buffer,
                intention_predictor=intention_predictor,
                predictions=predictions,
                fps=fps,
                frame_records=frame_records,
                return_track_only=True,
                latency_profile=active_profile,
            )
            if profile_this_frame:
                if yolo5_config.device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize(int(yolo5_config.device))
                latency_profile['End-to-end'].append(
                    (time.perf_counter() - frame_started) * 1000.0)
            if isinstance(result, Exception):
                print("错误为{}".format(result))
                break
            full_img, track_only_img = result

            # 计算实时 FPS 并叠加到画面
            _t_now = time.time()
            _real_fps = 1.0 / max(_t_now - _t_prev, 1e-6)
            _t_prev = _t_now
            cv2.putText(full_img, f"FPS: {_real_fps:.1f}",
                        (10, full_img.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            for trackid, label, prob in frame_records:
                csv_writer.writerow([
                    frame_id, '{:.3f}'.format(frame_id / fps), trackid,
                    label, '{:.4f}'.format(prob),
                ])

            if full_videowriter is None:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                size = (full_img.shape[1], full_img.shape[0])
                full_videowriter = cv2.VideoWriter(full_video_path, fourcc, fps, size)
                track_only_videowriter = cv2.VideoWriter(
                    track_only_video_path, fourcc, fps,
                    (track_only_img.shape[1], track_only_img.shape[0]))
            full_videowriter.write(full_img)
            track_only_videowriter.write(track_only_img)
            frame_id += 1

            # 推送到 Flask 串流
            stream_img = full_img if yolo5_config.display_mode == 'full' else track_only_img
            _set_frame(stream_img)

        sys.stdout.write("\r=> processing at %d; total: %d" % (mycap.get_index(), total_num))
        sys.stdout.flush()

    if full_videowriter is not None:
        full_videowriter.release()
    if track_only_videowriter is not None:
        track_only_videowriter.release()
    csv_file.close()
    mycap.release()
    save_latency_profile(latency_path, latency_profile,
                         len(latency_profile['End-to-end']))
    print("\n=> process done, total cost: {:.2f}s".format(time.time() - c))
    print("=> main task finished: {}".format(datetime.now().strftime('%H:%M:%S')))
    print("=> full video       : {}".format(full_video_path))
    print("=> track_only video : {}".format(track_only_video_path))
    print("=> prediction csv   : {}".format(csv_path))
    print("=> latency profile  : {}".format(latency_path))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='input video path',
                        # default='0',
                        help='video path or camera index (0)')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--weights', type=str, default='weights/yolo11s.pt')
    parser.add_argument('--onnx_path', type=str, default='PedAST-GCN/best.onnx')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--conf_thres', type=float, default=0.3)
    parser.add_argument('--iou_thres', type=float, default=0.4)
    parser.add_argument('--device', default='0',
                        help='YOLO/DeepSORT CUDA device index or "cpu"')
    parser.add_argument('--onnx_device', default='0',
                        help='ONNX device: CUDA index (default: 0) or "cpu"')
    parser.add_argument('--display_mode', default='full', choices=['full', 'track_only'],
                        help='which output is streamed to the browser')
    parser.add_argument('--classes', default=0, type=int)
    parser.add_argument('--port', type=int, default=5000,
                        help='Flask streaming port')
    parser.add_argument('--profile_warmup', type=int, default=30,
                        help='initial frames excluded from latency statistics')
    return parser


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = build_parser()
    yolo5_config = parser.parse_args()
    print(yolo5_config)
    main(yolo5_config)
    print("结果保存在：", yolo5_config.output)
