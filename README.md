# Pedestrian Crossing Intention Prediction on Jetson



本项目在 NVIDIA Jetson 上完成行人检测、跟踪、姿态估计和过街意图预测，并提供浏览器实时画面、结果视频、逐帧预测 CSV、延迟统计，以及针对检测结果的离线 LLM 标注与过街意图评价。



处理流程：



```text

输入视频/摄像头

     -> YOLO11s 行人检测

     -> DeepSORT 多目标跟踪

     -> RTMPose-s 人体关键点

     -> 16 帧时序缓存

     -> PedAST-GCN 过街意图预测

     -> 视频、CSV、延迟报告及 Flask 实时画面

```



## 1. 权重放置位置



运行前应具备以下目录结构：



```text

pedestrian_crossing_intention_prediction/

├── weights/

│   ├── yolo11s.pt                         # YOLO11s PyTorch 权重

│   └── yolo11s.engine                     # YOLO11s TensorRT 引擎

├── rtmpose/

│   └── weights/

│       ├── rtmpose-s_256x192.onnx         # RTMPose ONNX 权重

│       └── rtmpose-s_256x192-fp32.engine  # RTMPose TensorRT 引擎

├── PedAST-GCN/

│   └── best.onnx                          # 过街意图预测权重

└── deep_sort/

     └── deep_sort/deep/checkpoint/

         └── ckpt.t7                         # DeepSORT ReID 权重

```



| 文件 | 用途 | 必需情况 |

|---|---|---|

| `weights/yolo11s.pt` | YOLO11s 行人检测 | PyTorch YOLO 入口必需，也是生成 `.engine` 的源文件 |

| `weights/yolo11s.engine` | TensorRT 加速的 YOLO11s | TensorRT YOLO 入口必需 |

| `rtmpose/weights/rtmpose-s_256x192.onnx` | ONNX Runtime 姿态估计 | ONNX RTMPose 入口必需 |

| `rtmpose/weights/rtmpose-s_256x192-fp32.engine` | TensorRT 姿态估计 | TensorRT RTMPose 入口必需 |

| `PedAST-GCN/best.onnx` | 根据连续骨架和运动信息预测过街意图 | 所有 Jetson 入口必需 |

| `deep_sort/deep_sort/deep/checkpoint/ckpt.t7` | DeepSORT 外观特征提取 | 所有入口必需 |



> `.engine` 文件和 Jetson 的 GPU、JetPack、CUDA、TensorRT 版本相关。不要直接使用其他电脑或不同 Jetson 环境生成的 engine，推荐在实际运行的 Jetson 上重新生成。



## 2. 权重下载与生成



### 2.1 YOLO11s



从 [Ultralytics 官方 YOLO11 页面](https://docs.ultralytics.com/models/yolo11/)下载，或在项目根目录执行：



```bash

mkdir -p weights

wget -O weights/yolo11s.pt \\

   https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt

```



也可以让 Ultralytics 自动下载：



```bash

yolo predict model=yolo11s.pt source=input/test.mp4

mv yolo11s.pt weights/

```



在目标 Jetson 上生成 TensorRT FP16 engine：



```bash

yolo export model=weights/yolo11s.pt format=engine imgsz=640 half=True device=0

```



生成结果应为 `weights/yolo11s.engine`。Ultralytics 的导出参数说明见[官方模型导出文档](https://docs.ultralytics.com/modes/export/)。



### 2.2 RTMPose-s



模型来自 [rtmlib 官方模型库](https://github.com/Tau-J/rtmlib)。下载并解压：



```bash

mkdir -p rtmpose/weights

wget -O /tmp/rtmpose-s.zip \\

   https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip

unzip /tmp/rtmpose-s.zip -d /tmp/rtmpose-s

find /tmp/rtmpose-s -name '*.onnx'

```



将找到的 ONNX 文件复制并统一命名：



```bash

find /tmp/rtmpose-s -name '*.onnx' -exec \\

   cp {} rtmpose/weights/rtmpose-s_256x192.onnx \\;

```



如果 ONNX 位于解压后的子目录，请用 `find` 输出的实际路径替换上述源路径。



在目标 Jetson 上生成代码默认使用的 FP32 engine：



```bash

/usr/src/tensorrt/bin/trtexec \\

   --onnx=rtmpose/weights/rtmpose-s_256x192.onnx \\

   --saveEngine=rtmpose/weights/rtmpose-s_256x192-fp32.engine

```



项目默认选择 FP32 RTMPose engine，以获得更稳定的关键点结果。若更重视速度，也可以加 `--fp16` 生成 FP16 engine，再通过 `--pose_engine` 指定它。



### 2.3 DeepSORT



从 [deep_sort_pytorch 官方仓库说明](https://github.com/ZQPei/deep_sort_pytorch)中的 Google Drive 下载 `ckpt.t7`，放到：



```text

deep_sort/deep_sort/deep/checkpoint/ckpt.t7

```



官方下载目录：



```text

https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6

```



### 2.4 PedAST-GCN



`PedAST-GCN/best.onnx` 是本项目的过街意图预测模型，不是 Ultralytics 或 RTMPose 的公共权重。请从本项目的 GitHub Release、项目作者提供的网盘或训练产物中获取，并放到：



```text

PedAST-GCN/best.onnx

```



如果文件使用其他名称，可以在运行时指定：



```bash

--onnx_path PedAST-GCN/你的模型.onnx

```



## 3. 推理入口和评估工具



### 3.1 八个推理入口

这 8 个文件由 4 种推理后端组合及各自的文件夹批处理版本构成。



| 入口文件 | YOLO11s | RTMPose-s | 输入方式 | 特点 |

|---|---|---|---|---|

| `main_onnx_jetson_yolo11s.py` | PyTorch `.pt` | ONNX Runtime `.onnx` | 单视频/摄像头 | 基准版本，兼容性好，便于调试 |

| `main_onnx_jetson_yolo11s_folder.py` | PyTorch `.pt` | ONNX Runtime `.onnx` | 文件夹 | 对应基准版本的批处理包装器 |

| `main_onnx_jetson_yolo11s_TensorRT.py` | TensorRT `.engine` | ONNX Runtime `.onnx` | 单视频/摄像头 | 只加速 YOLO |

| `main_onnx_jetson_yolo11s_TensorRT_folder.py` | TensorRT `.engine` | ONNX Runtime `.onnx` | 文件夹 | 对应 YOLO TensorRT 版本的批处理包装器 |

| `main_onnx_jetson_yolo11s_TensorRT_RTMPoses.py` | PyTorch `.pt` | TensorRT `.engine` | 单视频/摄像头 | 只加速 RTMPose；尽管文件名容易让人以为 YOLO 也用了 TensorRT，实际实现以代码为准 |

| `main_onnx_jetson_yolo11s_TensorRT_RTMPoses_folder.py` | PyTorch `.pt` | TensorRT `.engine` | 文件夹 | 对应 RTMPose TensorRT 版本的批处理包装器 |

| `main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT.py` | TensorRT `.engine` | TensorRT `.engine` | 单视频/摄像头 | YOLO 和 RTMPose 都使用 TensorRT，通常最快 |

| `main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT_folder.py` | TensorRT `.engine` | TensorRT `.engine` | 文件夹 | 双 TensorRT 的批处理包装器 |



所有版本都会继续使用：



- DeepSORT：行人 ID 跟踪；

- 16 帧 temporal buffer：为每个行人保存连续时序信息；

- PedAST-GCN ONNX：输出过街意图及概率；

- Flask：通过浏览器显示实时结果；

- 视频和 CSV：保存完整结果与性能统计。



选择建议：



- 首次部署和排查问题：使用 `main_onnx_jetson_yolo11s.py`；

- 只想提高检测速度：使用 `main_onnx_jetson_yolo11s_TensorRT.py`；

- RTMPose 成为主要瓶颈：使用 `main_onnx_jetson_yolo11s_TensorRT_RTMPoses.py`；

- 正式部署且 engine 均已在当前 Jetson 生成：使用双 TensorRT 版本；

- 处理一个目录里的所有视频：选择对应的 `_folder.py`。

### 3.2 标注与评价工具

| 文件 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `LLM for pedestrian state recognition.py` | 一个 `<name>_track_only.mp4` | `<视频文件名>_annotations.csv` | 调用 Qwen 为带 bbox/ID 的视频生成逐帧行人状态标注 |
| `evaluate_crossing_intention.py` | 一份 annotations CSV 和一份 prediction CSV | `<name>_overall_metrics.csv`、`<name>_per_id_metrics.csv` | 评价单个视频 |
| `run_pipeline_batch_llm_eval.py` | `*_folder.py` 产生的 `results root` | 每个视频的标注和评价 CSV、`batch_llm_eval_summary.csv` | 批量执行 LLM 标注和单视频评价 |
| `aggregate_crossing_intention_metrics.py` | 已含 prediction/annotations 的 `results root` | `aggregate_overall_metrics.csv`、`aggregate_per_video_metrics.csv` | 合并所有视频样本，计算整体微平均指标 |

### 3.3 主要公共模块

| 模块 | 主要类/函数 | 作用 |
|---|---|---|
| `self_utils/video_folder_batch_runner.py` | `run_folder()` | 四个文件夹入口共用的批处理、结果目录和自动评价逻辑 |
| `self_utils/pose_extractor.py` | `PoseExtractor` | ONNX Runtime RTMPose 姿态提取 |
| `self_utils/pose_extractor_tensorrt.py` | `PoseExtractorTensorRT` | TensorRT RTMPose 姿态提取 |
| `self_utils/track_buffer.py` | `TrackBuffer` | 按行人 ID 保存意图模型所需的时序数据 |
| `self_utils/intention_predictor_onnx.py` | `IntentionPredictorONNX` | 调用 PedAST-GCN ONNX 模型进行意图预测 |
| `evaluate_crossing_intention.py` | `calculate_metrics()` | 单视频评价和总体汇总共用的指标计算函数 |



## 4. 环境安装



推荐环境：JetPack 6.x、Python 3.10，并使用与 JetPack 匹配的 PyTorch、TorchVision、TensorRT 和 ONNX Runtime GPU。



安装项目的通用 Python 依赖：



```bash

python3 -m pip install -r requirements_jetson.txt

```



Jetson 上的以下组件不要盲目安装桌面版 pip wheel：



- PyTorch 和 TorchVision：使用 NVIDIA 对应 JetPack 的 Jetson wheel；

- TensorRT：由 JetPack 提供；

- OpenCV：优先使用 JetPack 自带且支持 CUDA/GStreamer 的版本；

- ONNX Runtime GPU：使用与 JetPack、Python 和 aarch64 匹配的 wheel。



如果项目携带的 wheel 与当前系统匹配，可安装：



```bash

python3 -m pip install \\

   wheels/onnxruntime_gpu-1.20.1-cp310-cp310-linux_aarch64.whl

```



## 5. 运行方法



所有命令都应在项目根目录执行，因为配置和权重默认使用相对路径。



### 单个视频：基准版本



```bash

python3 main_onnx_jetson_yolo11s.py \\

   --input input/test.mp4 \\

   --output output \\

   --device 0 \\

   --onnx_device 0

```



### 单个视频：YOLO TensorRT



```bash

python3 main_onnx_jetson_yolo11s_TensorRT.py \\

   --input input/test.mp4 \\

   --weights weights/yolo11s.engine \\

   --output output

```



### 单个视频：双 TensorRT



```bash

python3 main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT.py \\

   --input input/test.mp4 \\

   --weights weights/yolo11s.engine \\

   --pose_engine rtmpose/weights/rtmpose-s_256x192-fp32.engine \\

   --onnx_path PedAST-GCN/best.onnx \\

   --output output

```



### 文件夹批处理



```bash

python3 main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT_folder.py \\

   --input input/paper_test \\

   --output output_jetson \\

   --recursive

```



`_folder.py` 支持的视频格式包括 MP4、AVI、MOV、MKV、WMV、WebM、MPEG、TS 和 MTS。它会逐个调用对应的单视频入口，并生成 `batch_summary.csv`。额外参数会继续传给单视频程序，例如：



```bash

python3 main_onnx_jetson_yolo11s_TensorRT_folder.py \\

   --input input/paper_test \\

   --output output_jetson \\

   --weights weights/yolo11s.engine \\

   --conf_thres 0.3

```



### 摄像头



```bash

python3 main_onnx_jetson_yolo11s.py --input 0

```



## 6. 浏览器实时画面



程序启动后，在同一局域网电脑的浏览器中访问：



```text

http://<Jetson-IP>:5000

```



例如 Jetson IP 为 `192.168.1.108`：



```text

http://192.168.1.108:5000

```



选择浏览器显示内容：



```bash

--display_mode full        # 完整检测、骨架及意图画面

--display_mode track_only  # 仅跟踪画面

--port 5000                # 修改服务端口

```



如果无法访问，请确认电脑和 Jetson 位于同一网络，并检查 Jetson 防火墙是否允许对应端口。



## 7. 输出文件



单个视频会在 `--output` 下建立以输入视频和后端命名的子目录，并产生：



```text

<name>_full.mp4             # 完整标注视频

<name>_track_only.mp4       # 仅跟踪视频

<name>_prediction.csv       # 每帧、每个行人的过街意图和概率

<name>_latency_profile.csv  # 各模块平均、P50、P95 延迟及占比

```



预测 CSV 字段：



| 字段 | 含义 |

|---|---|

| `frame_id` | 视频帧编号 |

| `time_sec` | 当前帧对应时间 |

| `person_id` | DeepSORT 跟踪 ID |

| `pred_intent` | 预测的过街意图类别 |

| `crossing_prob` | 预测为过街的概率 |



文件夹批处理还会输出 `batch_summary.csv`，记录每个视频是否成功、返回码和处理耗时。



## 8. 常用参数



| 参数 | 默认值 | 说明 |

|---|---:|---|

| `--input` | 代码中的本地测试路径 | 视频路径、摄像头编号或文件夹入口的目录路径 |

| `--output` | `./output` | 输出根目录 |

| `--weights` | `weights/yolo11s.pt` | YOLO `.pt` 或对应入口要求的 `.engine` |

| `--onnx_path` | `PedAST-GCN/best.onnx` | PedAST-GCN ONNX 权重 |

| `--pose_engine` | RTMPose FP32 engine | 仅 TensorRT RTMPose 入口支持 |

| `--img_size` | `640` | YOLO 输入尺寸，必须与 engine 导出尺寸一致 |

| `--conf_thres` | `0.3` | 检测置信度阈值 |

| `--iou_thres` | `0.4` | NMS IoU 阈值 |

| `--device` | `0` | YOLO/DeepSORT CUDA 设备；基准入口也支持 `cpu` |

| `--onnx_device` | `0` | ONNX Runtime CUDA 设备或 `cpu` |

| `--profile_warmup` | `30` | 延迟统计前忽略的预热帧数 |



## 9. 常见问题



### 找不到权重



确认从项目根目录运行，并检查文件：



```bash

ls -lh weights/yolo11s.pt

ls -lh deep_sort/deep_sort/deep/checkpoint/ckpt.t7

ls -lh rtmpose/weights/rtmpose-s_256x192.onnx

ls -lh PedAST-GCN/best.onnx

```



### TensorRT engine 无法加载



通常是 engine 来自不同 JetPack、TensorRT 或硬件环境。删除旧 engine，并在当前 Jetson 上使用原始 `.pt`/`.onnx` 重新生成。



### CUDA 或 ONNX Runtime provider 不可用



检查：



```bash

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

python3 -c "import tensorrt as trt; print(trt.__version__)"

```



GPU ONNX Runtime 应显示 `CUDAExecutionProvider`。如果只有 `CPUExecutionProvider`，说明安装的 wheel 不支持当前 Jetson CUDA 环境。



### 输出视频无法播放或摄像头打不开



优先检查 JetPack OpenCV 是否带 GStreamer 支持：



```bash

python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer

```

## 10. 文件夹检测与批量评估串联

`*_folder.py` 的实际结果目录是 `<--output>/<单视频脚本名>/`。批处理结束时会打印
`results root` 和一条可以直接复制的 `evaluation cmd`。

完整数据流如下：

```text
输入视频目录
  -> *_folder.py
  -> <output>/<单视频脚本名>/<视频输出名>/
       ├── <视频输出名>_track_only.mp4
       └── <视频输出名>_prediction.csv
  -> run_pipeline_batch_llm_eval.py
       ├── <视频输出名>_annotations.csv
       ├── <视频输出名>_overall_metrics.csv
       └── <视频输出名>_per_id_metrics.csv
  -> aggregate_crossing_intention_metrics.py
       ├── aggregate_overall_metrics.csv
       └── aggregate_per_video_metrics.csv
```

调用 LLM 前需要设置百炼 API Key：

```bash
export DASHSCOPE_API_KEY="你的 API Key"
```

也可以用 `--eval-after` 在检测完成后自动评估：

```bash
python3 main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT_folder.py \
  --input input/paper_test \
  --output output_jetson \
  --eval-after \
  --eval-skip-existing
```

分开运行时，把检测阶段打印的 `results root` 原样传给评估脚本：

```bash
python3 run_pipeline_batch_llm_eval.py \
  --input-dir output_jetson/main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT
```

常用批量评价选项：

| 参数 | 说明 |
|---|---|
| `--skip-existing` | 已存在 `<name>_overall_metrics.csv` 时跳过该视频 |
| `--force-llm` | 忽略已有 annotations，重新调用 LLM |
| `--stop-on-error` | 一个视频失败后立即停止，默认记录错误后继续 |
| `--llm_sample_fps` | LLM 每秒标注的目标帧数，默认 10 |
| `--sample_interval` | 评价时每个行人 ID 的采样帧间隔，默认 3 |
| `--future_start_sec` / `--future_end_sec` | 未来真实意图窗口，默认 1–2 秒 |

所有视频完成单独评价后，可进一步生成跨视频整体指标：

```bash
python3 aggregate_crossing_intention_metrics.py \
  --input-dir output_jetson/main_onnx_jetson_yolo11s_TensorRT_RTMPoses_TensorRT
```

如果只评价一个视频，也可以直接调用：

```bash
python3 evaluate_crossing_intention.py \
  --annotations output_jetson/.../<name>_annotations.csv \
  --predictions output_jetson/.../<name>_prediction.csv \
  --output-dir output_jetson/... \
  --video-name <name> \
  --fps 25
```
