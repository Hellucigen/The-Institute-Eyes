# The Institute Eyes（天眼）

基于 **YOLOv8-Pose** 的实时姿态/行为检测桌面应用（Tkinter，V0.39），并集成了 **HSTforU** 视频异常检测模型的参考实现。

## 组成

| 文件 | 说明 |
| --- | --- |
| `live_inference.py` | 主应用：摄像头/视频实时推理、姿态关键点分析、检测结果画廊（YOLOv8n-Pose） |
| `HSTforU_Ped2.py` | HSTforU（Ped2 场景）模型封装，用于视频异常检测 |
| `default.py` | HSTforU 配置文件（随参考实现） |
| `test_images/` | 测试图片 |
| `SourceHanSansCN-Heavy.otf` | 思源黑体（界面中文字体，SIL OFL 许可） |

## 说明

- `HSTforU/` 为第三方参考实现（MIT License），因包含大量示例视频/图片（约 105MB）不入库，请自行下载后放入本目录。
- 运行主应用需安装：`ultralytics`、`opencv-python`、`numpy`、`pillow`，并确保 `yolov8n-pose.pt` 可自动下载或手动放置。
