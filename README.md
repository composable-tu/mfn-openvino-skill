# 经 OpenVINO 优化的 MobileFaceNet 人脸身份识别 Skill

> [!note]
> 本项目需配套 [composable-tu/arcface-exp](https://github.com/composable-tu/arcface-exp) 项目中训练并经过 OpenVINO 导出的模型使用。
> 
> 项目自带模型基于 CASIA 人脸数据集训练到 ArcFace Loss 3.7，仅供演示使用，项目不对模型准确率做任何担保。

## 第一次使用？

安装该 Skill：

```Shell
npx skills add composable-tu/mfn-openvino-skill
```

`pip` 安装依赖：

```Shell
pip install numpy opencv-python openvino lancedb mediapipe pyarrow
```

如果你使用 `uv`：

```Shell
uv add numpy opencv-python openvino lancedb mediapipe pyarrow
```

然后，把 OpenVINO IR 模型文件放到：

- `openvino-face-skill/model/openvino/model.xml`
- `openvino-face-skill/model/openvino/model.bin`

## 用法

### 人脸入库

```Shell
python openvino-face-skill/scripts/enroll.py --name "alice" --image img_1.png
```

如果你使用 Astral uv：

```bash
uv run openvino-face-skill/scripts/enroll.py --name "alice" --image img_1.png
```

### 人脸识别

```Shell
python openvino-face-skill/scripts/identify.py --image img_1.png --threshold 0.35
```

如果使用 Astral uv：

```bash
uv run openvino-face-skill/scripts/identify.py --image img_1.png --threshold 0.35
```
