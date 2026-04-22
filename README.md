# MobileFaceNet 人脸身份识别 OpenVINO Skill

> [!note]
> 本项目需配套 [composable-tu/arcface-exp](https://github.com/composable-tu/arcface-exp) 项目中训练并经过 OpenVINO 导出的模型使用。

## 第一次使用？

`pip` 安装依赖：

```Shell
python -m pip install openvino opencv-python numpy
```

如果你使用 `uv`：

```Shell
uv add openvino opencv-python numpy
```

然后，把 OpenVINO IR 模型文件放到：

- `openvino-face-skill/model/openvino/model.xml`
- `openvino-face-skill/model/openvino/model.bin`

## 用法

### 人脸入库

```Shell
python openvino-face-skill/scripts/enroll.py --name "alice" --image img_1.png
```

### 人脸识别

```Shell
python openvino-face-skill/scripts/identify.py --image img_1.png --threshold 0.35
```

## 向量库格式

- 向量库目录包含：
  - `index.json`：入库记录列表（人名 + embedding 文件相对路径）
  - `embeddings/*.npy`：L2 归一化后的 embedding 向量

