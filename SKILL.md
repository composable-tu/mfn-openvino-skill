---
name: openvino-face-db
description: 使用 OpenVINO IR 模型管理一个简单的人脸向量库。适用于“根据图片入库保存人脸向量/根据图片在已入库向量中识别是谁”等需求（OpenVINO 推理 + 余弦相似度比对）。
---

# OpenVINO 人脸向量库（入库 + 识别）

## 这个 Skill 做什么

本 Skill 提供两个可直接运行的命令（Python 脚本），用 **OpenVINO** 从图片提取人脸 embedding，然后：

- **入库（Enroll）**：将 embedding 以“人名/标签”为 key 保存到本地向量库目录
- **识别（Identify）**：将 embedding 与向量库做余弦相似度比对，返回最相似的人名与是否通过阈值

## 假设与限制

- 输入图片应为**已裁剪的人脸图**（本子项目不做人脸检测/对齐）。
- 模型输入为 RGB \(112x112\)，dtype 为 `uint8`（与原仓库 `test-openvino.py` 一致）。
- 输出向量会做 L2 归一化；相似度为余弦相似度（点积）。

## 快速开始

### 入库一个人

运行：

```bash
python openvino-face-skill/scripts/enroll.py --name "alice" --image "path/to/alice.png"
```

### 识别一个人

运行：

```bash
python openvino-face-skill/scripts/identify.py --image "path/to/unknown.png" --threshold 0.35
```

## 使用指引（给 Agent 的操作提示词）

当用户提出“存人脸向量 / 入库 / enroll”时：

1. 获取或推断参数：`--name`、`--image`，可选 `--db`（默认 `openvino-face-skill/db`）、`--model`（默认 `openvino-face-skill/model/openvino/model.xml`）
2. 运行入库脚本并返回：
   - 保存路径（`.npy` 文件路径）
   - 向量维度

当用户提出“判断是谁 / 识别 / identify / recognize”时：

1. 获取或推断参数：`--image`，可选 `--db`、`--model`（默认 `openvino-face-skill/model/openvino/model.xml`）、以及合理的 `--threshold`（默认 `0.35`）
2. 运行识别脚本并返回：
   - 最佳匹配人名
   - 相似度分数
   - 是否通过阈值
   - 若未通过：建议降低阈值或为同一人入库更多样本

