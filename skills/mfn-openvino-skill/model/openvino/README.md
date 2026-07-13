# 模型放置目录

请把 OpenVINO IR 模型文件放到这里：

```
openvino-face-skill/
  model/
    openvino/
      model.xml
      model.bin
```

脚本默认会读取：

- `openvino-face-skill/model/openvino/model.xml`

如果你的模型文件名不同，运行脚本时请显式传入 `--model` 指向对应的 `*.xml` 文件。

