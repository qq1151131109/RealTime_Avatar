# 第三方依赖安装指南

本项目依赖以下第三方库，需要单独克隆和设置。

## LivePortrait

LivePortrait 是人脸渲染的核心引擎。

### 安装步骤

```bash
# 1. 克隆 LivePortrait 仓库
cd engine/third_party/
git clone https://github.com/KwaiVGI/LivePortrait.git

# 2. 下载预训练模型
cd LivePortrait
# 按照 LivePortrait 官方文档下载模型权重
# 通常放在 LivePortrait/pretrained_weights/ 目录
```

详细安装说明请参考：[LivePortrait 官方文档](https://github.com/KwaiVGI/LivePortrait)

## Duix-Mobile（可选）

如果你使用的 Avatar 源图像位于 Duix-Mobile 项目中，需要克隆该仓库：

```bash
# 在项目根目录
git clone <Duix-Mobile仓库地址> Duix-Mobile
```

或者，你可以：
1. 将 Avatar 源图像复制到其他位置
2. 修改 `engine/avatars/demo/avatar.json` 中的 `source_image` 路径

## 验证安装

运行系统检查脚本：

```bash
python check_system.py
```

该脚本会检查所有必需的依赖是否正确安装。

## 目录结构

安装完成后，目录结构应该如下：

```
实时数字人/
├── engine/
│   ├── third_party/
│   │   └── LivePortrait/          # LivePortrait 仓库
│   │       ├── src/
│   │       ├── pretrained_weights/ # 模型权重
│   │       └── ...
│   └── ...
├── Duix-Mobile/                    # 可选，仅当使用其中的 Avatar 图像时
└── ...
```

## 常见问题

### 1. LivePortrait 模型权重下载失败

如果网络问题导致下载失败，可以：
- 使用代理或镜像站点
- 手动下载模型文件并放到正确位置

### 2. Avatar 源图像路径错误

如果运行时提示找不到源图像，检查：
- `engine/avatars/demo/avatar.json` 中的 `source_image` 路径是否正确
- 路径是相对于 avatar 目录的相对路径

## 获取帮助

如遇到问题，请：
1. 运行 `python check_system.py` 检查依赖
2. 查看项目 [README.md](../README.md)
3. 提交 Issue

---

安装完成后，你可以运行：

```bash
# 测试系统（占位符模式）
python -m engine.demo_realtime_stub
```
