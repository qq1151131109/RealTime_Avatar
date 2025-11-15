# Repository Guidelines

本仓库用于在 PC 端实现 2D 照片级实时数字人引擎，并参考/集成 Duix-Mobile 与 LivePortrait 相关资源。

## 项目结构与模块组织

- `engine/`：PC 端数字人引擎原型（配置、Motion、渲染、实时 Demo 等），日常开发的主要目录。  
- `engine/third_party/LivePortrait/`：渲染内核依赖，请尽量避免修改上游源码。  
- `Duix-Mobile/`：上游移动端 SDK 与文档，仅作参考与资源来源，不建议直接改动。  
- `数字人引擎落地清单.md`：需求、阶段规划与当前进展的总览文档。

## 构建、运行与开发命令

- 安装依赖：`cd engine && pip install -r requirements.txt`  
- 创建配置：`cp engine/config.example.json engine/config.json` 并按需调整。  
- 运行实时 Demo：`python -m engine.demo_realtime_stub`（默认使用示例 Avatar 和占位 Audio2Motion）。  
- Android / iOS 集成参考：见 `Duix-Mobile/duix-android` 与 `Duix-Mobile/duix-ios` 中各自的 `README_zh.md`。

## 代码风格与命名约定

- Python：使用 4 空格缩进，遵循接近 PEP 8 的风格；模块、函数使用 `snake_case`，类使用 `PascalCase`。  
- 配置：统一使用 JSON（如 `config.json`），字段命名与 `EngineConfig` 保持一致。  
- 不在 `third_party/` 与 `Duix-Mobile/` 中做大规模重构；如需修改，上游风格优先。

## 测试指南

- 目前尚未引入完整测试目录，新增测试推荐使用 `pytest`，放在 `engine/tests/` 或 `tests/`，文件命名为 `test_*.py`。  
- 在提交前至少本地跑通：  
  - 单元测试（如存在）：`pytest`  
  - 实时 Demo：`python -m engine.demo_realtime_stub`，确认头像渲染与基本同步正常。

## Commit 与 Pull Request 规范

- Commit 信息使用简短祈使句，突出影响范围，例如：`engine: add pts sync helper`。  
- 一个 Commit/PR 聚焦一个清晰的修改主题（如“眨眼逻辑调整”或“配置系统增强”）。  
- PR 描述中说明：变更动机、主要改动点、测试方式与结果；如涉及可视效果，建议附上截图或简短录屏。

## Agent 使用说明（给 AI 助手）

- 回答与文档更新请使用简体中文。  
- 代码改动应优先集中在 `engine/`，不要随意修改 `third_party/` 和 `Duix-Mobile/`。  
- 变更公共接口或配置结构时，务必同步更新 `engine/README.md` 与 `数字人引擎落地清单.md` 中的相关描述。  
- 保持补丁最小化、可读且与现有架构一致，避免无必要的重构和依赖引入。

