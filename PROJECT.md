# ECG-R1 Web 项目文档

## 开发流程（强制）
在开始任何功能开发/修复前，先阅读本文件（PROJECT.md），确保不与现有设计冲突。

每次**功能更新**完成后必须：
1. 本地验证（至少能启动服务、跑一次推理或完成对应功能的最小验证）。
2. `git commit`（说明清楚做了什么）。
3. `git push` 到 GitHub。
4. 更新本文件的「已实现功能 / 待实现功能 / 版本记录」。

## 运行方式
### FastAPI
- 启动：`uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info`
- 页面：`http://localhost:8000`

### tmux（推荐线上常驻）
- 新建会话：`tmux new -s ecg_r1_web`
- 在会话内启动 uvicorn（同上）
- 退出但保持后台：`Ctrl-b d`

## 已实现功能（当前版本）
### 输入与推理
- 三模态输入：Image only / ECG signal only / Image + ECG signal
- 流式推理：前端优先使用 SSE（`/predict_stream`）
- 兼容 IDE WebView/代理缓冲：SSE 无增量时自动切换轮询 `/predict_progress/{request_id}`
- 推理结果分区：按 `<think>...</think>` 将内容分到 Reasoning 与 Diagnosis（并显示 `<think>` 标签）

### 结果展示与交互
- Report meta：Date / Model / Request ID
- Request ID 复制按钮
- Reasoning 可折叠/展开
- 流式输出动效：打字机逐步追加 + 光标闪烁（完成后停止）
- Like / Dislike 反馈：`/feedback` 写入对应 request 的 `data.json`

### 数据落盘
- 推理请求会在 `data_collection/{request_id}/` 保存：
  - `data.json`（包含输入、模型输出、反馈等）
  - 上传的 image/ecg 文件（如有）

## 待实现功能（Backlog）
### 产品与交互
- 下载 JSON 按钮：导出本次 request 的 `data.json`
- Print 按钮：打印报告样式优化
- 明确展示当前推理阶段（排队/加载模型/生成中/完成）
- 一键清空/重置输入与输出

### 稳定性与性能
- 轮询接口增加过期清理：避免 `stream_states` 长期增长
- 并发控制：限制同一时间推理数量（队列/限流）
- 更优的真正 token-level streaming（如果 swift 支持稳定 delta）

### 可靠性与测试
- 增加前端 E2E/最小回归脚本（至少覆盖：上传、开始、看到增量、done、Request ID）
- 增加后端健康检查与 GPU/模型加载状态页（当前已有 /status，后续可扩展）

## 版本记录（手动维护）
- 2026-01-31：第一版可用端到端 Demo；支持流式 + 轮询降级；UI 逐步完善
