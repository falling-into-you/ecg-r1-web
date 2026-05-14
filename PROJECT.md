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
- 配置：修改仓库内 `config.py`
- 启动 Web：`bash scripts/serve_web.sh`
- 启动直接 vLLM 推理：`bash scripts/serve_vllm.sh`
- 页面：`http://127.0.0.1:8000` 或 `http://<server-ip>:8000`

### tmux（推荐线上常驻）
- vLLM：`tmux new -d -s ecg-r1-rollout 'cd /data/jinjiarui/run/ecg-r1-web && bash scripts/serve_vllm.sh'`
- Web：`tmux new -d -s ecg-r1-web 'cd /data/jinjiarui/run/ecg-r1-web && bash scripts/serve_web.sh'`
- 查看日志：`tmux capture-pane -t ecg-r1-web -p -S -80`

### 停止与状态判断
- Web 和 vLLM 是两个独立进程。停止 Web 不会停止 vLLM；停止 vLLM 也不需要重启 Web。
- `ecg-r1-web` tmux 会话通常只负责 8000 端口的 FastAPI/uvicorn。
- `ecg-r1-rollout` tmux 会话通常负责 8023 端口的直接 vLLM 服务。
- 停止 vLLM 时，确认 8023 不再监听：
  - `tmux kill-session -t ecg-r1-rollout`
  - 或 `ss -ltnp 'sport = :8023'`
- `/status` 的 `online/loading/offline` 来自 Web 对 vLLM `VLLM_HEALTH_URL` 的健康检查，不来自特殊响应头。
- vLLM 健康检查不仅要求 8023 进程存活，还会在引擎生成异常后返回非 200，避免 EngineDead 仍显示 Online。
- vLLM 停止后，Web 可能在最近一次健康检查成功后的 120 秒内返回 `loading`，之后才返回 `offline`。

## Nginx 部署（域名代理到 8000）
DNS 解析无法直接携带端口；做法是让域名解析到服务器 IP，然后用 Nginx 在 80/443 端口反向代理到 `127.0.0.1:8000`。

### 1) DNS
- 添加 A 记录：`YOUR_DOMAIN` → 服务器公网 IPv4
- 可选：添加 AAAA 记录（IPv6）

### 2) Nginx 配置
仓库内提供示例配置：`deploy/nginx/ecg-r1-web.conf`，需要把 `YOUR_DOMAIN.com` 替换为真实域名。

Ubuntu 22.04 示例（需要 sudo 权限）：
- 安装：`sudo apt update && sudo apt install -y nginx`
- 方案 A（推荐，最稳）：直接用 conf.d 接管 80（避免默认站点/Host 匹配问题）
  - `sudo tee /etc/nginx/conf.d/ecg_r1_web.conf >/dev/null <<'EOF'`
  - `server {`
  - `    listen 80 default_server;`
  - `    listen [::]:80 default_server;`
  - `    server_name YOUR_DOMAIN.com _;`
  - `    client_max_body_size 50m;`
  - `    location / {`
  - `        proxy_pass http://127.0.0.1:8000;`
  - `        proxy_http_version 1.1;`
  - `        proxy_set_header Host $host;`
  - `        proxy_set_header X-Real-IP $remote_addr;`
  - `        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`
  - `        proxy_set_header X-Forwarded-Proto $scheme;`
  - `        proxy_buffering off;`
  - `        proxy_cache off;`
  - `        proxy_read_timeout 3600;`
  - `    }`
  - `}`
  - `EOF`
  - `sudo nginx -t`
  - `sudo systemctl restart nginx`
  - 验证：
    - `curl -s http://127.0.0.1:8000/status`
    - `curl -s -H "Host: YOUR_DOMAIN.com" http://127.0.0.1/status`
    - `curl -s -H "Host: YOUR_DOMAIN.com" http://127.0.0.1/ | head`
  - 可选清理（避免混淆）：`sudo rm -f /etc/nginx/sites-enabled/default`
  - 可选清理：`sudo rm -f /etc/nginx/sites-enabled/ecg-r1-web`
- 方案 B：使用 sites-enabled（更容易被 default / server_name 影响，不推荐）
  - `sudo cp deploy/nginx/ecg-r1-web.conf /etc/nginx/sites-available/ecg-r1-web`
  - `sudo ln -sf /etc/nginx/sites-available/ecg-r1-web /etc/nginx/sites-enabled/ecg-r1-web`
  - `sudo nginx -t && sudo systemctl reload nginx`

### 3) 关键参数（SSE/流式）
为避免中间层缓冲导致“流式无输出”，配置里启用了：
- `proxy_buffering off`
- `proxy_read_timeout 3600`

### 4) HTTPS（可选）
建议用 certbot 配置 443，并将 80 重定向到 https，再把 https 反代到 8000。

## 已实现功能（当前版本）
### 输入与推理
- 三模态输入：Image only / ECG signal only / Image + ECG signal
  - **ECG Signal**：强制要求选择 2 个文件（`.dat` + `.hea`），否则前端提示。
    - 后端会自动匹配 `.dat` 与 `.hea`，传递 Record Name 给推理引擎。
  - **ECG Image**：限制仅接受 `.png`, `.jpg`, `.jpeg`。
- 流式推理：前端优先使用 SSE（`/predict_stream`）
- 推理后端已拆成 provider：`mock` 可独立启动 Web，`vllm_direct` 连接独立直接 vLLM 服务；`swift_rollout` 仅保留为旧兼容 provider。
- 直接 vLLM 服务使用 `AsyncLLMEngine` 推理，不走 Swift GRPO rollout；`/predict_stream` 可接收真实生成增量。
- Web 进程不再通过 `ECG_R1_ROOT` 加载 ECG-R1 源码，也不在 FastAPI 启动时加载模型。
- 本仓库包含 `ecg_r1_runtime/` 与最小 `ecg_coca/` 运行时代码；模型权重和 ECG tower checkpoint 通过仓库内 `config.py` 指定。
- 兼容 IDE WebView/代理缓冲：SSE 无增量时自动切换轮询 `/predict_progress/{request_id}`
- 推理结果分区：
  - **Reasoning Process**：`<think>...</think>` 内容。
  - **Interpretation Summary**：主要诊断文本。
  - **Final Answer**：`<answer>...</answer>` 内容（如有），展示在单独的 Answer 区域。

### 结果展示与交互
- Report meta：Date / Model / Request ID
- Request ID 复制按钮
- Reasoning 可折叠/展开
- 结果展示分区：Interpretation Summary (原 Final Diagnosis) + Final Answer (新)
- 流式输出动效：打字机逐步追加 + 光标闪烁（完成后停止）
- Like / Dislike 反馈：点击后弹出反馈框，可填写可选文本并提交；`/feedback` 写入对应 request 的 `data.json`

### 数据落盘
- 推理请求会在 `data_collection/YYYY-MM-DD/{request_id}/` 保存：
  - `data.json`（包含输入、模型输出、反馈、用户信息、推理配置等）
  - 上传的 image/ecg 文件（如有）
- 不再使用单独的 `uploads/` 目录，避免重复与混乱

### 数据字段约定（data.json）
- `client.ip`：用户 IP（优先读 `X-Forwarded-For`，否则使用直连 IP）
- `client.geo`：地区信息（从请求头读取：`CF-IPCountry` / `X-Geo-*` 等；未提供则为空）
- `meta_info.model_display_name`：当前推理模型名称（与页面展示一致）
- `feedback`：点赞/点踩会更新 `feedback`、`feedback_at`、`feedback_client`，可选记录 `feedback_comment`

### Request ID 约定
- `request_id` 格式：`YYYYMMDD-<uuid>`（用于在无需额外索引的情况下定位落盘目录与回写反馈）

## 待实现功能（Backlog）
### 产品与交互
- 下载 JSON 按钮：导出本次 request 的 `data.json`
- Print 按钮：打印报告样式优化
- 明确展示当前推理阶段（排队/加载模型/生成中/完成）
- 一键清空/重置输入与输出

### 稳定性与性能
- 轮询接口增加过期清理：避免 `stream_states` 长期增长
- 并发控制：限制同一时间推理数量（队列/限流）
- 推理并发和长请求调度：直接 vLLM 已支持增量输出，后续需要补队列/限流。

### 可靠性与测试
- 增加前端 E2E/最小回归脚本（至少覆盖：上传、开始、看到增量、done、Request ID）
- 增加后端健康检查与 GPU/模型加载状态页（当前已有 /status，后续可扩展）

## 版本记录（手动维护）
- 2026-05-14：补充 vLLM/Web 独立启动停止说明；明确远程状态 badge 依赖 `/status` JSON；vLLM health 在引擎错误时返回非 200
- 2026-05-13：推理服务从 Swift GRPO rollout 切换为直接 vLLM `AsyncLLMEngine`；新增 `vllm_direct` provider 和 `scripts/serve_vllm.sh`，支持真实流式增量输出
- 2026-05-13：将启动配置收敛到仓库内 `config.py`；启动脚本不再读取外部 shell 环境变量作为配置来源，并按配置激活 conda 环境
- 2026-05-13：重构推理边界；新增 provider 架构、Swift rollout 适配、本仓库内 ECG-R1 vLLM runtime 代码、独立 Web/rollout 启动脚本
- 2026-01-31：第一版可用端到端 Demo；支持流式 + 轮询降级；UI 逐步完善
