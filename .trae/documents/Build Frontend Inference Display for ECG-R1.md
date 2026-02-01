# ECG-R1 Web 平台技术设计文档

## 1. 项目概述
ECG-R1 Web 是一个专为心电图（ECG）智能诊断模型设计的交互式前端平台。它基于 FastAPI 构建后端服务，采用原生 HTML/JS/CSS 实现轻量级前端，支持流式输出（Streaming）、多文件上传、实时解析思维链（Chain of Thought）以及用户反馈收集。

## 2. 系统架构

### 2.1 技术栈
- **后端**: Python 3.10+, FastAPI, Uvicorn, Jinja2 Templates
- **前端**: Vanilla JavaScript (ES6+), CSS3 (Flex/Grid), HTML5
- **通信**: SSE（流式）+ RESTful API
- **部署**: Nginx 反向代理 + 单实例 FastAPI
  - **8000**: FastAPI/uvicorn 进程监听（内部端口）
  - **80**: Web 前端入口（Nginx -> 8000）
  - **44000**: 远程推理 API 入口（Nginx -> 8000，API-only）

### 2.2 数据流向
1. **上传**: 用户上传 `.dat` 和 `.hea` 文件（或图片） -> 后端暂存 -> 调用 `ECG_R1` 模型引擎。
2. **推理**: 模型生成 Token 流 -> 后端封装为 SSE 事件 -> 推送至前端。
3. **解析**: 前端接收 chunk -> 解析标签（`<think>`, `<answer>`）-> 分流到不同 UI 区域。
4. **反馈**: 用户点赞/点踩并可选填写文本 -> `/feedback` 写入 `data.json`。

## 3. 关键功能与核心代码

### 3.1 文件上传与路径处理（wfdb）
后端支持同时接收多个文件，并自动识别 `wfdb` 格式所需的 Record Name（去除后缀），避免出现 `.hea.hea` 之类的路径拼接问题。

关键实现：后端接收文件后，将 `objects_dict['ecg']` 指向 record 的不带后缀路径。

### 3.2 流式推理与标签解析（前端状态机）
前端实现了一个流式解析状态机，实时识别模型输出中的 XML 风格标签，并分流展示：
- **Reasoning Process**：`<think>...</think>`
- **Interpretation Summary**：常规文本段落
- **Final Answer**：`<answer>...</answer>`

关键实现位置：
- [script.js](file:///data/jinjiarui/run/ecg-r1-web/static/script.js)

### 3.3 打字机效果与缓冲渲染
为避免流式输出过快导致 DOM 高频更新卡顿，前端把内容先累积到 `pending*` 缓冲区，再用 `requestAnimationFrame` 分片渲染（打字机效果）。

关键实现位置：
- [script.js](file:///data/jinjiarui/run/ecg-r1-web/static/script.js)

### 3.4 用户反馈系统（Like/Dislike + Optional Text）
用户点击点赞/点踩后弹出反馈框：
- 展示“你已点赞/点踩”的提示
- 提供可选文本输入（Optional）
- Submit 后调用 `/feedback` 写入 `data.json`

关键实现位置：
- [index.html](file:///data/jinjiarui/run/ecg-r1-web/templates/index.html)
- [style.css](file:///data/jinjiarui/run/ecg-r1-web/static/style.css)
- [script.js](file:///data/jinjiarui/run/ecg-r1-web/static/script.js)
- [main.py](file:///data/jinjiarui/run/ecg-r1-web/main.py)

### 3.5 移动端响应式
小屏幕下由左右两列改为上下布局，避免移动端“右侧内容消失”。

关键实现位置：
- [responsive.css](file:///data/jinjiarui/run/ecg-r1-web/static/responsive.css)

### 3.6 图标本地化
将 FontAwesome 资源下载到本地并通过本地 CSS 引用，同时设置心跳图标为站点 Favicon。

关键实现位置：
- [index.html](file:///data/jinjiarui/run/ecg-r1-web/templates/index.html)
- [static/icons](file:///data/jinjiarui/run/ecg-r1-web/static/icons)

## 4. 远程推理 API（Port 44000）

### 4.1 设计目标
为“未来前端不在本机服务器部署”的场景提供稳定的远程推理入口：
- 复用同一个 FastAPI 推理进程（8000），避免重复加载模型
- 通过 Nginx 新增暴露 44000 端口
- 44000 端口为 **API-only**：仅放行推理相关路径，其它路径（包括 `/`）返回 404
- 允许跨域调用（CORS）

### 4.2 端口行为
- `GET http://<server-ip>:44000/status`
- `POST http://<server-ip>:44000/predict_stream`（SSE）
- `POST http://<server-ip>:44000/predict`（JSON）
- `POST http://<server-ip>:44000/feedback`（可选）
- `GET http://<server-ip>:44000/` -> 404

### 4.3 Nginx 配置
系统生效配置文件：`/etc/nginx/conf.d/ecg_r1_web.conf`

仓库内提供可复用的版本化配置：
- [deploy/nginx/ecg_r1_web.conf](file:///data/jinjiarui/run/ecg-r1-web/deploy/nginx/ecg_r1_web.conf)

### 4.4 CORS（FastAPI）
后端通过 `CORSMiddleware` 允许跨域访问（不使用 cookies 场景）。

关键实现位置：
- [main.py](file:///data/jinjiarui/run/ecg-r1-web/main.py)

### 4.5 调用示例（curl）
流式推理（multipart/form-data，字段命名与现有前端保持一致）：

```bash
curl -N -X POST http://<server-ip>:44000/predict_stream \
  -F "ecg=@record.dat" \
  -F "ecg=@record.hea"
```

## 5. 目录结构
```
/data/jinjiarui/run/ecg-r1-web/
├── main.py
├── templates/
│   └── index.html
├── static/
│   ├── script.js
│   ├── style.css
│   ├── responsive.css
│   └── icons/
├── deploy/
│   └── nginx/
│       └── ecg_r1_web.conf
├── data.json
└── PROJECT.md
```
