# ECG-R1 Web 平台技术设计文档

## 1. 项目概述
ECG-R1 Web 是一个专为心电图（ECG）智能诊断模型设计的交互式前端平台。它基于 FastAPI 构建后端服务，采用原生 HTML/JS/CSS 实现轻量级前端，支持流式输出（Streaming）、多文件上传、实时解析思维链（Chain of Thought）以及用户反馈收集。

## 2. 系统架构

### 2.1 技术栈
- **后端**: Python 3.10+, FastAPI, Uvicorn, Jinja2 Templates
- **前端**: Vanilla JavaScript (ES6+), CSS3 (Flex/Grid), HTML5
- **通信**: HTTP/2 (SSE for streaming), RESTful API
- **部署**: Nginx 反向代理 (Port 80 -> 8000), Systemd Service

### 2.2 数据流向
1. **上传**: 用户上传 `.dat` 和 `.hea` 文件 -> 后端暂存 -> 调用 `ECG_R1` 模型引擎。
2. **推理**: 模型生成 Token 流 -> 后端封装为 SSE 事件 (`data: {...}`) -> 推送至前端。
3. **解析**: 前端接收 Chunk -> 正则匹配标签 (`<think>`, `<answer>`) -> 分流至不同 UI 区域。
4. **反馈**: 用户提交反馈 -> 后端追加写入 `data.json`。

## 3. 关键功能与核心代码

### 3.1 文件上传与路径处理
后端支持同时接收多个文件，并自动识别 `wfdb` 格式所需的 Record Name（去除后缀），解决了 `.hea.hea` 路径拼接错误问题。

**关键代码 (`main.py`)**:
```python
@app.post("/predict_stream")
async def predict_stream(ecg: list[UploadFile] = File(...)):
    # ... 保存文件 ...
    # 自动提取 Record Name (如 "record_100.dat" -> "record_100")
    record_name = os.path.splitext(ecg[0].filename)[0]
    objects_dict['ecg'] = os.path.join(temp_dir, record_name) 
    # ...
```

### 3.2 流式推理与标签解析
前端实现了一个状态机，能够实时解析模型输出的 XML 风格标签，将内容动态分流到三个区域：
1. **Reasoning Process** (`<think>...</think>`)
2. **Interpretation Summary** (常规文本)
3. **Final Answer** (`<answer>...</answer>`)

**关键代码 (`static/script.js` - `routeContentChunk`)**:
```javascript
function routeContentChunk(chunk) {
    // 处理 buffer 和边界情况
    // 状态机检测 <think>, </think>, <answer>, </answer>
    if (tag === '<think>') {
        streamState.inThink = true;
    } else if (tag === '<answer>') {
        streamState.inAnswer = true;
        // 显示隐藏的 Answer 区域
        if (answerSection.classList.contains('hidden')) {
             answerSection.classList.remove('hidden');
        }
    }
    // ... 分流内容到 pendingReasoning / pendingAnswer / pendingDiagnosis
}
```

### 3.3 打字机效果与缓冲区管理
为了保证流畅的视觉体验，前端使用了 `requestAnimationFrame` 实现平滑的打字机效果，并维护了独立的缓冲区队列，防止流式速度过快导致页面卡顿。

**关键代码 (`static/script.js` - `flushStep`)**:
```javascript
function flushStep() {
    const step = 48; // 每次渲染字符数
    // 优先级：Reasoning > Answer > Diagnosis
    if (pendingReasoning) {
        // ... 渲染 Reasoning ...
    } else if (pendingAnswer) {
        // ... 渲染 Final Answer ...
    } else if (pendingDiagnosis) {
        // ... 渲染 Summary ...
    }
    // 循环调用直到队列清空
}
```

### 3.4 用户反馈系统
支持在推理过程中或结束后随时提交反馈。系统通过 Response Header 立即获取 `X-Request-ID`，确保用户无需等待推理完成即可点赞/点踩。

**关键代码 (`static/script.js` & `main.py`)**:
- **前端**: `currentRequestId = response.headers.get('x-request-id');` (立即获取 ID)
- **后端**: 
```python
@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    entry = {
        "request_id": request.request_id,
        "feedback_type": request.feedback_type,
        "feedback_comment": request.comment, # 可选评论
        "timestamp": datetime.now().isoformat()
    }
    # 线程安全地写入 data.json
```

## 4. UI/UX 设计细节
- **响应式布局**: 通过 `static/responsive.css` 适配移动端，小屏幕下自动切换为单列垂直布局。
- **状态反馈**: 加载状态、打字机光标 (`.streaming` 类)、结果区域的动态展开/隐藏。
- **图标系统**: 本地化 FontAwesome 图标库，确保离线环境可用；自定义心跳 Favicon。

## 5. 目录结构
```
/data/jinjiarui/run/ecg-r1-web/
├── main.py              # 后端核心逻辑
├── templates/
│   └── index.html       # 主页面结构
├── static/
│   ├── script.js        # 前端交互与流式解析
│   ├── style.css        # 核心样式
│   ├── responsive.css   # 移动端适配样式
│   └── icons/           # 本地图标资源
├── data.json            # 反馈数据存储
└── PROJECT.md           # 项目变更日志
```
