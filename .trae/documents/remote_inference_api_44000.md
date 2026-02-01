# ECG-R1 远程推理服务（Port 44000）技术文档

本文档面向后续接手的工程师，描述显卡服务器“推理服务入口”的端口设计、接口协议、请求/响应格式、流式（SSE）事件语义、以及典型调用方式。

## 1. 背景与目标

未来 Web 页面可能部署在非显卡服务器（例如业务服务器 / 网关 / 静态站点托管）。显卡服务器只负责：

* 加载模型

* 接收推理请求

* 返回推理结果（含流式输出）

因此需要一个对外稳定的 API 端口：**44000**。

## 2. 端口与部署

### 2.1 端口约定

* **8000**：FastAPI/uvicorn 实际监听端口（内部端口）

* **80**：当前 Web 页面入口（Nginx -> 8000）

* **44000**：远程推理 API 入口（Nginx -> 8000，API-only）

### 2.2 重要设计点：为什么 44000 不是“再起一个 44000 的 FastAPI”

当前采用 **单实例 FastAPI（8000） + Nginx 多端口暴露**：

* 避免在 44000 再起一个 uvicorn 导致模型重复加载 / 显存重复占用

* 44000 在 Nginx 层实现 **API 白名单**，从网络边界直接限制“只能访问推理相关接口”

### 2.3 Nginx 配置

* 系统生效：`/etc/nginx/conf.d/ecg_r1_web.conf`

* 仓库模板（建议以此为准做部署）：[ecg\_r1\_web.conf](file:///data/jinjiarui/run/ecg-r1-web/deploy/nginx/ecg_r1_web.conf)

在 44000 端口上：

* 仅显式放行少数 `location = /xxx` 路径

* 其它路径统一 `404`（含 `/`），确保不会把前端页面暴露到 44000

## 3. 认证与安全（现状）

* 当前：未做鉴权（默认内网/可信环境）

* CORS：

  * Nginx 对 44000 增加 `Access-Control-Allow-*` 头

  * FastAPI 使用 `CORSMiddleware`（`allow_origins=["*"]` 且 `allow_credentials=False`）

如果需要对公网开放，建议后续在 Nginx 层增加：

* IP 白名单 / Basic Auth / JWT 网关 / 速率限制

## 4. 接口总览

> Base URL：`http://<gpu-server-ip>:44000`

| Method | Path             | 说明                      |
| ------ | ---------------- | ----------------------- |
| GET    | /status          | 健康检查 + 模型加载状态           |
| POST   | /predict\_stream | 流式推理（SSE）               |
| POST   | /predict         | 非流式推理（JSON）             |
| POST   | /feedback        | 对某次 request\_id 的反馈（可选） |

> API-only 行为：

* `GET /` => 404

* 未在白名单内的路径 => 404

## 5. 公共约定

### 5.1 输入格式（保持与现有前端一致）

推理接口采用 `multipart/form-data`。

支持两类输入（至少提供一种）：

* **image**：ECG 图片（png/jpg）

* **ecg**：ECG 信号文件（必须同时提供 `.dat` 与 `.hea`，字段名相同 `ecg`，多文件上传）

### 5.3 客户端 IP 与地理信息透传（非常重要）

当“业务前端服务器”作为反向代理把请求转发到显卡服务器 `:44000` 时，如果不透传客户端 IP，显卡服务器侧记录的 `client.ip` 可能会变成代理服务器的 IP。

推荐做法：由前端服务器在转发时追加/透传以下头部（标准做法）：

* `X-Forwarded-For: $proxy_add_x_forwarded_for`（保留链路：client, proxy1, proxy2...）

* `X-Real-IP: $remote_addr`

如果前端服务器本身在 CDN/网关之后（例如 Cloudflare），也可以额外透传：

* `CF-Connecting-IP`

* `CF-IPCountry`

为了在 Analytics 页面显示到“地区”级别，建议前端服务器（或网关）在可获得地理信息时透传：

* `X-Geo-Country`（国家码）

* `X-Geo-Region`（地区/省州）

* `X-Geo-City`（城市）

前端服务器 Nginx 示例片段：

```nginx
location / {
    proxy_pass http://<gpu-server-ip>:44000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

显卡服务器侧解析优先级：

* `CF-Connecting-IP` / `True-Client-IP` / `X-Real-IP`（若存在）

* 否则解析 `X-Forwarded-For` 的链路

安全提示：如果对公网开放 44000，这些转发头部可能被客户端伪造；推荐让 44000 仅对前端服务器开放。

### 5.2 Request ID

* `/predict`：在 JSON 返回体里返回 `request_id`

* `/predict_stream`：

  * 在响应头返回 `X-Request-ID`

  * 在 SSE 第一条 `event: ready` 的 data 中也会返回 `request_id`

`request_id` 可用于：

* 前端展示与复制

* `/feedback` 进行反馈关联

* `/predict_progress/{request_id}`（内部调试接口，44000 默认不暴露）

## 6. GET /status

### 6.1 请求

```http
GET /status
```

### 6.2 响应

* `200 application/json`

字段说明：

* `status`：`online | loading | offline`

* `detail`：人类可读描述

* `model_loading_status`：`pending | loading | success | failed`

示例：

```json
{"status":"online","detail":"System ready","model_loading_status":"success"}
```

## 7. POST /predict（非流式）

### 7.1 请求

* `Content-Type: multipart/form-data`

表单字段：

* `image`：可选，单文件

* `ecg`：可选，多文件（必须包含一个 `.dat` 和一个 `.hea`）

### 7.2 响应

* `200 application/json`

返回体：

* `result`：模型完整输出（可能包含 `<think>...</think>` 与 `<answer>...</answer>` 标签）

* `request_id`

示例：

```json
{"result":"...","request_id":"20260201-..."}
```

### 7.3 调用示例（图片）

```bash
curl -sS -X POST http://<gpu-server-ip>:44000/predict \
  -F "image=@/path/to/ecg.png;type=image/png"
```

### 7.4 调用示例（信号 .dat/.hea）

```bash
curl -sS -X POST http://<gpu-server-ip>:44000/predict \
  -F "ecg=@/path/to/record.dat" \
  -F "ecg=@/path/to/record.hea"
```

## 8. POST /predict\_stream（流式 SSE）

### 8.1 请求

* `Content-Type: multipart/form-data`

* 字段同 `/predict`

### 8.2 响应

* `200 text/event-stream`

* 响应头包含：

  * `X-Request-ID: <request_id>`

  * `X-Accel-Buffering: no`（提示 Nginx 不要缓冲）

### 8.3 SSE 事件语义

服务端以 SSE 事件持续推送：

* `event: ready`：第一条事件，`data` 为 `{"request_id": "..."}`

* `event: content`：模型生成的普通内容 chunk

* `event: reasoning`：模型生成的 reasoning chunk（如果底层模型提供 `reasoning_content`）

* `event: ping`：心跳，用于长连接保活（data 中含时间戳）

* `event: done`：结束事件（data 为 `{"request_id": "..."}`）

* `event: error`：错误事件（data 为 `{"detail": "..."}`）

**data 格式说明**

* `content/reasoning` 事件的 `data` 是 JSON 字符串（服务端做了 `json.dumps(chunk)`），例如：

  * `data: "The "`

  * `data: "ECG "`

客户端接收时需要做 JSON 解析得到真实文本。

### 8.4 调用示例（图片）

```bash
curl -i -N http://<gpu-server-ip>:44000/predict_stream \
  -F "image=@/path/to/ecg.png;type=image/png"
```

### 8.5 Python 参考实现（requests 按行解析 SSE）

如需 Python 端严格按 SSE 解析，建议使用支持 SSE 的库；也可以手写按行解析 `event:` / `data:`。

最简伪代码（按行解析）：

```python
import json
import requests

resp = requests.post(
    "http://<gpu-server-ip>:44000/predict_stream",
    files={"image": open("ecg.png", "rb")},
    stream=True,
)
resp.raise_for_status()

event = None
for raw in resp.iter_lines(decode_unicode=True):
    if not raw:
        continue
    if raw.startswith("event:"):
        event = raw.split(":", 1)[1].strip()
    elif raw.startswith("data:"):
        data = raw.split(":", 1)[1].strip()
        payload = json.loads(data)
        if event == "content":
            print(payload, end="", flush=True)
```

## 9. POST /feedback（可选）

### 9.1 请求

* `Content-Type: application/json`

请求体：

* `request_id`：必填

* `feedback`：必填，`like | dislike`

* `comment`：可选字符串

示例：

```json
{"request_id":"20260201-...","feedback":"like","comment":"good"}
```

### 9.2 响应

* `200 application/json`

```json
{"status":"success","message":"Feedback recorded"}
```

## 10. 前端不在显卡服务器时：状态 badge 的推荐做法

页面不在显卡服务器部署时，建议把“系统在线状态”完全来源于显卡服务器 `GET /status`。

当前页面已支持用 meta 配置远程 API base：

* [index.html](file:///data/jinjiarui/run/ecg-r1-web/templates/index.html) 中：`<meta name="ecg-api-base" content="">`

如果页面部署在别的服务器：

* 将 `content` 设置为 `http://<gpu-server-ip>:44000`

* 前端会每 10 秒轮询 `/status` 并更新 `System Online/Loading/Offline`

## 11. 一键验证脚本

* 综合验证（/status + /predict + /predict\_stream）：[test\_remote\_api\_44000.sh](file:///data/jinjiarui/run/ecg-r1-web/scripts/test_remote_api_44000.sh)

* 仅验证流式（使用指定图片）：[test\_predict\_stream\_with\_image.sh](file:///data/jinjiarui/run/ecg-r1-web/scripts/test_predict_stream_with_image.sh)

示例：

```bash
bash scripts/test_remote_api_44000.sh http://127.0.0.1:44000
bash scripts/test_predict_stream_with_image.sh http://127.0.0.1:44000 /data/jinjiarui/run/ecg-r1-web/scripts/47099212.png
```

## 12. 常见问题（Troubleshooting）

* 浏览器/脚本感觉“不流式”：

  * 请确认请求的是 `/predict_stream` 而不是 `/predict`

  * 响应头应包含 `Content-Type: text/event-stream`

  * 终端中如果用了 `head` 截断，会导致 curl 退出码 23（属于预期）

* 请求经常超时：

  * 模型首轮推理可能较慢，脚本可调大 `TIMEOUT_PREDICT` / `TIMEOUT_STREAM`

* 44000 能打开页面：

  * 说明 Nginx 未做 API 白名单或未对 `/` 返回 404，需要检查 44000 server block 配置

可关闭该策略：`GEO_OVERRIDE_TW_AS_CN=0`。
