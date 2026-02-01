# 前端服务器复用指南（对接 GPU 推理端口 44000）

本文档面向“前端服务器/业务服务器”的工程师，说明如何复用本仓库的前端页面（HTML/CSS/JS），并将推理请求转发到显卡服务器的 **44000** 端口，同时正确透传真实用户 IP 与可选的地理信息/额外元数据。

## 1. 总体思路（推荐架构）

**前端服务器**负责：
- 托管前端页面静态资源（HTML/CSS/JS）
- 作为反向代理，把浏览器的推理请求转发到 **显卡服务器 :44000**
- 透传用户 IP（以及可选地理信息）到显卡服务器，供统计与审计

**显卡服务器**负责：
- 运行模型推理服务（FastAPI 内部 8000，通过 Nginx 暴露 44000 为 API-only）
- 返回推理结果（支持 SSE 流式）

这样做的好处：
- 避免浏览器直接跨域访问显卡服务器（减少 CORS 复杂度）
- 显卡服务器 44000 可限制为“仅允许前端服务器访问”，防止伪造 `X-Forwarded-For` 等头污染统计

## 2. 复用哪些前端文件

直接复用以下资源即可：
- 页面模板：`templates/index.html`
- 静态资源：`static/style.css`、`static/responsive.css`、`static/script.js` 以及 `static/icons/*`

如果你的前端服务器不使用 Jinja2：
- 只要把 `index.html` 中的 `{{ model_display_name }}` 替换成固定字符串（或删掉该字段）即可。

## 3. 关键配置：API_BASE（无需改 JS 的方式）

本仓库前端已支持通过 meta 配置 API Base（用于把 `/status`、`/predict_stream`、`/predict`、`/feedback` 全部改为走远程/反代）。

在 `index.html` `<head>` 中有：
```html
<meta name="ecg-api-base" content="">
```

你只需要把 `content` 设置为你的 API 前缀：

### 3.1 推荐：走前端服务器同域反代（强烈推荐）
```html
<meta name="ecg-api-base" content="/ecg_api">
```

浏览器请求会变成：
- `GET /ecg_api/status`
- `POST /ecg_api/predict_stream`
- `POST /ecg_api/predict`
- `POST /ecg_api/feedback`

### 3.2 不推荐：浏览器直连显卡服务器（需要 CORS）
```html
<meta name="ecg-api-base" content="http://<gpu-server-ip>:44000">
```

此方案需要显卡服务器配置好 CORS，且仍要考虑 44000 的访问控制。

## 4. 前端服务器 Nginx 反代配置（带 IP 透传）

示例（前端服务器 Nginx）：
```nginx
location /ecg_api/ {
    proxy_pass http://<gpu-server-ip>:44000/;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_buffering off;
}
```

说明：
- `X-Forwarded-For` 会保留链路：`client, proxy1, proxy2...`
- 显卡服务器会优先使用 `CF-Connecting-IP / True-Client-IP / X-Real-IP`，否则解析 `X-Forwarded-For`

可信代理设置（显卡服务器侧）：
- 服务端只在请求来源属于 `TRUSTED_PROXY_CIDRS` 时才信任上述头部，默认仅信任本机 Nginx：`127.0.0.1/32,::1/128`。
- 如果你的前端服务器在另一台机器上，需要把它的出口 IP/网段加入该列表，例如：
  - `TRUSTED_PROXY_CIDRS=127.0.0.1/32,::1/128,<frontend-public-ip>/32`

如何确定“前端服务器出口 IP/网段”：
- 同一 VPC/内网直连：通常填写前端服务器的内网网段（例如 `10.0.0.0/16`），以前端访问 44000 时的对端 IP 为准。
- 公网访问：填写前端服务器的公网出口 IP（建议固定 EIP/固定 NAT 出口）。
- 前端前面还有 CDN/网关：以显卡服务器实际看到的上游对端 IP 为准，必要时在显卡服务器日志确认再加入白名单。

## 5. 可选：透传地理信息（用于 Analytics 显示到“地区”级）

如果你的前端服务器/网关能获得地理信息（例如来自 CDN、网关、内网定位服务），建议透传：
- `X-Geo-Country`：国家码（例如 CN）
- `X-Geo-Region`：地区/省州（例如 Beijing）
- `X-Geo-City`：城市（例如 Beijing）

如果没有这些信息，显卡服务器的 Analytics 会对公网 IP 做一次地理解析（有缓存）；内网 IP/127.0.0.1 仍可能显示 Unknown（属正常）。

## 6. 推理请求格式（浏览器端保持不变）

推理接口使用 `multipart/form-data`，字段与现有页面保持一致：
- 图片：`image`（png/jpg）
- 信号：`ecg`（多文件，必须同时提交 `.dat` 与 `.hea`，字段名都叫 `ecg`）

流式推理：
```bash
curl -i -N http://<frontend-domain>/ecg_api/predict_stream \
  -F "image=@/path/to/ecg.png;type=image/png"
```

非流式推理：
```bash
curl -sS http://<frontend-domain>/ecg_api/predict \
  -F "image=@/path/to/ecg.png;type=image/png"
```

## 7. 额外信息（元数据）如何传给显卡服务器

如果你希望在显卡服务器侧记录额外字段（例如 `hospital_id`、`user_id`、`source_app`、`trace_id` 等），建议做法：
- 浏览器端：在 `FormData` 里追加额外字段（或在 `<form>` 中加入隐藏 `<input>`）
- 前端服务器：直接转发该 `multipart/form-data`
- 显卡服务器：后端需要显式读取并写入记录（如有需要可再加白名单字段）

目前后端主要处理 `image/ecg` 文件字段；如你需要“把额外字段也写入 data.json”，需要后端增加读取逻辑。

## 8. 安全建议：只允许前端服务器访问显卡服务器 44000

原因：
- 如果 44000 对公网开放，客户端可以伪造 `X-Forwarded-For` 等头污染 IP 统计与审计

推荐策略：
- 云安全组/防火墙：显卡服务器 `TCP 44000` 只放行前端服务器 IP/网段
- 或显卡服务器 Nginx 44000 server block 做 `allow/deny`

## 9. 自测

在前端服务器上：
- `curl http://127.0.0.1/ecg_api/status` 应返回 online JSON
- 用浏览器上传一条推理请求，确认：
  - 推理正常
  - `X-Forwarded-For` 链路正确（可在显卡服务器 data collection 的记录里看到 client.ip）
