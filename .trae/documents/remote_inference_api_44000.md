# 远程推理 API（Port 44000）设计说明

## 1. 目标
- 新增对外端口 **44000** 作为“远程推理服务入口”
- 保持现有 **80/8000** 的 Web 页面与本地访问不变
- 远程调用方请求格式与现有前端一致（multipart/form-data）
- 支持 **SSE 流式返回**

## 2. 实现方案
采用 **单实例 FastAPI（8000） + Nginx 多端口暴露**：
- FastAPI/uvicorn 仅监听 8000（只跑一套模型进程）
- Nginx 监听：
  - 80：Web 前端入口（页面 + API）
  - 44000：API-only（只暴露推理相关 API）

说明：这不是“错误地把 44000 代理到 8000”。
这里的关键是 **44000 在 Nginx 层做了 API 白名单**，并且 `/` 返回 404，因此不会出现前端页面。

## 3. 44000 端口白名单
- `GET /status`
- `POST /predict_stream`
- `POST /predict`
- `POST /feedback`（可选）
- 其它路径：404

## 4. CORS
- Nginx 在 44000 server block 统一添加 `Access-Control-Allow-*` 头，并对 OPTIONS 返回 204。
- FastAPI 启用 `CORSMiddleware`（不使用 cookies 场景），支持浏览器跨域调用。

## 5. 配置文件
- 系统生效：`/etc/nginx/conf.d/ecg_r1_web.conf`
- 仓库模板：`deploy/nginx/ecg_r1_web.conf`

## 6. 调用示例
```bash
curl -N -X POST http://<server-ip>:44000/predict_stream \
  -F "ecg=@record.dat" \
  -F "ecg=@record.hea"
```

## 7. 自检
- `curl -I http://127.0.0.1:44000/` -> 404
- `curl http://127.0.0.1:44000/status` -> online JSON
- `curl -i -X OPTIONS http://127.0.0.1:44000/predict_stream` -> 204 + CORS 头
