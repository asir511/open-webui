import os
import uvicorn

if __name__ == "__main__":
    # 设置端口，默认为 8080
    port = int(os.environ.get("PORT", 8080))

    # 运行 Uvicorn 服务器
    uvicorn.run(
        "open_webui.main:app",
        host="0.0.0.0",
        port=port,
        forwarded_allow_ips="*",
        reload=True
    )