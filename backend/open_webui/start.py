# start.py
import os, sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6666))
    uvicorn.run(
        "open_webui.main:app",
        host="0.0.0.0",
        port=port,
        forwarded_allow_ips="*",
        reload=False,
        app_dir="/home/santi/python/open-webui-jeecgboot/open-webui/backend",
    )
