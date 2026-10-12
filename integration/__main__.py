"""Точка входа: python -m integration"""
import uvicorn

try:
    from integration.service.main import app
except ImportError:
    from service.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
