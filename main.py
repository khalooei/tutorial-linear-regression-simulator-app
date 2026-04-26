"""
khalooei_visualization_optimization — educational demo: sample space vs parameter space.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).resolve().parent
app = FastAPI(
    title="Khalooei — Visualization & Optimization",
    description="Sample space vs parameter space (linear regression, loss landscape).",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})
