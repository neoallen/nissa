#!/usr/bin/env python3
"""
Watch Timer Web App — FastAPI server with ngrok tunnel.

Usage:
    uv run web_app.py              # Start with ngrok tunnel
    uv run web_app.py --no-ngrok   # Start on localhost only
"""

import os
import sys
import shutil
import tempfile
import time
import json
import subprocess
import traceback
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tg-python"))
import tg_timer

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Watch Timer")

HTML_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")
_html_cache = None


def get_html() -> str:
    global _html_cache
    if _html_cache is None:
        if not os.path.exists(HTML_PATH):
            return "<h2>Error: templates/index.html not found</h2>"
        with open(HTML_PATH, encoding="utf-8") as f:
            _html_cache = f.read()
    return _html_cache


@app.get("/", response_class=HTMLResponse)
async def index():
    return get_html()


@app.post("/api/analyze")
async def api_analyze(file: UploadFile, bph: int = Form(21600),
                      la: float = Form(52.0)):
    tmp_path = None
    wav_path = None
    try:
        suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

        try:
            samples, sr = tg_timer.read_audio(tmp_path)
        except FileNotFoundError:
            raise HTTPException(500, "ffmpeg/ffprobe not found. Required for audio decoding.")
        except Exception as e:
            raise HTTPException(400, f"Audio decode failed: {e}")

        if samples is None or len(samples) == 0:
            raise HTTPException(400, "No audio data found in recording.")

        if len(samples) < sr:
            raise HTTPException(400,
                f"Recording too short ({len(samples)/sr:.1f}s). Minimum 1 second.")

        result_obj = tg_timer.analyze_audio(samples, sr, bph, la,
                                   source_label=file.filename or "web_recording")

        if result_obj is None:
            raise HTTPException(400,
                "No valid measurements. Try a different BPH or ensure clear tick sounds.")

        return JSONResponse(content=result_obj)

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Internal analysis error.")
    finally:
        for p in (tmp_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("WARNING: ffmpeg/ffprobe not found in PATH.")
        print("  Audio decoding from non-WAV formats will fail.")
        print("  Install ffmpeg: https://ffmpeg.org/download.html")
        print()


def _print_qr(url: str):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    import qrcode
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make()
    qr.print_ascii()


def _find_ngrok() -> str | None:
    candidates = [
        "ngrok",
        "ngrok.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Microsoft", "WinGet", "Packages",
                     "Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe",
                     "ngrok.exe"),
    ]
    for p in candidates:
        if shutil.which(p):
            return p
    return None


def start_ngrok(port: int = 8000):
    ngrok_bin = _find_ngrok()
    if not ngrok_bin:
        print("ngrok not found. Install: winget install Ngrok.Ngrok")
        print("Or download from https://ngrok.com/download")
        return None

    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        subprocess.run([ngrok_bin, "config", "add-authtoken", auth_token],
                       capture_output=True, timeout=15)

    try:
        proc = subprocess.Popen(
            [ngrok_bin, "http", str(port), "--log=stdout"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Failed to start ngrok: {e}")
        return None

    for i in range(30):
        time.sleep(0.5)
        try:
            resp = urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2)
            body = resp.read()
            data = json.loads(body)
            tunnels = data.get("tunnels", [])
            if tunnels:
                url = tunnels[0].get("public_url", "")
                if url:
                    print(f"ngrok tunnel active: {url}")
                    print("Open this URL on your iPhone to access the app.")
                    print()
                    _print_qr(url)
                    return url
        except Exception:
            continue

    print("ngrok started but could not retrieve public URL.")
    print("Check http://localhost:4040/status manually.")
    return None


def main():
    from dotenv import load_dotenv
    load_dotenv()

    use_ngrok = "--no-ngrok" not in sys.argv
    port = 8000

    check_ffmpeg()

    ngrok_url = None
    if use_ngrok:
        ngrok_url = start_ngrok(port)

    print(f"Starting Watch Timer server at http://localhost:{port}")
    if not ngrok_url:
        print("No ngrok tunnel. Access from localhost only.")
    print()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
