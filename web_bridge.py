import threading
import time
import webbrowser
from pathlib import Path
from flask import Flask, request, jsonify, Response, render_template

# 내부 상태 공유
_coords_event = threading.Event()
_coords_holder = {"lat": None, "lon": None}

def create_app():
    """Flask 앱 생성 (템플릿/정적 파일 경로 셋업 포함)"""
    base_dir = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    @app.get("/")
    def root():
        return Response('<meta http-equiv="refresh" content="0; url=/ask" />', mimetype="text/html")

    @app.get("/ask")
    def ask_location():
        return render_template("ask.html")

    @app.post("/location")
    def receive_location():
        data = request.get_json(force=True, silent=True) or {}
        if data.get("declined"):
            print("사용자가 위치 정보를 제공하지 않음")
            _coords_event.set()
            return jsonify({"ok": True, "declined": True})

        lat = data.get("lat")
        lon = data.get("lon")
        # 간단 검증
        try:
            lat = float(lat); lon = float(lon)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("invalid range")
        except Exception:
            return jsonify({"ok": False, "error": "invalid coordinates"}), 400

        _coords_holder["lat"] = lat
        _coords_holder["lon"] = lon
        _coords_event.set()
        return jsonify({"ok": True})

    return app

def _run_server(app, host="127.0.0.1", port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)

def get_coords_via_browser(timeout=30, url="http://127.0.0.1:5000/ask"):
    """
    브라우저에서 JS Geolocation으로 좌표를 받아오는 동기 함수.
    timeout(초) 안에 못 받으면 (None, None) 반환.
    """
    app = create_app()
    server_thread = threading.Thread(target=_run_server, args=(app,), daemon=True)
    server_thread.start()

    time.sleep(0.5)  # 서버 기동 대기
    webbrowser.open(url, new=1, autoraise=True)

    got = _coords_event.wait(timeout=timeout)
    if not got:
        return None, None
    return _coords_holder["lat"], _coords_holder["lon"]