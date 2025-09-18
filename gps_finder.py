import threading
import webbrowser
import time
import requests 
import pandas as pd 
import xml.etree.ElementTree as ET
from geopy.geocoders import Nominatim 
from flask import Flask, request, jsonify, Response

def get_location_from_address(address: str):
    '''
    이 함수는 환자의 도로명 주소 또는 인근 지하철역 정보를 통해, 사용자의 위치 정보를 가져온다.
    '''
    geolocator = Nominatim(user_agent="gps_finder")
    location = geolocator.geocode(address)
    return location.latitude, location.longitude


def get_hosp_df(patient_lon: float, patient_lat: float, hospital_num: int):
    '''
    이 함수는 환자의 위도와 경도, 요청한 병원 수의 정보를 dataframe으로 변경해준다. 
    '''
    url = 'http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytLcinfoInqire'
    params = {
        'serviceKey': '8efa490558160d4f08647fcc311b86d88f7da4ec5e5cd4bc1886a607bd0b2793',
        'WGS84_LON': patient_lon,
        'WGS84_LAT': patient_lat,
        # 'pageNo': '1',
        'numOfRows': hospital_num
    }
    hospital_response = requests.get(url, params=params)

    xml_root = ET.fromstring(hospital_response.content)
    hops_items = xml_root.findall(".//item")

    hops_data = []
    for hops in hops_items:
        hops_data.append({
            "병원명": hops.find("dutyName").text,
            "주소": hops.find("dutyAddr").text,
            "전화번호": hops.find("dutyTel1").text,
            "거리(km)": hops.find("distance").text,
            "운영시간": f"{hops.find('startTime').text} ~ {hops.find('endTime').text}"
        })
        
    hops_df = pd.DataFrame(hops_data)
    return hops_df


# ================== JS 연동 ==================
app = Flask(__name__)
_coords_event = threading.Event()
_coords_holder = {"lat": None, "lon": None}

ASK_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>현재 위치 요청</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 2rem; }
  .card { max-width: 540px; margin: 0 auto; padding: 1.5rem; border: 1px solid #ddd; border-radius: 12px; }
  button { padding: .8rem 1rem; border: 0; border-radius: 8px; cursor: pointer; }
</style>
</head>
<body>
  <div class="card">
    <h1>현재 위치 사용 동의</h1>
    <p>가까운 병원을 찾기 위해 브라우저의 위치 권한이 필요합니다.</p>
    <button id="btn">위치 제공하기</button>
    <button id="btn-decline" style="margin-left:10px; background:#ccc;">위치 정보 제공하지 않기</button>
    <p id="status"></p>
  </div>
<script>
const btn = document.getElementById('btn');
const statusEl = document.getElementById('status');
const declineBtn = document.getElementById('btn-decline');
declineBtn.addEventListener('click', () => {
  statusEl.textContent = '사용자가 위치 정보를 제공하지 않기로 선택했습니다.';
  fetch('/location', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({lat: null, lon: null, declined: true})
  });
});

function postCoords(lat, lon) {
  fetch('/location', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({lat: lat, lon: lon})
  }).then(res => res.json())
    .then(() => {
      statusEl.textContent = '좌표 전송 완료! 이 창은 닫아도 됩니다.';
    }).catch(err => {
      statusEl.textContent = '전송 실패: ' + err;
    });
}

btn.addEventListener('click', () => {
  statusEl.textContent = '위치 확인 중...';
  if (!navigator.geolocation) {
    statusEl.textContent = '이 브라우저는 위치 정보를 지원하지 않습니다.';
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      statusEl.textContent = `위치 획득: ${lat.toFixed(6)}, ${lon.toFixed(6)} (전송 중)`;
      postCoords(lat, lon);
    },
    (err) => {
      statusEl.textContent = '위치 획득 실패: ' + err.message + ' — 주소 입력으로 대체하세요.';
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );
});
</script>
</body>
</html>
"""

@app.get("/")
def root():
    # 간단 안내 페이지
    return Response('<meta http-equiv="refresh" content="0; url=/ask" />', mimetype="text/html")

@app.get("/ask")
def ask_location():
    # 위치 권한 요청 페이지
    return Response(ASK_PAGE, mimetype="text/html")

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

def run_server():
    # Flask 개발 서버 실행 (localhost는 Geolocation 허용)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

def get_coords_via_browser(timeout=30):
    """
    브라우저에서 JS Geolocation으로 좌표를 받아오는 동기 함수.
    timeout(초) 안에 못 받으면 None 반환.
    """
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    # 브라우저 열기
    time.sleep(0.5)  # 서버 기동 대기
    webbrowser.open("http://127.0.0.1:5000/ask", new=1, autoraise=True)
    # 좌표 수신 대기
    got = _coords_event.wait(timeout=timeout)
    if not got:
        return None, None
    return _coords_holder["lat"], _coords_holder["lon"]

# ================== main code ==================
if __name__ == "__main__":
    print("브라우저가 열립니다. 위치 권한을 허용해 주세요.")
    lat, lon = get_coords_via_browser(timeout=45)

    if lat is None or lon is None:
        print("\n[안내] 브라우저에서 위치를 받지 못했습니다.")
        use_addr = input("대신 도로명 주소 또는 근처 역 정보를 입력하세요 (예: 영동대로57길 28, 학여울역):\n> ").strip()
        lat, lon = get_location_from_address(use_addr)

    print(f"\n[사용자 좌표] 위도: {lat}, 경도: {lon}")
    try:
        hops_df = get_hosp_df(patient_lon=lon, patient_lat=lat, hospital_num=10)
        # 콘솔 출력
        pd.set_option("display.max_colwidth", None)
        print("\n[가까운 병원 목록]")
        print(hops_df)
        # 필요하면 CSV 저장
        hops_df.to_csv("nearby_hospitals.csv", index=False, encoding="utf-8-sig")
        print("\nCSV로 저장됨: nearby_hospitals.csv")
    except Exception as e:
        print(f"병원 조회 중 오류: {e}")
