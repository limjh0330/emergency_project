import requests
import pandas as pd
import xml.etree.ElementTree as ET
from geopy.geocoders import Nominatim
from web_bridge import get_coords_via_browser

def get_location_from_address(address: str):
    geolocator = Nominatim(user_agent="gps_finder")
    location = geolocator.geocode(address)
    if not location:
        raise ValueError("주소로 위치를 찾을 수 없습니다.")
    return location.latitude, location.longitude

def get_hosp_df(patient_lon: float, patient_lat: float, hospital_num: int):
    url = 'http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytLcinfoInqire'
    params = {
        'serviceKey': '8efa490558160d4f08647fcc311b86d88f7da4ec5e5cd4bc1886a607bd0b2793',
        'WGS84_LON': patient_lon,
        'WGS84_LAT': patient_lat,
        'numOfRows': hospital_num
    }
    hospital_response = requests.get(url, params=params, timeout=10)
    hospital_response.raise_for_status()

    xml_root = ET.fromstring(hospital_response.content)
    hops_items = xml_root.findall(".//item")

    hops_data = []
    for hops in hops_items:
        def _text(tag):
            node = hops.find(tag)
            return node.text if node is not None else ""
        hops_data.append({
            "병원명": _text("dutyName"),
            "주소": _text("dutyAddr"),
            "전화번호": _text("dutyTel1"),
            "거리(km)": _text("distance"),
            "운영시간": f"{_text('startTime')} ~ {_text('endTime')}"
        })
    return pd.DataFrame(hops_data)

if __name__ == "__main__":
    print("브라우저가 열립니다. 위치 권한을 허용하거나, 거부 시 주소를 입력하세요.")
    lat, lon = get_coords_via_browser(timeout=45)

    # 거부 또는 시간초과 시 주소 입력 대체
    if lat is None or lon is None:
        print("\n[안내] 브라우저에서 위치를 받지 못했습니다.")
        use_addr = input("대신 도로명 주소 또는 근처 역 정보를 입력하세요 (예: 영동대로57길 28, 학여울역):\n> ").strip()
        lat, lon = get_location_from_address(use_addr)

    print(f"\n[사용자 좌표] 위도: {lat}, 경도: {lon}")
    try:
        hops_df = get_hosp_df(patient_lon=lon, patient_lat=lat, hospital_num=10)
        pd.set_option("display.max_colwidth", None)
        print("\n[가까운 병원 목록]")
        print(hops_df)
        hops_df.to_csv("nearby_hospitals.csv", index=False, encoding="utf-8-sig")
        print("\nCSV로 저장됨: nearby_hospitals.csv")
    except Exception as e:
        print(f"병원 조회 중 오류: {e}")