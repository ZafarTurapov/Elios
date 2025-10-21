import os
import requests

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

url = f"{BASE_URL}/v2/orders"

r = requests.get(url, headers=headers)

print("Ордера:")
print(r.status_code)
print(r.json())
