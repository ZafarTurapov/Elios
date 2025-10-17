import os

import requests

from core.utils.alpaca_headers import alpaca_headers

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

headers = alpaca_headers()

url = f"{BASE_URL}/v2/positions"

r = requests.get(url, headers=headers)

print("Текущие позиции:")
print(r.status_code)
print(r.json())
