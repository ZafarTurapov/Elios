import os
def _env_key():
    return os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
def _env_secret():
    return os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or ""

def alpaca_headers(key: str | None = None,
                   secret: str | None = None,
                   accept_json: bool = True,
                   content_json: bool = False) -> dict:
    k = _env_key() if key is None else key
    s = _env_secret() if secret is None else secret
    h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
    if accept_json:  h["Accept"] = "application/json"
    if content_json: h["Content-Type"] = "application/json"
    return h
