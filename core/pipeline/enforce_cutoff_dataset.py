from pathlib import Path
import pandas as pd

CUTOFF = "2012-01-01"
P = Path("/root/stockbot/core/data/train/dataset.parquet")

if not P.exists():
    print(f"[enforce_cutoff] dataset not found: {P}")
    raise SystemExit(2)

df = pd.read_parquet(P)
before = len(df)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
cut = pd.Timestamp(CUTOFF)
df = df[df["date"] >= cut].reset_index(drop=True)
after = len(df)
df.to_parquet(P, index=False)

share = (df["date"] >= cut).mean()
print(f"[enforce_cutoff] {CUTOFF}: {before} -> {after} | date_min={df['date'].min()} date_max={df['date'].max()} | share>={CUTOFF}={share:.3f}")
