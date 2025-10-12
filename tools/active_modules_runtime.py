from modulefinder import ModuleFinder
from pathlib import Path
import importlib.metadata as imd
import sys

ROOT = Path("/root/stockbot")
ENTRY_FILES = [
    ROOT/"core/trading/signal_engine.py",
    ROOT/"core/trading/trade_executor.py",
    ROOT/"core/trading/sell_engine.py",
    ROOT/"core/trading/positions_sync.py",
]
ENTRY_FILES = [p for p in ENTRY_FILES if p.exists()]

def is_internal(name:str)->bool:
    return name.startswith("core.") or name.startswith("tools.")

def top_level(name:str)->str:
    return name.split(".")[0] if name else ""

internal = {}
external = set()

for script in ENTRY_FILES:
    f = ModuleFinder()
    f.run_script(str(script))  # статический разбор импортов
    for name, mod in f.modules.items():
        if name == "__main__":
            continue
        if is_internal(name):
            # путь к файлу может быть None для namespace; отфильтруем
            p = Path(mod.__file__) if getattr(mod, "__file__", None) else None
            internal[name] = p
        else:
            external.add(top_level(name))

# внешние: отделим pip-дистрибуции от stdlib
pkg_map = {}
try:
    pkg_map = imd.packages_distributions()  # top-level -> [dist,...]
except Exception:
    pkg_map = {}

third_party = {}
stdlib_guess = set()
for name in sorted(external):
    if not name or name in {"builtins", "__future__"}:
        continue
    dists = pkg_map.get(name)
    if dists:
        vers = []
        for d in dists:
            try: vers.append(f"{d}=={imd.version(d)}")
            except Exception: vers.append(d)
        third_party[name] = ", ".join(sorted(set(vers)))
    else:
        stdlib_guess.add(name)

print("# Elios — активные Python-модули (ModuleFinder от entrypoints)\n")
print("## Entry points")
for p in ENTRY_FILES:
    print("-", (p.relative_to(ROOT)))

print("\n## Внутренние модули проекта (с путями)")
for name in sorted(internal.keys()):
    p = internal[name]
    rel = str(p.relative_to(ROOT)) if (p and str(p).startswith(str(ROOT))) else (str(p) if p else "—")
    print(f"- {name}  ·  {rel}")

print("\n## Внешние зависимости (pip-дистрибуции)")
for name in sorted(third_party.keys()):
    print(f"- {name}  ·  {third_party[name]}")

print("\n## Вероятно stdlib/встроенные")
for name in sorted(stdlib_guess):
    print(f"- {name}")
