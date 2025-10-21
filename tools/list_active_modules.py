import ast
from pathlib import Path
import importlib.metadata as imd

ROOT = Path("/root/stockbot")
PKG_ROOTS = [ROOT / "core", ROOT / "tools"]

ENTRY_FILES = [
    ROOT / "core/trading/signal_engine.py",
    ROOT / "core/trading/trade_executor.py",
    ROOT / "core/trading/sell_engine.py",
    ROOT / "core/trading/positions_sync.py",
]
ENTRY_FILES = [p for p in ENTRY_FILES if p.exists()]


def file_to_mod(p: Path) -> str | None:
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or not parts[-1].endswith(".py"):
        return None
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_imports(p: Path) -> set[str]:
    try:
        src = p.read_text(encoding="utf-8")
    except Exception:
        return set()
    try:
        tree = ast.parse(src, filename=str(p))
    except Exception:
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if base:
                for a in n.names:
                    nm = a.name
                    out.add(base if nm == "*" else f"{base}.{nm}")
            else:
                for a in n.names:
                    out.add(a.name)
    return out


def is_internal(m: str) -> bool:
    return m.startswith("core.") or m.startswith("tools.")


def top_level(m: str) -> str:
    return m.split(".")[0] if m else ""


# Индексация проекта
all_py = [p for base in PKG_ROOTS if base.exists() for p in base.rglob("*.py")]
mod_by_file = {p: file_to_mod(p) for p in all_py if file_to_mod(p)}
file_by_mod = {m: f for f, m in mod_by_file.items()}
imports_by_file = {p: parse_imports(p) for p in mod_by_file}

# Граф внутренних импортов и внешний список
internal_graph: dict[str, set[str]] = {mod_by_file[p]: set() for p in mod_by_file}
for p, imps in imports_by_file.items():
    src_mod = mod_by_file[p]
    for m in imps:
        if is_internal(m) or m.startswith("core.") or m.startswith("tools."):
            internal_graph[src_mod].add(m)

# DFS от точек входа
entry_mods = [file_to_mod(p) for p in ENTRY_FILES if file_to_mod(p)]
visited: set[str] = set()
stack = [m for m in entry_mods if m]
while stack:
    cur = stack.pop()
    if cur in visited:
        continue
    visited.add(cur)
    for nxt in internal_graph.get(cur, ()):
        stack.append(nxt)

# Внешние импорты только из reachable-модулей
ext_used: set[str] = set()
for p, imps in imports_by_file.items():
    src_mod = mod_by_file[p]
    if src_mod not in visited:
        continue
    for m in imps:
        if not is_internal(m):
            ext_used.add(top_level(m))

# Привязка к pip-дистрибуциям
try:
    pkg_map = imd.packages_distributions()  # {top: [dist,...]}
except Exception:
    pkg_map = {}

third_party = []
stdlib_guess = []
for name in sorted(ext_used):
    if not name or name in {"__future__", "builtins"}:
        continue
    dists = pkg_map.get(name)
    if dists:
        vers = []
        for d in sorted(set(dists)):
            try:
                vers.append(f"{d}=={imd.version(d)}")
            except Exception:
                vers.append(d)
        third_party.append((name, ", ".join(vers)))
    else:
        stdlib_guess.append(name)

# Вывод
print("# Elios — активные Python-модули (по графу импортов от entrypoints)")
print()
print("## Entry points")
for m in entry_mods:
    print(f"- {m}")
print()
print(f"## Внутренние модули проекта (reachable) · {len(visited)} шт.")
for m in sorted(visited):
    path = file_by_mod.get(m)
    pstr = str(path.relative_to(ROOT)) if path else "?"
    print(f"- {m}  ·  {pstr}")
print()
print(f"## Внешние зависимости (pip-дистрибуции) · {len(third_party)} шт.")
for name, ver in third_party:
    print(f"- {name}  ·  {ver}")
print()
print(f"## Возможный stdlib/встроенные (без pip-дистрибуции) · {len(stdlib_guess)} шт.")
for name in stdlib_guess:
    print(f"- {name}")
