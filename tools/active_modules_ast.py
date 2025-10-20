import ast
import importlib.metadata as imd
from pathlib import Path

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


# индекс файлов проекта
all_py = [p for base in PKG_ROOTS if base.exists() for p in base.rglob("*.py")]
mod_by_file = {p: file_to_mod(p) for p in all_py if file_to_mod(p)}
file_by_mod = {m: f for f, m in mod_by_file.items()}


def resolve_relative(base_mod: str, level: int, name: str | None) -> str:
    """Преобразуем from ..foo import bar относительно base_mod в абсолютный модуль."""
    if level <= 0:
        return name or ""
    parts = base_mod.split(".")
    # отрезаем level сегментов от конца
    if level <= len(parts):
        parent = parts[:-level]
    else:
        parent = []
    return ".".join([*parent, *(name.split(".") if name else [])]).strip(".")


def parse_imports(p: Path, src_mod: str) -> set[str]:
    try:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(p))
    except Exception:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            # import a, b.c → берём полные имена модулей (без атрибутов)
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            # from X import y,z  → добавляем ТОЛЬКО базовый модуль X (без .y)
            base = n.module or ""
            if n.level and src_mod:
                base = resolve_relative(src_mod, n.level, base)
            if base:
                out.add(base)
            else:
                # редкий случай "from . import y": добавляем родителя
                if src_mod:
                    parent = ".".join(src_mod.split(".")[:-1])
                    if parent:
                        out.add(parent)
    return out


def is_internal(m: str) -> bool:
    return m.startswith("core.") or m.startswith("tools.")


def top_level(m: str) -> str:
    return m.split(".")[0] if m else ""


# строим граф импортов (только имена модулей)
imports_by_mod = {}
for p, m in mod_by_file.items():
    imports_by_mod[m] = parse_imports(p, m)

# обход из точек входа — только достижимые внутренние
entry_mods = [file_to_mod(p) for p in ENTRY_FILES if p.exists()]
visited = set()
stack = [m for m in entry_mods if m]
while stack:
    cur = stack.pop()
    if cur in visited:
        continue
    visited.add(cur)
    for dep in imports_by_mod.get(cur, ()):
        if is_internal(dep):
            stack.append(dep)

# внешние зависимости из reachable
ext_used = set()
for m in visited:
    for dep in imports_by_mod.get(m, ()):
        if not is_internal(dep):
            ext_used.add(top_level(dep))

# pip-дистрибуции vs stdlib
try:
    pkg_map = imd.packages_distributions()  # top-level -> [dist,...]
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
        for d in dists:
            try:
                vers.append(f"{d}=={imd.version(d)}")
            except Exception:
                vers.append(d)
        third_party.append((name, ", ".join(sorted(set(vers)))))
    else:
        stdlib_guess.append(name)


def rel(m: str) -> str:
    p = file_by_mod.get(m)
    try:
        return str(p.relative_to(ROOT)) if p else "—"
    except Exception:
        return str(p) if p else "—"


print("# Elios — активные Python-модули (AST, с относит. импортами)\n")
print("## Entry points")
for m in entry_mods:
    print(f"- {m}")
print(f"\n## Внутренние модули проекта (reachable) · {len(visited)} шт.")
for m in sorted(visited):
    print(f"- {m}  ·  {rel(m)}")
print(f"\n## Внешние зависимости (pip) · {len(third_party)} шт.")
for name, ver in third_party:
    print(f"- {name}  ·  {ver}")
print(f"\n## Возможный stdlib/встроенные · {len(stdlib_guess)} шт.")
for name in sorted(stdlib_guess):
    print(f"- {name}")
