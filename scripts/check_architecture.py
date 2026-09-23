"""Check local module boundaries and import cycles before publication."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


def audit(root: Path) -> list[str]:
    modules = {
        f"{package}.{file.stem}": file
        for package in ("app", "scripts")
        for file in (root / package).glob("*.py")
    }
    server_root = root / "server" / "server_app"
    for file in server_root.rglob("*.py"):
        relative = file.relative_to(server_root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(["server_app", *parts])] = file
    dependencies: dict[str, set[str]] = defaultdict(set)
    issues: list[str] = []
    for owner, file in modules.items():
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = owner.split(".")[:-node.level]
                    target = ".".join([*base, *(node.module.split(".") if node.module else [])])
                else:
                    target = node.module or ""
                if target:
                    candidates.append(target)
                    candidates.extend(
                        f"{target}.{alias.name}" for alias in node.names
                        if alias.name != "*"
                    )
            elif isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            for target in candidates:
                if owner.startswith("app.") and target.startswith("scripts."):
                    issues.append(f"{owner} imports experiment code {target}")
                if owner.startswith("server_app.") and target.startswith(("app.", "scripts.")):
                    issues.append(f"server module {owner} imports desktop/experiment code {target}")
                if (
                    owner.startswith("app.")
                    and not owner.startswith("app.gui")
                    and target.startswith("app.gui")
                ):
                    issues.append(f"core module {owner} imports UI {target}")
                if target in modules and target != owner:
                    dependencies[owner].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, path: list[str]) -> None:
        if module in visiting:
            issues.append("import cycle: " + " -> ".join(path + [module]))
            return
        if module in visited:
            return
        visiting.add(module)
        for target in sorted(dependencies[module]):
            visit(target, path + [module])
        visiting.remove(module)
        visited.add(module)

    for module in sorted(modules):
        visit(module, [])
    return issues


if __name__ == "__main__":
    problems = audit(Path(__file__).resolve().parents[1])
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print("Architecture check OK: no import cycles or inverted desktop/server dependencies.")
