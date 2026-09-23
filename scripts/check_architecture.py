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
    dependencies: dict[str, set[str]] = defaultdict(set)
    issues: list[str] = []
    for owner, file in modules.items():
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        package = owner.split(".", 1)[0]
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module:
                    candidates = [f"{package}.{node.module.split('.')[0]}"]
                elif node.level == 0 and node.module:
                    candidates = [".".join(node.module.split(".")[:2])]
            elif isinstance(node, ast.Import):
                candidates = [".".join(alias.name.split(".")[:2]) for alias in node.names]
            for target in candidates:
                if owner.startswith("app.") and target.startswith("scripts."):
                    issues.append(f"{owner} imports experiment code {target}")
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
    print("Architecture check OK: no import cycles or core-to-UI dependencies.")
