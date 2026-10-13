"""Gate de sintaxe de um notebook: JSON válido, células de código compiláveis, sem saída gravada.

Uso: python scripts/check_notebook.py notebooks/arquivo.ipynb [--forbid torch ...]

Não executa nada. Magics (%) e comandos de shell (!) são ignorados na compilação.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys


def check(path: str, forbid: list[str]) -> list[str]:
    errors: list[str] = []
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    if nb.get("nbformat") != 4:
        errors.append(f"nbformat {nb.get('nbformat')!r}, esperado 4")
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if cell.get("outputs"):
            errors.append(f"célula {i}: tem saída gravada")
        clean = "\n".join(
            "pass" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.splitlines()
        )
        try:
            tree = ast.parse(clean)
        except SyntaxError as e:
            errors.append(f"célula {i}: erro de sintaxe na linha {e.lineno}: {e.msg}")
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in forbid:
                    errors.append(f"célula {i}: importa {n!r}, proibido neste notebook")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--forbid", nargs="*", default=["torch", "torch_geometric"])
    args = ap.parse_args()
    errs = check(args.notebook, args.forbid)
    for e in errs:
        print("ERRO", e)
    print(f"check_notebook: {len(errs)} erro(s) em {args.notebook}")
    sys.exit(1 if errs else 0)
