from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean_error_outputs(path: Path) -> int:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    removed = 0
    for cell in notebook.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        kept_outputs = []
        for output in outputs:
            if output.get("output_type") == "error":
                removed += 1
            else:
                kept_outputs.append(output)
        cell["outputs"] = kept_outputs
    if removed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove error outputs from Jupyter notebooks.")
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()

    for notebook_path in args.notebooks:
        removed = clean_error_outputs(notebook_path)
        print(f"{notebook_path}: removed {removed} error output(s)")


if __name__ == "__main__":
    main()
