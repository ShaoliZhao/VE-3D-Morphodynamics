"""Fail on common release mistakes without third-party dependencies."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "SUBMISSION_TEXT.md",
    "docs/method.md",
    "docs/experiments.md",
    "references/reading.md",
    "examples/make_synthetic.py",
    "tests/test_core.py",
]


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".toml", ".yml"}
    )
    if re.search(r"[\u4e00-\u9fff]", text):
        raise SystemExit("Chinese text found; public repository is English-only")
    if any(ROOT.rglob("*.h5ad")):
        raise SystemExit("Challenge data detected")
    print("Repository audit passed.")


if __name__ == "__main__":
    main()
