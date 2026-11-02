"""Check tracked content before publishing. Does not print file contents."""
from pathlib import Path
import re, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT = {"README.md", ".gitignore", "pyproject.toml"}
ALLOWED_DIRS = {"src", "tests", "docs", "configs", "scripts"}
ALLOWED_SUFFIXES = {".py", ".md", ".toml"}
PATTERNS = {
    "local_home": r"/(?:Users|home)/(?!path/|example/)[A-Za-z0-9_-]+/",
    "experiment_server": r"douslab|jl57095|overleaf\.com/project/",
    "credential": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)",
    "patient_identifier": r"\bS\d{2}\b",
}

def check():
    query = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True)
    if query.returncode != 0:
        raise SystemExit("Initialize and stage the intended files first.")
    paths = [Path(p) for p in query.stdout.decode().split("\0") if p]
    if not paths:
        raise SystemExit("No staged/tracked files to check.")
    errors = []
    for rel in paths:
        p = ROOT / rel
        if p.is_symlink():
            errors.append((str(rel), "symlink")); continue
        if str(rel) not in ALLOWED_ROOT:
            if rel.parts[0] not in ALLOWED_DIRS or rel.suffix not in ALLOWED_SUFFIXES:
                errors.append((str(rel), "file type/path outside allowlist")); continue
        if p.stat().st_size > 100000:
            errors.append((str(rel), "unexpected large text file"))
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append((str(rel), "binary/non-UTF8")); continue
        # This checker contains the signatures it checks, not actual credentials.
        if rel == Path("scripts/check_repository.py"):
            continue
        for name, pattern in PATTERNS.items():
            if re.search(pattern, text):
                errors.append((str(rel), name))
    if errors:
        for path, reason in errors:
            print(path + ": " + reason)
        raise SystemExit(1)
    print(f"PASS: {len(paths)} allowlisted code/protocol/documentation files; no data/result artifacts or sensitive path signatures.")

if __name__ == "__main__":
    check()
