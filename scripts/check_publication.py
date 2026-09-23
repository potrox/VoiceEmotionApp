"""Fail-closed check of files that would enter a public Git repository.

This is a safety net, not a guarantee: review diffs before publishing and never
use ``git add -f`` to override the source-only .gitignore.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path


MAX_FILE_BYTES = 1_000_000
PUBLIC_MODEL_PATH = "models/model_v11_dusha_62k.pkl"
PUBLIC_MODEL_SHA256 = "8514637098edf946639e387d895b51ea478608ab49fa605fc3552d798c9c43d0"
PUBLIC_TOP_LEVEL = {
    ".gitignore",
    "README.md",
    "DATASET_GUIDE.md",
    "requirements.txt",
    "requirements-teacher.txt",
    "config.example.json",
    "main.py",
    "bootstrap.py",
    "install.py",
    "build.py",
    "run.bat",
    "install.bat",
    "build_exe.bat",
    ".githooks/pre-commit",
}
PUBLIC_SOURCE_FILES = {
    "app/__init__.py",
    "app/asr.py",
    "app/audio.py",
    "app/config.py",
    "app/constants.py",
    "app/dataset.py",
    "app/dataset_recovery.py",
    "app/dependencies.py",
    "app/embedding_features.py",
    "app/features.py",
    "app/gui.py",
    "app/gui_dataset.py",
    "app/gui_dialogs.py",
    "app/gui_home.py",
    "app/gui_recognition.py",
    "app/gui_recording.py",
    "app/gui_settings.py",
    "app/gui_shared.py",
    "app/gui_testing.py",
    "app/gui_training.py",
    "app/honest_models.py",
    "app/models.py",
    "app/recognizer.py",
    "app/recorder.py",
    "app/storage.py",
    "app/workers.py",
    "scripts/build_multimodal_bundle.py",
    "scripts/check_architecture.py",
    "scripts/check_publication.py",
    "scripts/evaluate_audio_teacher.py",
    "scripts/evaluate_auto_asr_pilot.py",
    "scripts/evaluate_balanced_multimodal_62k.py",
    "scripts/evaluate_emotion2vec_teacher.py",
    "scripts/evaluation_components.py",
    "scripts/experiment_common.py",
    "scripts/multimodal_components.py",
    "scripts/extract_emotion2vec_embeddings.py",
    "scripts/finalize_emotion2vec_model.py",
    "scripts/finalize_expanded_multimodal_fusion.py",
    "scripts/finalize_multimodal_fusion.py",
    "scripts/prepare_dusha_archive.py",
    "scripts/prepare_dusha_subset.py",
    "scripts/prepare_full_confident_manifest.py",
    "scripts/prepare_secondary_holdout.py",
    "scripts/run_context_hierarchy_pilot.py",
    "scripts/run_contrastive_pair_pilot.py",
    "scripts/run_dusha_experiment.py",
    "scripts/run_emotion2vec_embedding_pilot.py",
    "scripts/run_expanded_text_fusion.py",
    "scripts/run_specialist_experiment.py",
    "scripts/run_transitive_graph_pilot.py",
    "scripts/wait_for_embedding_evaluation.py",
    "tests/test_asr.py",
    "tests/test_architecture.py",
    "tests/test_experiment_components.py",
    "tests/test_honest_models.py",
    "tests/test_publication_guard.py",
}
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
KNOWN_TOKEN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})"
)
ASSIGNED_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|password|token|credential)\b\s*[:=]\s*"
    r"(?:['\"]([^'\"\r\n]{8,})['\"]|([^\s,#]{12,}))"
)
PRIVATE_PATH = re.compile(
    r"(?i)(?:[A-Z]:\\Users\\|/Users/|/home/)(?!\.\.\.|<|your|user)[A-Za-z0-9._-]+"
)
AUTH_URL = re.compile(r"(?i)\b(?:https?|postgres(?:ql)?|mongodb(?:\+srv)?)://[^\s/@:]+:[^\s/@]+@")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PLACEHOLDERS = {"example", "placeholder", "changeme", "your_token", "your_api_key"}


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def _repo_root() -> Path:
    return Path(_git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").decode().strip())


def _app_prefix(root: Path) -> str:
    app = Path(__file__).resolve().parents[1]
    relative = app.relative_to(root).as_posix()
    if relative == ".":
        return ""
    if relative != "VoiceEmotionApp_refactor":
        raise RuntimeError(f"Unexpected repository layout: {relative}")
    return relative + "/"


def allowed_path(path: str, prefix: str) -> bool:
    if prefix and path == ".gitignore":
        return True
    if not path.startswith(prefix):
        return False
    relative = path[len(prefix):]
    return relative in PUBLIC_TOP_LEVEL or relative in PUBLIC_SOURCE_FILES or relative == PUBLIC_MODEL_PATH


def released_model_problems(data: bytes) -> list[str]:
    if hashlib.sha256(data).hexdigest() != PUBLIC_MODEL_SHA256:
        return ["released model does not match the audited SHA-256"]
    return []


def content_problems(data: bytes) -> list[str]:
    if len(data) > MAX_FILE_BYTES:
        return [f"file exceeds {MAX_FILE_BYTES} bytes"]
    if b"\x00" in data:
        return ["binary file"]
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError:
        return ["file is not UTF-8 text"]
    problems = []
    for description, pattern in (
        ("private key", PRIVATE_KEY),
        ("known token format", KNOWN_TOKEN),
        ("personal home path", PRIVATE_PATH),
        ("credential in URL", AUTH_URL),
        ("email address", EMAIL),
    ):
        if pattern.search(value):
            problems.append(description)
    for match in ASSIGNED_SECRET.finditer(value):
        assigned = (match.group(1) or match.group(2)).strip().lower()
        if assigned not in PLACEHOLDERS and not assigned.startswith(("os.environ", "getenv(")):
            problems.append("assigned credential-like value")
            break
    return problems


def _paths(root: Path, staged: bool) -> list[str]:
    tracked = _git(root, "ls-files", "--cached", "-z").decode("utf-8").split("\0")
    if staged:
        return sorted(set(filter(None, tracked)))
    visible = _git(root, "ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    return sorted(set(filter(None, tracked + visible)))


def check(staged: bool) -> int:
    root = _repo_root()
    prefix = _app_prefix(root)
    problems = []
    paths = _paths(root, staged)
    for path in paths:
        if not allowed_path(path, prefix):
            problems.append(f"{path}: not on public source allowlist")
            continue
        if staged:
            try:
                data = _git(root, "show", f":{path}")
            except RuntimeError as exc:
                problems.append(f"{path}: unable to read Git index ({exc})")
                continue
        else:
            file = root / path
            if not file.is_file():
                continue
            data = file.read_bytes()
        validator = released_model_problems if path == prefix + PUBLIC_MODEL_PATH else content_problems
        problems.extend(f"{path}: {message}" for message in validator(data))
    if problems:
        print("Publication check FAILED:", file=sys.stderr)
        for message in problems:
            print(f"  - {message}", file=sys.stderr)
        print("Remove the file from the index or replace private content; do not bypass with git add -f.", file=sys.stderr)
        return 1
    print(f"Publication check OK: {len(paths)} source files checked ({'index' if staged else 'worktree'}).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="check all files in the Git index (pre-commit)")
    mode.add_argument("--worktree", action="store_true", help="check all public candidates on disk")
    args = parser.parse_args()
    try:
        sys.exit(check(staged=args.staged))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Publication check could not run: {exc}", file=sys.stderr)
        sys.exit(2)
