from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for sharded embeddings and run full CV")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.evaluation_args:
        raise ValueError("Evaluation command arguments are required after --.")
    # argparse.REMAINDER can preserve the separator itself. Passing it on to
    # the evaluation script would make every following option positional.
    evaluation_args = list(args.evaluation_args)
    if evaluation_args and evaluation_args[0] == "--":
        evaluation_args = evaluation_args[1:]
    if not evaluation_args:
        raise ValueError("Evaluation command arguments are required after --.")

    while True:
        count = sum(1 for _ in args.cache_dir.glob("*.npy"))
        print(f"Embedding cache: {count}/{args.expected}", flush=True)
        if count >= args.expected:
            break
        time.sleep(max(5, args.poll_seconds))
    time.sleep(5)
    script = Path(__file__).with_name("run_emotion2vec_embedding_pilot.py")
    command = [sys.executable, str(script), *evaluation_args]
    print("All embeddings ready; starting full nested CV.", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print("Full embedding evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
