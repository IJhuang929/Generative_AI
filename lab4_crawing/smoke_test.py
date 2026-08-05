"""
Smoke test: simulate TA grading flow, crawl only 2025-01-01 ~ 2025-01-02.
All outputs go under smoke_output/ so existing files are untouched.
"""
import glob
import os
import subprocess
import sys

SMOKE_DIR   = "smoke_output"
ANSWER_DIR  = "2025_answer"
START, END  = "0101", "0102"
KEYWORD     = "台股"
EXCLUDE     = {"eval.py", "smoke_test.py"}

# 2025-01-01 00:00 CST ~ 2025-01-02 00:00 CST
SMOKE_START_TS = 1735660800
SMOKE_END_TS   = 1735747200


def find_student_script():
    scripts = [f for f in glob.glob("*.py") if f not in EXCLUDE]
    if not scripts:
        sys.exit("[FAIL] No student script found (*.py excluding eval.py, smoke_test.py)")
    if len(scripts) > 1:
        print(f"[WARN] Multiple scripts found: {scripts}, using {scripts[0]}")
    return scripts[0]


def step(label):
    print(f"\n{'='*60}")
    print(label)
    print('='*60)


def run_eval(answer_dir, output_dir):
    step(f"[Step 4] python eval.py {answer_dir} {output_dir}")
    result = subprocess.run(
        ["python", "eval.py", answer_dir, output_dir],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        sys.exit(f"[FAIL] eval.py exited with code {result.returncode}")
    print("[OK] exit 0")


def main():
    origin = os.path.abspath(".")
    script = find_student_script()
    smoke_abs = os.path.join(origin, SMOKE_DIR)

    print(f"Student script : {script}")
    print(f"Smoke directory: {SMOKE_DIR}/")
    print(f"Crawl range    : 2025-01-01 ~ 2025-01-02 (smoke only)")

    # Step 1 — create smoke_output/ and chdir into it
    step(f"[Step 1] Creating {SMOKE_DIR}/ and importing student script")
    os.makedirs(smoke_abs, exist_ok=True)
    os.chdir(smoke_abs)
    sys.path.insert(0, origin)
    import importlib
    mod = importlib.import_module(script[:-3])   # strip .py

    # Patch timestamps and paths for smoke test
    mod.START_TS      = SMOKE_START_TS
    mod.END_TS        = SMOKE_END_TS
    mod.OUTPUT_DIR    = ""                 # write directly to cwd (smoke_output/)
    mod.ARTICLES_PATH = "articles.jsonl"
    mod.POPULAR_PATH  = "popular_articles.jsonl"
    print(f"  Patched START_TS={SMOKE_START_TS}, END_TS={SMOKE_END_TS}")
    print(f"  Patched OUTPUT_DIR='' (files land directly in {SMOKE_DIR}/)")
    print(f"  Patched ARTICLES_PATH=articles.jsonl, POPULAR_PATH=popular_articles.jsonl")

    # Step 2 — crawl
    step(f"[Step 2] crawl  (cwd: {SMOKE_DIR}/)")
    mod.crawl()

    if not os.path.exists("articles.jsonl"):
        sys.exit("[FAIL] articles.jsonl not found after crawl")
    count = sum(1 for _ in open("articles.jsonl"))
    print(f"  articles.jsonl: {count} articles")

    # Step 3 — push / popular / keyword
    step(f"[Step 3a] push {START} {END}  (cwd: {SMOKE_DIR}/)")
    mod.push(START, END)

    step(f"[Step 3b] popular {START} {END}  (cwd: {SMOKE_DIR}/)")
    mod.popular(START, END)

    step(f"[Step 3c] keyword {START} {END} {KEYWORD}  (cwd: {SMOKE_DIR}/)")
    mod.keyword(START, END, KEYWORD)

    # Step 4 — eval (back to origin so eval.py / 2025_answer are reachable)
    os.chdir(origin)
    run_eval(ANSWER_DIR, os.path.join(SMOKE_DIR))


if __name__ == "__main__":
    main()
