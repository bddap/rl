import json
import sys

paths = sys.argv[sys.argv.index("--") + 1 :]
*evals, out = paths

def load(path: str) -> dict:
    with open(path) as f:
        lines = [l for l in f if l.startswith("SALLY_EVAL ")]
    if len(lines) != 1:
        sys.exit(f"{path}: expected one SALLY_EVAL line, found {len(lines)}")
    return json.loads(lines[0].removeprefix("SALLY_EVAL "))


def cell(task: dict) -> str:
    if len(task["episodes"]) == 1:
        return f"{task['mean']:.4g}"
    scores = [e["score"] for e in task["episodes"]]
    return f"{task['mean']:.4g} ({min(scores):.4g} to {max(scores):.4g})"


reports = [load(p) for p in evals]
names = [r["policy"] for r in reports]
tasks = [t["task"] for t in reports[0]["tasks"]]
for r in reports:
    if [t["task"] for t in r["tasks"]] != tasks:
        sys.exit(f"{r['policy']}: scored other tasks than {names[0]}")
rows = ["| task | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
for i, task in enumerate(tasks):
    cells = [cell(r["tasks"][i]) for r in reports]
    rows.append(f"| `{task}` | " + " | ".join(cells) + " |")
with open(out, "w") as f:
    f.write("\n".join(rows) + "\n")
