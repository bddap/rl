#!/usr/bin/env python3
"""Extract GCR_CHORDS from net/src/controls.rs into combos.json.

Parsed from the real source rather than hand-copied so the mock can't drift from
the registry it claims to present.
"""
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()

# Isolate the GCR_CHORDS const body.
m = re.search(r"GCR_CHORDS[^=]*=\s*ChordRegistry::new\(&\[(.*?)\n\]\);", src, re.S)
if not m:
    sys.exit("GCR_CHORDS not found")
body = m.group(1)

DIR = {"Up": "U", "Down": "D", "Left": "L", "Right": "R"}
entries = []
for em in re.finditer(
    r"ChordEntry\s*\{\s*code:\s*(.*?),\s*action:\s*Action::(\w+),\s*label:\s*\"([^\"]*)\"", body, re.S
):
    code_src, action, label = em.groups()
    if "QUIT_CODE" in code_src:
        code = "UUDDLR"  # crab_world::chord::QUIT_CODE
    else:
        code = "".join(DIR[d] for d in re.findall(r"ChordDir::(\w+)", code_src))
    entries.append({"code": code, "action": action, "label": label})

if len(entries) < 20:
    sys.exit(f"only {len(entries)} entries parsed — parser broke")
json.dump(entries, open(Path(__file__).parent / "combos.json", "w"), indent=1)
print(f"{len(entries)} entries")
