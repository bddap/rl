import json
import sys

import bpy
import numpy as np

shots, out = sys.argv[sys.argv.index("--") + 1 :]


def load(path: str) -> np.ndarray:
    image = bpy.data.images.load(path)
    w, h = image.size
    pixels = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(h, w, 4)[:, :, :3]


rows, report = [], {}
for look in ("clay", "own"):
    for view in ("quarter", "front", "top"):
        blend = load(f"{shots}/blend-{look}-{view}.png")
        glb = load(f"{shots}/glb-{look}-{view}.png")
        diff = np.abs(blend - glb)
        report[f"{look}-{view}"] = {
            "mean_abs_diff": round(float(diff.mean()), 5),
            "pixels_diff_over_0.1": round(float((diff.max(axis=2) > 0.1).mean()), 5),
            "background_stdev": round(float(blend[:40, :40].std()), 5),
        }
        rows.append(np.concatenate([blend, glb, np.clip(diff * 4, 0, 1)], axis=1))
sheet = np.concatenate(rows[::-1], axis=0)
h, w, _ = sheet.shape
image = bpy.data.images.new("sheet", w, h)
image.pixels.foreach_set(np.concatenate([sheet, np.ones((h, w, 1), np.float32)], axis=2).ravel())
image.filepath_raw = out
image.file_format = "PNG"
image.save()
print("SALLY_COMPARE " + json.dumps(report))
for view in ("quarter", "front", "top"):
    assert report[f"clay-{view}"]["pixels_diff_over_0.1"] == 0, view
