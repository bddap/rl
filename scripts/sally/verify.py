import json
import struct
import sys
from typing import Any

import bpy
import numpy as np

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
Gltf = dict[str, Any]


def read_glb(path: str) -> tuple[Gltf, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    magic, _, _ = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF", path
    json_len, json_type = struct.unpack_from("<I4s", data, 12)
    assert json_type == b"JSON"
    doc = json.loads(data[20 : 20 + json_len])
    off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<I4s", data, off)
    assert bin_type == b"BIN\0"
    return doc, data[off + 8 : off + 8 + bin_len]


def accessor(doc: Gltf, blob: bytes, idx: int) -> np.ndarray:
    acc = doc["accessors"][idx]
    assert "sparse" not in acc and not acc.get("normalized"), acc
    view = doc["bufferViews"][acc["bufferView"]]
    n = COMPONENTS[acc["type"]]
    dtype = np.dtype(DTYPES[acc["componentType"]])
    stride = view.get("byteStride", n * dtype.itemsize)
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    rows = np.ndarray(
        (acc["count"], n), dtype=dtype, buffer=blob, offset=start, strides=(stride, dtype.itemsize)
    )
    return rows.copy()


def canonical(tris: np.ndarray) -> np.ndarray:
    """Rotate each triangle to start at its smallest index, keeping its winding."""
    shift = tris.argmin(axis=1)
    rotated = np.stack([tris[np.arange(len(tris)), (shift + k) % 3] for k in range(3)], axis=1)
    return np.unique(rotated, axis=0)


name, glb = sys.argv[sys.argv.index("--") + 1 :]
mesh = bpy.data.objects[name].data
mesh.calc_loop_triangles()
blend_pos = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
mesh.vertices.foreach_get("co", blend_pos)
blend_pos = blend_pos.reshape(-1, 3)
blend_tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
mesh.loop_triangles.foreach_get("vertices", blend_tris)
blend_tris = blend_tris.reshape(-1, 3)
assert len(np.unique(blend_pos, axis=0)) == len(blend_pos), "coincident vertices make the position mapping ambiguous"
index_of = {tuple(p): i for i, p in enumerate(blend_pos.tolist())}

doc, blob = read_glb(glb)
assert len(doc["meshes"]) == 1, [m.get("name") for m in doc["meshes"]]
for node in doc["nodes"]:
    for key, identity in (("translation", [0, 0, 0]), ("rotation", [0, 0, 0, 1]), ("scale", [1, 1, 1])):
        assert node.get(key, identity) == identity, (node["name"], key, node[key])
    assert "matrix" not in node, node["name"]

primitives = doc["meshes"][0]["primitives"]
glb_tris = []
for prim in primitives:
    assert prim.get("mode", 4) == 4
    assert doc["accessors"][prim["attributes"]["POSITION"]]["componentType"] == 5126
    pos = accessor(doc, blob, prim["attributes"]["POSITION"])
    zup = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], axis=1)
    to_blend = np.array([index_of[tuple(p)] for p in zup.tolist()])
    glb_tris.append(to_blend[accessor(doc, blob, prim["indices"]).reshape(-1, 3)])
glb_tris = np.concatenate(glb_tris)

report = {
    "blend_vertices": len(blend_pos),
    "blend_triangles": len(blend_tris),
    "glb_vertices": len(np.unique(glb_tris)),
    "glb_vertices_split_at_seams": sum(doc["accessors"][p["attributes"]["POSITION"]]["count"] for p in primitives),
    "glb_triangles": len(glb_tris),
    "same_triangles_and_winding": bool(np.array_equal(canonical(blend_tris), canonical(glb_tris))),
}
print("SALLY_VERIFY " + json.dumps(report))
assert report["glb_vertices"] == report["blend_vertices"]
assert report["glb_triangles"] == report["blend_triangles"]
assert report["same_triangles_and_winding"]
