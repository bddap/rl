import json
import math
import re
import sys
from dataclasses import dataclass, field

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

UP = Vector((0, 0, 1))
ANTERIOR = Vector((0, -1, 0))
INSTANCES = {"cheliped": ["pincer"], "walking leg": ["leg_01", "leg_02", "leg_03", "leg_04"]}


@dataclass
class Joint:
    name: str
    plane: str
    extent: float | None
    span: tuple[float, float] | None


@dataclass
class Bone:
    name: str
    parent: int | None
    segment: str = ""
    joint: Joint | None = None
    head: Vector = field(default_factory=Vector)
    tail: Vector = field(default_factory=Vector)
    axis: Vector = field(default_factory=lambda: UP.copy())
    limits: tuple[float, float] = (0.0, 0.0)


def rows(lines: list[str], heading: str) -> list[list[str]]:
    out: list[list[str]] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out[2:]


def parse(path: str) -> tuple[dict[str, float], dict[str, list[tuple[str, list[str]]]], dict[tuple[str, str], Joint]]:
    lines = open(path).read().splitlines()
    constants = {r[0].strip("`"): float(r[1]) for r in rows(lines, "## Constants")}
    segments: dict[str, list[tuple[str, list[str]]]] = {}
    for appendage, segment, regions in rows(lines, "## Segments"):
        segments.setdefault(appendage, []).append((segment, re.findall(r"`([^`]+)`", regions)))
    joints: dict[tuple[str, str], Joint] = {}
    for appendage, name, segment, plane, span, _ in rows(lines, "## Joints"):
        lo, sep, hi = span.partition("..")
        joints[(appendage, segment)] = Joint(
            name, plane, None if sep else float(span), (float(lo), float(hi)) if sep else None
        )
    return constants, segments, joints


def hull_of(points: list[Vector]) -> tuple[list[Vector], float]:
    bm = bmesh.new()
    for p in points:
        bm.verts.new(p)
    bmesh.ops.convex_hull(bm, input=list(bm.verts))
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    hull = [v.co.copy() for v in bm.verts]
    volume = bm.calc_volume()
    bm.free()
    return hull, volume


def signed_angle(parent: Vector, child: Vector, axis: Vector) -> float:
    p = parent - axis * parent.dot(axis)
    c = child - axis * child.dot(axis)
    return math.degrees(math.atan2(p.cross(c).dot(axis), p.dot(c)))


def main() -> None:
    table_path, name, out, report_path = sys.argv[sys.argv.index("--") + 1 :]
    constants, segments, joints = parse(table_path)

    crab = bpy.data.objects[name]
    crab.parent = None
    crab.matrix_world = Matrix.Identity(4)
    for mod in list(crab.modifiers):
        crab.modifiers.remove(mod)
    for obj in list(bpy.data.objects):
        if obj != crab:
            bpy.data.objects.remove(obj)

    mesh = crab.data
    n = len(mesh.vertices)
    pos = np.empty(n * 3)
    mesh.vertices.foreach_get("co", pos)
    pos = pos.reshape(-1, 3)
    groups = {g.index: g.name for g in crab.vertex_groups}
    region = [groups[max(v.groups, key=lambda g: g.weight).group] if v.groups else "" for v in mesh.vertices]

    shell = np.array([r.startswith("Def_shell.") for r in region])
    carapace_width = float(np.ptp(pos[shell, 0]))
    scale = constants["carapace_width_m"] / carapace_width
    pos *= scale

    bones = [Bone("body", None)]
    owner = np.zeros(n, dtype=np.int64)
    chains: list[list[int]] = []
    for appendage, prefixes in INSTANCES.items():
        for prefix in prefixes:
            for side in ("L", "R"):
                chain: list[int] = []
                for segment, regions in segments[appendage]:
                    names = {f"Def_{prefix}.{r}.{side}" for r in regions}
                    mask = np.array([r in names for r in region])
                    assert mask.any(), (prefix, side, segment)
                    label = prefix.replace("pincer", "cheliped")
                    bones.append(
                        Bone(f"{label}.{segment}.{side}", chain[-1] if chain else 0, segment, joints[(appendage, segment)])
                    )
                    chain.append(len(bones) - 1)
                    owner[mask] = len(bones) - 1
                chains.append(chain)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    edges = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
    bm.free()

    def seam(a: int, b: int) -> Vector:
        ends = owner[edges]
        crossing = edges[((ends[:, 0] == a) & (ends[:, 1] == b)) | ((ends[:, 0] == b) & (ends[:, 1] == a))]
        assert len(crossing), (bones[a].name, bones[b].name)
        return Vector(pos[crossing.ravel()].mean(axis=0))

    def farthest(points: np.ndarray, origin: Vector, direction: Vector | None = None) -> Vector:
        offsets = points - np.array(origin)
        score = offsets @ np.array(direction) if direction is not None else np.linalg.norm(offsets, axis=1)
        return Vector(points[np.argmax(score)])

    for chain in chains:
        for b in chain:
            bones[b].head = seam(bones[b].parent, b)
        for here, there in zip(chain, chain[1:]):
            bones[here].tail = bones[there].head
        tip = bones[chain[-1]]
        tip.tail = farthest(pos[owner == chain[-1]], tip.head)

    joint_report: list[dict[str, object]] = []
    report: dict[str, object] = {"scale": scale, "carapace_width_mesh_units": carapace_width, "joints": joint_report}
    for chain in chains:
        lateral = bones[chain[2]].tail - bones[chain[0]].head
        lateral.z = 0
        lateral.normalize()
        anterior = UP.cross(lateral)
        if anterior.dot(ANTERIOR) < 0:
            anterior = -anterior
        for b in chain:
            bone = bones[b]
            joint = bone.joint
            assert joint is not None
            child = (bone.tail - bone.head).normalized()
            if bone.parent == 0:
                parent = Vector((math.copysign(1, lateral.x), 0, 0))
            else:
                up = bones[bone.parent]
                parent = (up.tail - up.head).normalized()
            transverse = lateral.cross(UP).normalized()
            frontal = child.cross(anterior).normalized()
            if joint.plane == "transverse":
                axis = transverse
            elif joint.plane == "frontal":
                axis = frontal
            elif joint.plane == "oblique":
                axis = (transverse + frontal).normalized()
            elif joint.plane == "chela":
                fixed = farthest(pos[owner == bone.parent], bone.head, child) - bone.head
                axis = fixed.cross(child).normalized()
                parent = fixed.normalized()
            else:
                raise ValueError(joint.plane)
            rest = signed_angle(parent, child, axis)
            if joint.span is not None:
                lo, hi = joint.span
            else:
                assert joint.extent is not None
                lo, hi = rest - joint.extent / 2, rest + joint.extent / 2
            bone.axis = axis
            bone.limits = (lo - rest, hi - rest)
            joint_report.append(
                {"bone": bone.name, "joint": joint.name, "rest_deg": round(rest, 1),
                 "range_deg": [round(lo, 1), round(hi, 1)], "rest_in_range": lo <= rest <= hi}
            )

    out_of_range = [j["bone"] for j in joint_report if not j["rest_in_range"]]
    assert not out_of_range, f"rest pose outside the anatomical range: {out_of_range}"

    body = bones[0]
    body_pts = pos[owner == 0]
    body.head = Vector(body_pts.mean(axis=0))
    body.tail = body.head + ANTERIOR * float(np.ptp(body_pts[:, 1])) / 2

    mesh.vertices.foreach_set("co", pos.ravel().astype(np.float32))
    mesh.update()

    armature = bpy.data.armatures.new("sally")
    rig = bpy.data.objects.new("sally", armature)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit: list[bpy.types.EditBone] = []
    for bone in bones:
        eb = armature.edit_bones.new(bone.name)
        eb.head, eb.tail = bone.head, bone.tail
        eb.align_roll(bone.axis)
        if bone.parent is not None:
            eb.parent = edit[bone.parent]
        edit.append(eb)
    bpy.ops.object.mode_set(mode="OBJECT")

    hulls: list[tuple[list[Vector], float]] = []
    for i, bone in enumerate(bones):
        to_bone = armature.bones[bone.name].matrix_local.inverted()
        hulls.append(hull_of([to_bone @ Vector(p) for p in pos[owner == i]]))
    density = constants["mass_kg"] / sum(volume for _, volume in hulls)
    for bone, (hull, volume) in zip(bones, hulls):
        data = armature.bones[bone.name]
        extras = {"hull": [round(c, 6) for v in hull for c in v], "density": density}
        if bone.parent is not None:
            axis = (data.matrix_local.to_3x3().inverted() @ bone.axis).normalized()
            extras["axis"] = [round(c, 6) for c in axis]
            extras["limits_deg"] = [round(x, 2) for x in bone.limits]
        data["rl"] = extras

    for g in list(crab.vertex_groups):
        crab.vertex_groups.remove(g)
    vgroups = [crab.vertex_groups.new(name=b.name) for b in bones]
    weights = np.zeros((n, len(bones)))
    weights[np.arange(n), owner] = 1.0
    rings = int(constants["skin_blend_rings"])
    band = np.zeros(n, dtype=bool)
    band[edges[owner[edges[:, 0]] != owner[edges[:, 1]]].ravel()] = True
    for _ in range(rings):
        grown = band.copy()
        grown[edges[band[edges[:, 0]], 1]] = True
        grown[edges[band[edges[:, 1]], 0]] = True
        band = grown
    degree = np.ones(n)
    np.add.at(degree, edges.ravel(), 1)
    for _ in range(rings):
        acc = weights.copy()
        np.add.at(acc, edges[:, 0], weights[edges[:, 1]])
        np.add.at(acc, edges[:, 1], weights[edges[:, 0]])
        weights[band] = acc[band] / degree[band, None]
    for v in range(n):
        top = np.argsort(weights[v])[::-1][:4]
        top = top[weights[v, top] > 1e-3]
        total = weights[v, top].sum()
        for b in top:
            vgroups[b].add([v], float(weights[v, b] / total), "REPLACE")

    crab.parent = rig
    crab.modifiers.new("skeleton", "ARMATURE").object = rig
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj in (crab, rig))
    bpy.ops.export_scene.gltf(
        filepath=out,
        export_format="GLB",
        use_selection=True,
        export_apply=False,
        export_skins=True,
        export_animations=False,
        export_morph=False,
        export_cameras=False,
        export_lights=False,
        export_extras=True,
        export_yup=True,
    )
    report["bones"] = len(bones)
    report["density_kg_m3"] = round(density, 1)
    report["hull_volume_m3"] = {b.name: v for b, (_, v) in zip(bones, hulls)}
    report["vertices"] = {b.name: int((owner == i).sum()) for i, b in enumerate(bones)}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print("SALLY_RIG " + json.dumps({k: report[k] for k in ("scale", "bones", "density_kg_m3")}))


main()
