import json
import math
import sys

import bmesh
import bpy
from mathutils import Quaternion, Vector

# Degrees from rest by "[<side>:]<appendage>.<segment>", `*` standing for every walking
# leg; positive is dorsal, anterior or opening as in skeleton.md. Each pose and its
# view mirror a photographed posture of a live crab.
POSES: dict[str, tuple[str, dict[str, float]]] = {
    "stand": ("front", {"cheliped.basi-ischium": 10}),
    "stride": (
        "quarter",
        {
            "L:leg_01.basi-ischium": 15,
            "L:leg_03.basi-ischium": 15,
            "R:leg_02.basi-ischium": 15,
            "R:leg_04.basi-ischium": 15,
            "L:leg_01.coxa": 12,
            "L:leg_03.coxa": 12,
            "R:leg_02.coxa": 12,
            "R:leg_04.coxa": 12,
            "L:leg_02.coxa": -12,
            "L:leg_04.coxa": -12,
            "R:leg_01.coxa": -12,
            "R:leg_03.coxa": -12,
            "L:*.carpus": 10,
            "R:*.carpus": -25,
            "cheliped.basi-ischium": 10,
        },
    ),
    "crouch": (
        "front",
        {
            "*.basi-ischium": 30,
            "*.carpus": -60,
            "*.dactylus": -40,
            "cheliped.basi-ischium": -10,
            "cheliped.carpus": -60,
            "cheliped.coxa": 20,
        },
    ),
    "feed": (
        "side",
        {
            "L:cheliped.basi-ischium": 65,
            "L:cheliped.merus": 10,
            "L:cheliped.carpus": 20,
            "L:cheliped.propodus": 30,
            "L:cheliped.dactylus": 30,
            "*.basi-ischium": 15,
            "*.carpus": -40,
        },
    ),
}

VIEWS = {
    "quarter": (35, 30),
    "front": (0, 8),
    "side": (90, 8),
}


def angle_for(pose: dict[str, float], bone: str) -> float | None:
    if bone.count(".") != 2:
        return None
    appendage, segment, side = bone.split(".")
    keys = [f"{side}:{appendage}.{segment}", f"{appendage}.{segment}"]
    if appendage.startswith("leg_"):
        keys += [f"{side}:*.{segment}", f"*.{segment}"]
    for key in keys:
        if key in pose:
            return pose[key]
    return None


def main() -> None:
    rig_glb, out_prefix, look = sys.argv[sys.argv.index("--") + 1 :]
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    bpy.ops.import_scene.gltf(filepath=rig_glb)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    crab = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig)
    for obj in list(bpy.data.objects):
        if obj not in (rig, crab):
            bpy.data.objects.remove(obj)

    if look == "hulls":
        material = bpy.data.materials.new("hull")
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.35, 0.08, 1.0)
        for bone in rig.data.bones:
            rl = bone["rl"]
            pts = list(rl["hull"])
            verts = [Vector(pts[i : i + 3]) for i in range(0, len(pts), 3)]
            me = bpy.data.meshes.new(bone.name)
            me.from_pydata(verts, [], [])
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.convex_hull(bm, input=bm.verts)
            bm.to_mesh(me)
            bm.free()
            me.materials.append(material)
            obj = bpy.data.objects.new(f"hull.{bone.name}", me)
            bpy.context.scene.collection.objects.link(obj)
            obj.parent = rig
            obj.parent_type = "BONE"
            obj.parent_bone = bone.name
            obj.matrix_world = rig.matrix_world @ bone.matrix_local
        crab.hide_render = True

    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    world = bpy.data.worlds.new("neutral")
    world.color = (0.05, 0.05, 0.05)
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    scene.collection.objects.link(sun)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.lens = 50
    camera.data.clip_start = 0.001
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.seed = 0
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "PNG"

    report: dict[str, dict[str, float]] = {}
    for name, (view, pose) in POSES.items():
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion()
            angle = angle_for(pose, pb.name)
            if angle is None:
                continue
            rl = pb.bone["rl"]
            lo, hi = rl["limits_deg"]
            assert lo <= angle <= hi, (name, pb.name, angle, (lo, hi))
            pb.rotation_quaternion = Quaternion(Vector(rl["axis"]), math.radians(angle))
        rig.location = (0, 0, 0)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = crab.evaluated_get(depsgraph)
        points = [crab.matrix_world @ v.co for v in evaluated.data.vertices]
        low = Vector(min(p[i] for p in points) for i in range(3))
        high = Vector(max(p[i] for p in points) for i in range(3))
        rig.location.z = -low.z
        low.z, high.z = 0.0, high.z - low.z
        report[name] = {"lift_m": round(rig.location.z, 5)}
        centre = (low + high) / 2
        radius = (high - low).length / 2
        azimuth, elevation = VIEWS[view]
        a, e = math.radians(azimuth), math.radians(elevation)
        direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        camera.location = centre + direction * radius * 2.6
        camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{out_prefix}-{name}.png"
        bpy.ops.render.render(write_still=True)
    print("SALLY_POSE " + json.dumps(report))


main()
