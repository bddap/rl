import json
import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

# Renders an episode `sally-eval --record` wrote: the rig's own skinned mesh, every
# bone posed where its physics body was each tick. glTF's Y-up model space maps to
# Blender's Z-up as the glTF importer maps it.
GLTF_TO_BLENDER = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
TICKS_PER_FRAME = 3
CHECKER_M = 0.02


def body_matrix(p: list[float]) -> Matrix:
    x, y, z, qx, qy, qz, qw = p
    return Matrix.Translation((x, y, z)) @ Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()


def main() -> None:
    rig_glb, recording, out, label = sys.argv[sys.argv.index("--") + 1 :]
    with open(recording) as f:
        rec = json.load(f)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    bpy.ops.import_scene.gltf(filepath=rig_glb)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    crab = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig)
    for obj in list(bpy.data.objects):
        if obj not in (rig, crab):
            bpy.data.objects.remove(obj)

    names = rec["bones"]
    rest = [body_matrix(p) for p in rec["rest"]]
    to_arm = rig.matrix_world.inverted() @ GLTF_TO_BLENDER
    from_arm = GLTF_TO_BLENDER.inverted() @ rig.matrix_world
    bones = [rig.pose.bones[n] for n in names]
    index = {n: i for i, n in enumerate(names)}
    for pb in bones:
        pb.rotation_mode = "QUATERNION"

    shown = rec["poses"][::TICKS_PER_FRAME]
    keys: list[list[float]] = [[] for _ in bones]
    for f, frame in enumerate(shown):
        # Each body's displacement from its rest in model space, as an armature-space
        # transform; a bone's basis is that relative to its parent's, about its rest.
        moved = [to_arm @ (body_matrix(p) @ r.inverted()) @ from_arm for p, r in zip(frame, rest)]
        for i, pb in enumerate(bones):
            d = moved[i]
            parent = pb.parent
            while parent is not None and parent.name not in index:
                parent = parent.parent
            if parent is not None:
                d = moved[index[parent.name]].inverted() @ d
            r = pb.bone.matrix_local
            loc, rot, _ = (r.inverted() @ d @ r).decompose()
            keys[i].append([*loc, *rot])
    rig.animation_data_create()
    action = bpy.data.actions.new("episode")
    rig.animation_data.action = action
    for pb, values in zip(bones, keys):
        for channel, (path, index_in) in enumerate(
            [("location", k) for k in range(3)] + [("rotation_quaternion", k) for k in range(4)]
        ):
            curve = action.fcurve_ensure_for_datablock(rig, f'pose.bones["{pb.name}"].{path}', index=index_in)
            curve.keyframe_points.add(len(values))
            co = [x for f, v in enumerate(values) for x in (f, v[channel])]
            curve.keyframe_points.foreach_set("co", co)
            curve.keyframe_points.foreach_set("interpolation", [0] * len(values))
            curve.update()

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, len(shown) - 1
    scene.render.fps = rec["hz"]
    scene.render.fps_base = TICKS_PER_FRAME

    bpy.ops.mesh.primitive_plane_add(size=4.0)
    ground = bpy.context.active_object
    mat = bpy.data.materials.new("ground")
    nodes = mat.node_tree.nodes
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 4.0 / CHECKER_M
    checker.inputs["Color1"].default_value = (0.55, 0.55, 0.52, 1)
    checker.inputs["Color2"].default_value = (0.35, 0.35, 0.33, 1)
    mat.node_tree.links.new(checker.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    ground.data.materials.append(mat)

    world = bpy.data.worlds.new("sky")
    world.color = (0.6, 0.65, 0.7)
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(35), 0, math.radians(30))
    scene.collection.objects.link(sun)

    # The camera follows the carapace's path averaged over a second, so a walking crab
    # stays in frame and the checker ground shows how far it went.
    root = [GLTF_TO_BLENDER @ Vector(frame[0][:3]) for frame in shown]
    half = max(1, round(rec["hz"] / TICKS_PER_FRAME / 2))
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.lens = 35
    camera.data.clip_start = 0.001
    scene.collection.objects.link(camera)
    scene.camera = camera
    a, e = math.radians(20), math.radians(30)
    direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    for f in range(len(shown)):
        near = root[max(0, f - half) : f + half + 1]
        centre = sum(near, Vector()) / len(near)
        centre.z = 0.02
        camera.location = centre + direction * 0.32
        camera.keyframe_insert("location", frame=f)

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.render.use_persistent_data = True
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.use_stamp = True
    for field in scene.render.bl_rna.properties.keys():
        if field.startswith("use_stamp_") and field not in ("use_stamp_note", "use_stamp_frame", "use_stamp_labels"):
            setattr(scene.render, field, False)
    scene.render.use_stamp_note = True
    scene.render.stamp_note_text = label
    scene.render.stamp_font_size = 20
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = out
    bpy.ops.render.render(animation=True)


main()
