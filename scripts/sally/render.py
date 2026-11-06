import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
name, source, out_prefix, look = args

crab = bpy.data.objects[name]
if source.endswith(".glb"):
    bpy.data.objects.remove(crab)
    bpy.ops.import_scene.gltf(filepath=source)
    crab = next(o for o in bpy.context.selected_objects if o.type == "MESH")
else:
    for mod in crab.modifiers:
        mod.show_render = False
        mod.show_viewport = False
for obj in list(bpy.data.objects):
    if obj != crab:
        bpy.data.objects.remove(obj)

scene = bpy.context.scene
world = bpy.data.worlds.new("neutral")
world.color = (0.05, 0.05, 0.05)
scene.world = world
if look == "clay":
    material = bpy.data.materials.new("clay")
    material.diffuse_color = (0.6, 0.6, 0.6, 1.0)
    bpy.context.view_layer.material_override = material

points = [crab.matrix_world @ v.co for v in crab.data.vertices]
low = Vector(min(p[i] for p in points) for i in range(3))
high = Vector(max(p[i] for p in points) for i in range(3))
centre = (low + high) / 2
radius = (high - low).length / 2

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
scene.collection.objects.link(sun)

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
camera.data.lens = 50
scene.collection.objects.link(camera)
scene.camera = camera

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.seed = 0
scene.render.resolution_x = 960
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

for view, (azimuth, elevation) in {"quarter": (35, 30), "front": (0, 10), "top": (0, 89)}.items():
    a, e = math.radians(azimuth), math.radians(elevation)
    direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    camera.location = centre + direction * radius * 2.8
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out_prefix}-{view}.png"
    bpy.ops.render.render(write_still=True)
