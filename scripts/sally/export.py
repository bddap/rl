import sys

import bpy

name, out = sys.argv[sys.argv.index("--") + 1 :]
crab = bpy.data.objects[name]
for obj in bpy.context.view_layer.objects:
    obj.select_set(obj == crab)
bpy.ops.export_scene.gltf(
    filepath=out,
    export_format="GLB",
    use_selection=True,
    export_apply=False,
    export_skins=False,
    export_animations=False,
    export_morph=False,
    export_cameras=False,
    export_lights=False,
    export_extras=False,
    export_yup=True,
)
