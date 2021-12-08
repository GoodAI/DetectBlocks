import bpy
import numpy as np
import os
from mathutils import Euler


def main():
    input_path = "/home/tkachuk/dlr/master-thesis/detect_blocks/data_generation/models_with_poses.txt"
    output_dir = "/home/tkachuk/dlr/master-thesis/detect_blocks/data_generation/tex"

    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    camera = bpy.data.objects['Camera']
    camera.location = (0.0, 0.0, 0.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.clip_end = 5000

    light = bpy.data.objects['Light']
    light.location = (1.0, -4.0, -6.0)

    views_txt = np.loadtxt(input_path, delimiter=' ', dtype=str)
    model_paths = views_txt[:, 0]
    model_poses = np.array(views_txt[:, 1:], dtype=np.float32)

    for i, model_path in enumerate(model_paths):
        model_path = model_path.replace("_obj.ply", ".obj")
        bpy.ops.import_scene.obj(filepath=model_path)
        camera.location.z = model_poses[i][2]
        euler = Euler((model_poses[i][3], model_poses[i][4], model_poses[i][5]))
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.rotation_euler.rotate(euler)
        bpy.context.scene.render.filepath = os.path.join(output_dir,
                                                         f"{i:06d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.ops.object.delete()


if __name__ == '__main__':
    main()
