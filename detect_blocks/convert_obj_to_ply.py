import bpy
import json
import os

DATASET_PATH = '/path/to/dataset/root/dir'

with open(os.path.join(DATASET_PATH, 'metadata_files.txt')) as file:
    for i, line in enumerate(file):
        metadata_path = os.path.join(DATASET_PATH, line.strip())
        with open(metadata_path.replace('\\', '/')) as metadata_file:
            metadata = json.load(metadata_file)
            model_path = metadata['model_path'].replace('\\', '/')
            model_path = os.path.join(DATASET_PATH, model_path)
            model_path_ply = model_path.replace('.obj', '.ply')
            if os.path.isfile(model_path_ply):
                print('Skipping {}: file exists'.format(model_path_ply))
                continue
            bpy.ops.import_scene.obj(filepath=model_path)
            objs = bpy.data.objects
            if "Cube" in objs:
                objs.remove(objs["Cube"], do_unlink=True)
            bpy.ops.export_mesh.ply(filepath=model_path_ply, use_ascii=True)
            bpy.ops.object.delete()
