import argparse
import bpy
import json
import os
import pathlib


def convert(metadata):
    dataset_path = pathlib.Path(metadata).parent
    with open(metadata) as file:
        for i, line in enumerate(file):
            metadata_path = os.path.join(dataset_path, line.strip())
            with open(metadata_path.replace("\\", "/")) as metadata_file:
                metadata = json.load(metadata_file)
                model_path = metadata["model_path"].replace("\\", "/")
                model_path = os.path.join(dataset_path, model_path)
                model_path_ply = model_path.replace(".obj", ".ply")
                if os.path.isfile(model_path_ply):
                    print("Skipping {}: file exists".format(model_path_ply))
                    continue
                bpy.ops.import_scene.obj(filepath=model_path)
                objs = bpy.data.objects
                if "Cube" in objs:
                    objs.remove(objs["Cube"], do_unlink=True)
                bpy.ops.export_mesh.ply(filepath=model_path_ply, use_ascii=True)
                bpy.ops.object.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert obj to ply")
    parser.add_argument("METADATA", type=lambda p: pathlib.Path(p).resolve())

    args = parser.parse_args()
    convert(args.METADATA)
