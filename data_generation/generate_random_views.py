import argparse
import numpy as np
import os
from auto_pose.meshrenderer.pysixd import transform as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_list_file", type=str, required=True)
    parser.add_argument("--views_per_model", type=int, default=1600)
    parser.add_argument("--output_path", type=str,
                        default="models_with_poses.txt")
    arguments = parser.parse_args()

    translation = 1500.0

    eulers = []
    for i in range(arguments.views_per_model):
        eulers.append(T.euler_from_matrix(T.random_rotation_matrix()))

    model_list = np.loadtxt(arguments.model_list_file,
                            delimiter=' ', dtype=str)
    with open(arguments.output_path, 'w') as views_file:
        for line in model_list:
            model_path = "models/" + line[0] + "/" + line[1] + ".ply"
            model_name = line[0]
            model_stage = line[1]
            model_path = os.path.join(arguments.dataset_path, model_path)
            for i, euler in enumerate(eulers):
                views_file.write(f"{model_path} {model_name}{model_stage} "
                                 f"{i:04d} 0.0 0.0 {translation} "
                                 f"{euler[0]} {euler[1]} {euler[2]}\n")


if __name__ == '__main__':
    main()
