import argparse
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from auto_pose.meshrenderer import meshrenderer_phong
from auto_pose.meshrenderer.pysixd import transform as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pose_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    arguments = parser.parse_args()

    antialiasing = 8
    vertex_scale = 1
    h = w = 512
    K = np.array([[1075.65, 0, 512/2], [0, 1073.90, 512/2], [0, 0, 1]])
    clip_near = 10
    clip_far = 10000

    df = pd.read_csv(arguments.model_pose_list, delimiter=" ", header=None, dtype=str)
    model_paths = list(df[0])
    scr_names = df[1] + "_" + df[2]
    model_poses = np.array(df.iloc[:, 3:], dtype=np.float32)
    obj_ids = list(set(model_paths))

    train_x_dir = os.path.join(arguments.output_dir, "opengl", "train_x")
    if not os.path.exists(train_x_dir):
        os.makedirs(train_x_dir)
    train_y_dir = os.path.join(arguments.output_dir, "opengl", "train_y")
    if not os.path.exists(train_y_dir):
        os.makedirs(train_y_dir)

    renderer = meshrenderer_phong.Renderer(
        obj_ids, antialiasing, vertex_scale=vertex_scale,
        vertex_tmp_store_folder="/tmp")

    for i, model_path in enumerate(tqdm(model_paths)):
        t = np.array([model_poses[i][0], model_poses[i][1], model_poses[i][2]])
        R = T.euler_matrix(
            model_poses[i][3] + np.pi, -model_poses[i][4], -model_poses[i][5])
        rendered_image, _ = renderer.render(obj_ids.index(model_path),
                                     w, h, K, R[:3, :3], t,
                                     near=clip_near, far=clip_far)
        mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
        x, y, width, height = cv2.boundingRect(mask)
        bbx_side = int(max(width, height) * 1.2)
        x += width // 2 - bbx_side // 2
        y += height // 2 - bbx_side // 2
        train_y = cv2.resize(rendered_image[
            max(0, y):min(y + bbx_side, rendered_image.shape[0]),
            max(0, x):min(x + bbx_side, rendered_image.shape[1])
        ], (128, 128))
        cv2.imwrite(os.path.join(train_y_dir, f"{scr_names[i]}.png"), train_y)

        zoom = np.random.uniform(0.7, 1.0)
        train_x = np.zeros((128, 128, 3))
        new_size = int(np.floor(128 * zoom))
        max_offset = 128 - new_size
        offsets = (np.random.uniform(size=(2,)) * max_offset).astype(int)
        train_x[offsets[0]:offsets[0]+new_size,
                offsets[1]:offsets[1]+new_size] = \
            cv2.resize(train_y, (new_size, new_size))

        cv2.imwrite(os.path.join(train_x_dir, f"{scr_names[i]}.png"), train_x)


if __name__ == '__main__':
    main()
