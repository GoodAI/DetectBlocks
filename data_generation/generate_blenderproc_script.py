import argparse
import numpy as np
import os
import shutil
from auto_pose.meshrenderer.pysixd import transform as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pose_list", type=str, required=True)
    parser.add_argument("--blenderproc_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    arguments = parser.parse_args()

    views_txt = np.loadtxt(arguments.model_pose_list, delimiter=' ', dtype=str)
    model_paths = views_txt[:, 0]
    model_names = views_txt[:, 1]
    model_poses = np.array(views_txt[:, 3:], dtype=np.float32)

    poses_dict = {}
    model_names_dict = {}
    for i, model_path in enumerate(model_paths):
        if model_path not in poses_dict:
            poses_dict[model_path] = []
            model_names_dict[model_path] = model_names[i]
        model_pose = model_poses[i]
        model_R = T.euler_matrix(
            model_pose[3] + np.pi, -model_pose[4], -model_pose[5])
        model_t = model_pose[:3] / 100.0
        cam_R_euler = np.array(T.euler_from_matrix(model_R.T[:3, :3]))
        cam_R_euler[0] += np.pi
        cam_t = -model_R.T[:3, :3] @ model_t
        poses_dict[model_path].append(np.hstack((cam_t, cam_R_euler)))

    bpr = arguments.blenderproc_root
    out_dir = os.path.join(arguments.output_dir, "blenderproc")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    shutil.copy2("blenderproc_config.yaml", out_dir)
    cam_poses_dir = os.path.join(out_dir, "cam_poses")
    if not os.path.exists(cam_poses_dir):
        os.makedirs(cam_poses_dir)

    with open(os.path.join(out_dir,
                           "generate_data_blenderproc.sh"), 'w') as out_script:
        for model_idx, model_path in enumerate(poses_dict.keys()):
            np.savetxt(os.path.join(cam_poses_dir, f"{model_idx:03d}"),
                       np.stack(poses_dict[model_path]))
            out_script.write(
                f"python {bpr}/run.py {out_dir}/blenderproc_config.yaml "
                f"{cam_poses_dir}/{model_idx:03d} {model_path} "
                f"{out_dir}/output/{model_names_dict[model_path]}\n")


if __name__ == '__main__':
    main()
