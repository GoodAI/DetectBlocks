import cv2
import json
import numpy as np
import os
from tqdm import tqdm

from auto_pose.meshrenderer import meshrenderer_phong
from auto_pose.meshrenderer.pysixd import transform as T


def posix(path_string):
    return path_string.replace('\\', os.sep)


def dataset_from_metadata(dataset_path, metadata_filename, pink_background):
    data_dict = metadata_dict_from_file(dataset_path, metadata_filename)
    bg_str = 'pink' if pink_background else 'original'
    for block_id in tqdm(data_dict.keys()):
        for stage, metas in data_dict[block_id].items():
            stage_data = []
            for metadata in metas:
                scrsh_path = \
                    posix(metadata['screenshot_{}_path'.format(bg_str)])
                mask_path = posix(metadata['screenshot_mask_path'])
                screenshot = cv2.imread(
                    os.path.join(dataset_path, scrsh_path))
                mask = cv2.cvtColor(
                    cv2.imread(os.path.join(dataset_path, mask_path)),
                    cv2.COLOR_BGR2GRAY)
                x, y, w, h = cv2.boundingRect(mask)
                bbx_side = int(max(w, h) * 1.2)
                x += w // 2 - bbx_side // 2
                y += h // 2 - bbx_side // 2
                cropped_screenshot = cv2.resize(screenshot[
                    max(0, y):min(y + bbx_side, screenshot.shape[0]),
                    max(0, x):min(x + bbx_side, screenshot.shape[1])
                ], (128, 128))
                stage_data.append(cropped_screenshot / 255.0)

            data_dict[block_id][stage] = stage_data

    return data_dict


def render_images_from_metadata(dataset_path, metadata_filename,
                                pink_background=False, pose_status='known'):
    antialiasing = 8
    vertex_scale = 1
    h = w = 512
    K = np.array([[1075.65, 0, 512/2], [0, 1073.90, 512/2], [0, 0, 1]])
    clip_near = 10
    clip_far = 10000

    principal_views = np.array([
        [0.0, 0.0, 0.0],
        [0.5 * np.pi, 0.0, 0.0],
        [np.pi, 0.0, 0.0],
        [1.5 * np.pi, 0.0, 0.0],
        [0.0, 0.5 * np.pi, 0.0],
        [0.0, -0.5 * np.pi, 0.0]
    ])

    data_dict, metadata_list = metadata_dict_from_file(dataset_path,
                                                       metadata_filename, True)

    model_paths = set([posix(meta['model_path']) for meta in metadata_list])
    obj_ids = {os.path.join(dataset_path, path): id
               for id, path in enumerate(sorted(model_paths))}

    print('Initializing renderer...')
    renderer = meshrenderer_phong.Renderer(
        sorted(list(obj_ids.keys())),
        antialiasing, vertex_scale=vertex_scale,
        vertex_tmp_store_folder='/tmp')

    print('Rendering images from metadata:')
    for block_id in tqdm(data_dict.keys()):
        for stage, metas in data_dict[block_id].items():
            stage_data = []
            if pose_status != 'unknown':
                for metadata in metas:
                    obj_id = obj_ids[os.path.join(
                        dataset_path, posix(metadata['model_path']))]
                    right = metadata['orientation']['right']
                    forward = metadata['orientation']['forward']
                    up = metadata['orientation']['up']
                    position = metadata['orientation']['position']
                    R = np.array([
                        [right['x'], forward['x'], up['x']],
                        [-right['y'], -forward['y'], -up['y']],
                        [-right['z'], -forward['z'], -up['z']],
                    ])
                    t = np.array([0.0, 0.0, -300 * position['z']])

                    if pose_status == 'perturbed':
                        pert_euler = np.random.uniform(-0.15, 0.15, (3,))
                        R_pert = T.euler_matrix(*pert_euler)[:3, :3]
                        R = np.matmul(R, R_pert)

                    rendered_image, _ = renderer.render(
                        obj_id, w, h, K, R, t,
                        near=clip_near, far=clip_far)

                    stage_data.append(rendered_image)
            else:
                t = np.array([0.0, 0.0, 1500.0])
                obj_id = obj_ids[os.path.join(dataset_path,
                                              posix(metas[0]['model_path']))]

                for view_euler in principal_views:
                    R = T.euler_matrix(*view_euler)[:3, :3]

                    rendered_full_model = False
                    while not rendered_full_model:
                        rendered_image, _ = renderer.render(
                            obj_id, w, h, K, R, t,
                            near=clip_near, far=clip_far)
                        mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
                        left, top, width, height = cv2.boundingRect(mask)
                        rendered_full_model = width < w and height < h
                        if not rendered_full_model:
                            t[2] *= 1.5

                    stage_data.append(rendered_image)

            data_dict[block_id][stage] = []

            block_valid = True
            for rendered_image in stage_data:
                rendered_image = center_object(rendered_image)
                if rendered_image.shape[0] < 1 or rendered_image.shape[1] < 0:
                    block_valid = False
                    break
                rendered_image = cv2.resize(rendered_image, (128, 128))
                if pink_background:
                    rendered_image = \
                        fill_background(rendered_image, [150, 0, 150])
                data_dict[block_id][stage].append(rendered_image / 255.0)
            if not block_valid:
                print(block_id, stage, R, t)

    return data_dict


def center_object(image):
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left, top, width, height = cv2.boundingRect(mask)
    bbx_side = int(max(width, height) * 1.2)
    left += width // 2 - bbx_side // 2
    top += height // 2 - bbx_side // 2
    return image[max(0, top):min(top + bbx_side, image.shape[0]),
                 max(0, left):min(left + bbx_side, image.shape[1])]


def fill_background(image, color):
    bg_mask = (image[:, :, 0] == 0) & \
              (image[:, :, 1] == 0) & \
              (image[:, :, 2] == 0)
    image[bg_mask] = color
    return image


def metadata_dict_from_file(dataset_path, metadata_filename,
                            return_metadata_list=False):
    data_dict = {}
    metadata_list = []
    with open(os.path.join(dataset_path, metadata_filename)) as meta_list:
        for line in meta_list:
            metadata_path = os.path.join(dataset_path, posix(line.strip()))
            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)
                if metadata['block_id'] not in data_dict:
                    data_dict[metadata['block_id']] = {}
                stage_dict = data_dict[metadata['block_id']]
                if metadata['stage'] not in stage_dict:
                    stage_dict[metadata['stage']] = []
                metadata['model_path'] = \
                    metadata['model_path'].replace('.obj', '.ply')
                stage_dict[metadata['stage']].append(metadata)
                metadata_list.append(metadata)
    if return_metadata_list:
        return data_dict, metadata_list
    else:
        return data_dict
