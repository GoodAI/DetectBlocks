import cv2
import json
import numpy as np

from detect_blocks.detectors.aae_similarity_detector import \
        AAESimilarityDetector


def main():
    """
    This example shows how to use the AAE similarity detector to determine
    whether the .ply model at the given path is depicted in the given
    screenshot (.png image).

    We consider 4 models: 2 stages of 2 different blocks. For each model, a
    screenshot and a path to the 3D model is given as well as the pose of the
    model in the screenshot.
    """

    metadata_files = {
        'BasicAssembler_stage2': 'screenshots/BasicAssembler/12/metadata.json',
        'BasicAssembler_stage3': 'screenshots/BasicAssembler/18/metadata.json',
        'BlastFurnace_stage2': 'screenshots/BlastFurnace/12/metadata.json',
        'BlastFurnace_stage3': 'screenshots/BlastFurnace/18/metadata.json',
    }

    # load screenshots, models and poses for each block
    blocks_data = {block_id: {} for block_id in metadata_files.keys()}
    for block_id, metadata_path in metadata_files.items():
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

            # load screenshots and mask
            screenshot_original = cv2.imread(
                metadata['screenshot_original_path'].replace('\\', '/'))
            screenshot_mask = cv2.cvtColor(cv2.imread(
                metadata['screenshot_mask_path'].replace('\\', '/')),
                    cv2.COLOR_BGR2GRAY)

            # crop and resize the screenshots
            x, y, w, h = cv2.boundingRect(screenshot_mask)
            bbx_side = int(max(w, h) * 1.2)
            x += w // 2 - bbx_side // 2
            y += h // 2 - bbx_side // 2
            screenshot_original = cv2.resize(screenshot_original[
                max(0, y):min(y + bbx_side, screenshot_original.shape[0]),
                max(0, x):min(x + bbx_side, screenshot_original.shape[1])
            ], (128, 128)) / 255.0
            blocks_data[block_id]['screenshot_original'] = screenshot_original

            # load pose
            right = metadata['orientation']['right']
            forward = metadata['orientation']['forward']
            up = metadata['orientation']['up']
            position = metadata['orientation']['position']
            pose = np.array([
                [right['x'], forward['x'], up['x'], 0.0],
                [-right['y'], -forward['y'], -up['y'], 0.0],
                [-right['z'], -forward['z'], -up['z'], -300 * position['z']],
                [0.0, 0.0, 0.0, 1.0]
            ])
            blocks_data[block_id]['pose'] = pose

            # load model path
            blocks_data[block_id]['model_path'] = \
                metadata['model_path'].replace(
                    '\\', '/').replace('.obj', '.ply')

    # init a similarity detector based on the augmented autoencoder
    detector = AAESimilarityDetector(pose_given=True)

    # for each pair of blocks, load the screenshot of the first and model
    # of the second and get the prediction of the detector
    for id_block1, data_block1 in blocks_data.items():
        print('')
        for id_block2, data_block2 in blocks_data.items():
            is_same_block_and_stage = detector.is_model_in_screenshot(
                data_block1['screenshot_original'],
                data_block2['model_path'], data_block2['pose'])
            print('Screenshot {} contains model {}: {}'.format(
                id_block1, id_block2, is_same_block_and_stage))

    # always close session to avoid GPU memory leaks
    detector.close_session()


if __name__ == '__main__':
    main()
