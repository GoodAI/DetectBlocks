#!/usr/bin/env python

import argparse
import cv2
import json
import numpy as np
import pandas as pd

from pathlib import Path

import detect_blocks.utils as utils
from detect_blocks.detectors.aae_similarity_detector import AAESimilarityDetector


def extract_images_with_mask(image_paths, mask_path):
    mask = cv2.imread(str(mask_path).strip(), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = cv2.boundingRect(mask)
    bbx_side = int(max(w, h) * 1.2)
    x += w // 2 - bbx_side // 2
    y += h // 2 - bbx_side // 2

    images = []
    for path in image_paths:
        image = cv2.imread(str(path).strip(), cv2.IMREAD_COLOR)
        image = cv2.resize(
            image[
                max(0, y) : min(y + bbx_side, image.shape[0]),
                max(0, x) : min(x + bbx_side, image.shape[1]),
            ],
            (128, 128),
        )
        images.append(image / 255.0)

    return images


def orientation_to_pose(orientation):
    right = orientation["right"]
    forward = orientation["forward"]
    up = orientation["up"]
    position = orientation["position"]
    return np.array(
        [
            [right["x"], forward["x"], up["x"], 0.0],
            [-right["y"], -forward["y"], -up["y"], 0.0],
            [-right["z"], -forward["z"], -up["z"], -300 * position["z"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def extract_metadata_info(metadata, total_stages, view):
    d_block = []
    d_stage = []
    d_view = []
    d_img_orig_path = []
    d_img_pink_path = []
    d_img_mask_path = []
    d_model_path = []
    d_pose = []

    base_directory = Path(utils.posix(metadata)).resolve(strict=True).parent

    last_block, skip_block = None, None
    with open(metadata, "r") as f:
        for p in f:
            metadata_path = base_directory / Path(utils.posix(p.strip()))
            (
                current_block,
                view_s,
            ) = metadata_path.parts[-3:-1]
            current_view = int(view_s)

            # there are 6 renders of a stage
            if current_view % 6 != view:
                continue

            if current_block != last_block:
                # load info about stage etc from metadata
                last_block = current_block
                with open(metadata_path) as mf:
                    data = json.load(mf)
                    skip_block = int(data["total_stages"]) != total_stages

            if current_block == skip_block:
                continue

            with open(metadata_path) as mf:
                data = json.load(mf)

                image_original, image_pink, image_mask, model = [
                    str(base_directory / utils.posix(p))
                    for p in (
                        data["screenshot_original_path"],
                        data["screenshot_pink_path"],
                        data["screenshot_mask_path"],
                        data["model_path"].replace(".obj", ".ply"),
                    )
                ]

                d_block.append(current_block)
                d_stage.append(int(data["stage"]))
                d_view.append(current_view)
                d_img_orig_path.append(image_original)
                d_img_pink_path.append(image_pink)
                d_img_mask_path.append(image_mask)
                d_model_path.append(model)
                d_pose.append(orientation_to_pose(data["orientation"]))

    return dict(
        block=d_block,
        stage=d_stage,
        view=d_view,
        image_original_path=d_img_orig_path,
        image_pink_path=d_img_pink_path,
        image_mask_path=d_img_mask_path,
        model_path=d_model_path,
        pose=d_pose,
    )


def extract_embeddings(metadata_dict):
    df_data = []

    detector = AAESimilarityDetector(pose_given=True)

    for i, image_model in enumerate(
        detector.render_in_pose_generator(
            metadata_dict["model_path"], metadata_dict["pose"]
        )
    ):
        images = extract_images_with_mask(
            [
                metadata_dict["image_original_path"][i],
                metadata_dict["image_pink_path"][i],
            ],
            metadata_dict["image_mask_path"][i],
        )
        images.append(image_model)

        (
            image_original_embeddings,
            image_pink_embeddings,
            model_embeddings,
        ) = detector.generate_embeddings(images)

        df_data.append(
            dict(
                embeddings_original=image_original_embeddings,
                embeddings_pink=image_pink_embeddings,
                embeddings_model=model_embeddings,
            )
        )

    detector.close_session()

    df = pd.DataFrame(df_data)
    df["block"] = metadata_dict["block"]
    df["stage"] = metadata_dict["stage"]
    df["view"] = metadata_dict["view"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("METADATA")
    parser.add_argument("OUTPUT", help="Pandas h5 file")
    parser.add_argument(
        "--total-stages",
        type=int,
        help="Use only blocks with 2, 3, 4 total stages",
        default=4,
    )
    parser.add_argument(
        "--view", type=int, help="One of six different views. ie number 0-5", default=0
    )

    args = parser.parse_args()

    data = extract_metadata_info(args.METADATA, args.total_stages, args.view)

    df = extract_embeddings(data)

    df.to_hdf(args.OUTPUT, key="table", complevel=4, mode="w")
