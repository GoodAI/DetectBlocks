#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd


def evaluate_embeddings(embeddings):
    for col in ("embeddings_original", "embeddings_pink", "embeddings_model"):
        new_col = f"{col}_normalized"
        embeddings[new_col] = embeddings.apply(
            lambda x: x[col] / np.linalg.norm(x[col]), axis=1
        )

    data = []

    positives, negatives = 0, 0

    for block, group in embeddings.groupby("block"):
        group_rows = len(group)
        if group_rows < 2:
            continue

        for model_i in range(group_rows):
            em_model = group.iloc[model_i]["embeddings_model_normalized"]
            model_name = f"{block}_{model_i}"

            best_match_image = -1
            best_match_distance = 10

            for image_i in range(group_rows):
                em_orig = group.iloc[image_i]["embeddings_original_normalized"]
                em_pink = group.iloc[image_i]["embeddings_pink_normalized"]
                label = image_i == model_i
                distance_original = np.linalg.norm(em_model - em_orig)
                distance_pink = np.linalg.norm(em_model - em_pink)

                result = dict(
                    block=block,
                    image=f"{block}_{image_i}",
                    model=model_name,
                    label=label,
                    distance_original=distance_original,
                    distance_pink=distance_pink,
                )
                data.append(result)

                if distance_original < best_match_distance:
                    best_match_distance = distance_original
                    best_match_image = image_i

            if best_match_image == model_i:
                positives += 1
            else:
                negatives += 1
                print(model_name)

    print(positives, negatives, positives + negatives)

    return pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("EMBEDDINGS")
    parser.add_argument("OUTPUT", help="Pandas h5 file")

    args = parser.parse_args()

    df = evaluate_embeddings(pd.read_hdf(args.EMBEDDINGS, key="table"))

    df.to_hdf(args.OUTPUT, key="pairs", complevel=4, mode="w")
