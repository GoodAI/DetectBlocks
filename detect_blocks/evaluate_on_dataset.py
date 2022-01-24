import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from scipy.special import expit
from tqdm import tqdm

import utils
from detectors.aae_similarity_detector import AAESimilarityDetector


def evaluate_on_dataset(arguments):
    pose_given = arguments.pose_status != 'unknown'
    detector = AAESimilarityDetector(arguments.cfg_name, arguments.at_step,
                                     arguments.threshold, pose_given)

    print('Generating screenshot dataset from metadata...')
    scrsh_dataset = utils.dataset_from_metadata(
        arguments.dataset_path, arguments.metadata_file,
        arguments.pink_background)

    print('Generating rendered dataset from metadata...')
    rendered_dataset = utils.render_images_from_metadata(
        arguments.dataset_path, arguments.metadata_file,
        arguments.pink_background, arguments.pose_status)

    print('Generating embeddings for screenshots:')
    for block_id in tqdm(scrsh_dataset.keys()):
        for stage, images in scrsh_dataset[block_id].items():
            scrsh_dataset[block_id][stage] = \
                detector.generate_embeddings(images)
    print('Generating embeddings for rendered images:')
    for block_id in tqdm(rendered_dataset.keys()):
        for stage, images in rendered_dataset[block_id].items():
            rendered_dataset[block_id][stage] = \
                detector.generate_embeddings(images)

    nums_of_negatives = [10]
    n_datasets = len(nums_of_negatives) + 1
    predictions = [[] for i in range(n_datasets)]
    labels = [[] for i in range(n_datasets)]

    # evaluate accuracy according to evaluation algorithm 1
    # (use different stages of the same block as negative examples)
    for block_id in scrsh_dataset.keys():
        for scrsh_stage, scrsh_embs in scrsh_dataset[block_id].items():
            cur_preds = []
            cur_labels = []
            for ren_stage, ren_embs in rendered_dataset[block_id].items():
                sims = \
                    detector.emb_pairwise_similarities(scrsh_embs, ren_embs)
                if arguments.multi_view:
                    if pose_given:
                        cur_preds.append([np.mean(np.diag(sims))])
                    else:
                        cur_preds.append([np.max(sims)])
                    cur_labels.append(scrsh_stage == ren_stage)
                else:
                    if pose_given:
                        cur_preds.append(np.diag(sims))
                    else:
                        cur_preds.append(np.max(sims, axis=1))
                    cur_labels.extend([scrsh_stage == ren_stage]*sims.shape[0])
            if arguments.mod_stage_alg:
                cur_preds = np.vstack(cur_preds).T
                argmax = np.argmax(cur_preds, axis=1)
                cur_preds = np.zeros_like(cur_preds)
                cur_preds[range(cur_preds.shape[0]), argmax] = 1.0
                cur_labels = np.zeros_like(cur_preds)
                cur_labels[range(cur_preds.shape[0]), scrsh_stage] = 1.0
                predictions[0].extend(cur_preds.reshape(1, -1)[0])
                labels[0].extend(cur_labels.reshape(1, -1)[0])
            else:
                cur_preds = [pred for sub_pred in cur_preds for pred in sub_pred]
                predictions[0].extend(cur_preds)
                labels[0].extend(cur_labels)
    predictions[0] = np.array(predictions[0])
    labels[0] = np.array(labels[0], dtype=bool)

    # evaluate accuracy according to evaluation algorithm 2
    # (use random stages of 10 different blocks as negative examples).
    # 'nums_of_negatives' allows to test for other numbers of negative blocks
    for i, n in enumerate(nums_of_negatives):
        for block_id in scrsh_dataset.keys():
            for scrsh_stage, scrsh_embs in scrsh_dataset[block_id].items():
                sims = detector.emb_pairwise_similarities(
                    scrsh_embs, rendered_dataset[block_id][scrsh_stage])
                if arguments.multi_view:
                    if pose_given:
                        predictions[i+1].append(np.mean(np.diag(sims)))
                    else:
                        predictions[i+1].append(np.max(sims))
                    labels[i+1].append(True)
                else:
                    if pose_given:
                        predictions[i+1].extend(np.diag(sims))
                    else:
                        predictions[i+1].extend(np.max(sims, axis=1))
                    labels[i+1].extend([True] * sims.shape[0])
                neg_choices = list(rendered_dataset.keys())
                neg_choices.remove(block_id)
                neg_block_ids = random.sample(neg_choices, n)
                for neg_block_id in neg_block_ids:
                    stage = random.choice(
                        list(rendered_dataset[neg_block_id].keys()))
                    ren_embs = rendered_dataset[neg_block_id][stage]
                    sims = \
                        detector.emb_pairwise_similarities(scrsh_embs,
                                                           ren_embs)
                    if arguments.multi_view:
                        if pose_given:
                            predictions[i+1].append(np.mean(np.diag(sims)))
                        else:
                            predictions[i+1].append(np.max(sims))
                        labels[i+1].append(False)
                    else:
                        if pose_given:
                            predictions[i+1].extend(np.diag(sims))
                        else:
                            predictions[i+1].extend(np.max(sims, axis=1))
                        labels[i+1].extend([False] * sims.shape[0])
        predictions[i+1] = np.array(predictions[i+1])
        labels[i+1] = np.array(labels[i+1], dtype=bool)

    accuracies = np.zeros((n_datasets, 2000))
    thresholds = np.linspace(-1.0, 1.0, 2000)
    for j in range(n_datasets):
        for i, thresh in enumerate(thresholds):
            accuracies[j][i] = np.mean((predictions[j] > thresh) == labels[j])

    max_acc_idx = np.argmax(accuracies[1])
    best_thresh = thresholds[max_acc_idx]
    print('Threshold: {}'.format(best_thresh))
    for i in range(1, n_datasets):
        predictions[i] = expit(predictions[i] - best_thresh)

    n_thresholds = 1000
    accuracies = np.zeros((n_datasets, n_thresholds))
    precisions = np.zeros((n_datasets, n_thresholds))
    recalls = np.zeros((n_datasets, n_thresholds))
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    for i in range(n_datasets):
        for j, thresh in enumerate(thresholds):
            accuracies[i][j] = np.mean((predictions[i] > thresh) == labels[i])
            precisions[i][j] = np.sum(predictions[i][labels[i]] > thresh) / (float(np.sum(predictions[i] > thresh)) + 1e-6)
            recalls[i][j] = np.sum(predictions[i][labels[i]] > thresh) / (float(np.sum(labels[i])) + 1e-6)

    for i in range(n_datasets):
        accuracy = accuracies[i][int(n_thresholds / 2)]
        precision = precisions[i][int(n_thresholds / 2)]
        recall = recalls[i][int(n_thresholds / 2)]
        print('Accuracy/precision/recall on Task '
            '{}: {:.3f}/{:.3f}/{:.3f}'.format(i+1, accuracy, precision, recall))

    fig, axes = plt.subplots(nrows=2, ncols=n_datasets, figsize=(20, 10))
    for i in range(n_datasets):
        accuracy = accuracies[i][int(n_thresholds / 2)]
        axes[0][i].set_title(
            'Task {}: Accuracies\n({:.3f} at threshold 0.5)'.format(
                i+1, accuracy))
        axes[0][i].set_xlabel('Threshold')
        axes[0][i].set_ylabel('Accuracy')
        axes[0][i].set_ylim(0.0, 1.0)
        axes[0][i].plot(thresholds, accuracies[i], marker='.')
        axes[1][i].set_title('Task {}: Precision-Recall Curve'.format(i+1))
        axes[1][i].set_xlabel('Recall')
        axes[1][i].set_ylabel('Precision')
        axes[1][i].plot(recalls[i], precisions[i], marker='.')

    bg_str = 'pink' if arguments.pink_background else 'original'
    pose_str = arguments.pose_status
    model_str = 'aae_{}_{}'.format(arguments.cfg_name, arguments.at_step)
    plot_filename = '{}_bg_{}_pose_{}.png'.format(bg_str, pose_str, model_str)
    plot_save_path = os.path.join(arguments.plot_dir, plot_filename)
    print('Saving plot to {}'.format(plot_save_path))
    if not os.path.exists(arguments.plot_dir):
        os.makedirs(arguments.plot_dir)
    plt.savefig(plot_save_path, bbox_inches='tight')

    detector.close_session()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--at_step',  type=int, default=170000)
    parser.add_argument('--cfg_name', type=str, default='ult2')
    parser.add_argument('--metadata_file', type=str,
                        default='metadata_files.txt')
    parser.add_argument('--plot_dir', type=str,
                        default='/tmp/detect_blocks_plots')
    parser.add_argument('--pose_status', type=str, default='known',
                        choices=['known', 'perturbed', 'unknown'])
    parser.add_argument('--threshold',  type=float, default=None)
    parser.add_argument('-mod_stage_alg',  action='store_true', default=False)
    parser.add_argument('-multi_view',  action='store_true', default=False)
    parser.add_argument('-pink_background', action='store_true', default=False)
    arguments = parser.parse_args()

    evaluate_on_dataset(arguments)


if __name__ == '__main__':
    main()
