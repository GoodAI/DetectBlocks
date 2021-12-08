import argparse
import configparser
import cv2
import hashlib
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

from auto_pose.ae import utils as u


def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path is None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("opengl_data_path")
    parser.add_argument("blenderproc_data_path")
    arguments = parser.parse_args()

    exp_group, exp_name = arguments.experiment_name.split('/')

    args_file_path = u.get_config_file_path(workspace_path, exp_name, exp_group)
    dataset_path = u.get_dataset_path(workspace_path)

    args = configparser.ConfigParser(inline_comment_prefixes="#")
    args.read(args_file_path)

    group_ids = eval(args.get('Paths', 'MODEL_PATH'))
    model_list = eval(args.get('Paths', 'MODEL_LIST'))
    num_groups = len(group_ids)
    group_size = len(model_list) / num_groups

    model_names = []
    for model_path in model_list:
        start_idx = model_path.rfind('models/') + len('models/')
        end_idx = model_path.rfind('.')
        model_name = model_path[start_idx:end_idx].replace('/','')
        model_names.append(model_name)

    for i, group_id in enumerate(group_ids):
        md5_string = str(str(args.items('Dataset')) + group_id)
        md5_string = md5_string.encode('utf-8')
        current_config_hash = hashlib.md5(md5_string).hexdigest()

        current_file_name = os.path.join(dataset_path,
                                         current_config_hash + '.tfrecord')
        
        if not os.path.exists(current_file_name):
            writer = tf.python_io.TFRecordWriter(current_file_name)
            print('Generating tfrecord for model group {}'.format(group_id))
        else:
            print('TFrecord exists for model group {}'.format(group_id))
            continue

        for j in range(group_size):
            print('Loading images for model {}'.format(
                model_names[group_size*i+j]))
            for k in tqdm(range(args.getint('Dataset', 'NOOF_TRAINING_IMGS'))):
                train_x = cv2.imread(os.path.join(
                    arguments.blenderproc_data_path,
                    'output/{model_name}/rgb_{k:04d}.png'.format(
                        model_name=model_names[group_size*i+j], k=k)))
                train_x = train_x.astype(np.uint8)
                mask_x = (train_x[:, :, 0] == 0) & \
                         (train_x[:, :, 1] == 0) & \
                         (train_x[:, :, 2] == 0)
                train_y = cv2.imread(os.path.join(
                    arguments.opengl_data_path,
                    'train_y/{model_name}_{k:04d}.png'.format(
                        model_name=model_names[group_size*i+j], k=k)))
                train_y = train_y.astype(np.uint8)

                train_x_bytes = train_x.tostring()
                mask_x_bytes = mask_x.tostring()
                train_y_bytes = train_y.tostring()

                feature = {}
                feature['train_x'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[train_x_bytes]))
                feature['train_y'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[train_y_bytes]))
                feature['mask'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[mask_x_bytes]))

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                writer.write(serialized)

        writer.close()


if __name__ == '__main__':
    main()
