import configparser
import numpy as np
import os
import tensorflow as tf
from scipy.special import expit

from auto_pose.ae import ae_factory as factory
from auto_pose.ae import utils as U
from .similarity_detector import SimilarityDetector


class AAESimilarityDetector(SimilarityDetector):

    def __init__(self, cfg_name='ult2', at_step=170000,
                 threshold=None, pose_given=False):
        if threshold is None:
            threshold = 0.537 if pose_given else 0.67
        super(AAESimilarityDetector, self).__init__(threshold)
        self.pose_given = pose_given
        self.encoder, self.session = self.load_encoder(cfg_name, at_step)

    def generate_embeddings(self, x):
        x = np.stack(x) if isinstance(x, list) else x
        return self.session.run(self.encoder.z, {self.encoder.x: x})

    def emb_group_similarity(self, embs1, embs2):
        embs1 = np.stack(embs1) if isinstance(embs1, list) else embs1
        embs2 = np.stack(embs2) if isinstance(embs2, list) else embs2
        cos_similarities = self.emb_pairwise_similarities(embs1, embs2)
        if self.pose_given:
            return expit(np.mean(np.diag(cos_similarities)) - self.threshold)
        else:
            return expit(np.max(cos_similarities) - self.threshold)

    def emb_pairwise_similarities(self, embs1, embs2):
        embs1 = np.stack(embs1) if isinstance(embs1, list) else embs1
        embs2 = np.stack(embs2) if isinstance(embs2, list) else embs2
        embs1 = embs1 / np.linalg.norm(embs1, axis=1, keepdims=True)
        embs2 = embs2 / np.linalg.norm(embs2, axis=1, keepdims=True)
        return np.matmul(embs1, embs2.T)

    def load_encoder(self, cfg_name, at_step):
        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        if workspace_path is None:
            print('Please define a workspace path:\n')
            print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
            exit(-1)
        experiment_group = 'eval'

        cfg_file_path = U.get_config_file_path(
            workspace_path, cfg_name, experiment_group)
        log_dir = U.get_log_dir(
            workspace_path, cfg_name, experiment_group)

        dataset_path = U.get_dataset_path(workspace_path)

        if not os.path.exists(cfg_file_path):
            print('Could not find config file:\n')
            print('{}\n'.format(cfg_file_path))
            exit(-1)

        args = configparser.ConfigParser()
        args.read(cfg_file_path)

        checkpoint_file_basename = U.get_checkpoint_basefilename(
            log_dir, latest=at_step, joint=True)
        if not tf.compat.v1.train.checkpoint_exists(checkpoint_file_basename):
            checkpoint_file_basename = U.get_checkpoint_basefilename(
                log_dir, latest=at_step, joint=False)

        with tf.compat.v1.variable_scope(cfg_name):
            dataset = factory.build_dataset(dataset_path, args)
            queue = factory.build_queue(dataset, args)
            encoder = factory.build_encoder(queue.x, args)
            restore_saver = tf.compat.v1.train.Saver(
                save_relative_paths=True, max_to_keep=100)

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        restore_saver.restore(sess, checkpoint_file_basename)

        return encoder, sess

    def close_session(self):
        self.session.close()
