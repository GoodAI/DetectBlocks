import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.special import expit
from tensorflow.python.keras.backend import eager_learning_phase_scope

from .similarity_detector import SimilarityDetector


class SiameseSimilarityDetector(SimilarityDetector):

    def __init__(self, model='contrastive', threshold=None, pose_given=False):
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            tf.config.experimental.set_memory_growth(gpus[0], True)

        if workspace_path is None:
            print('Please define a workspace path:\n')
            print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
            exit(-1)
        experiment_group = 'siamese'

        if model == 'siam_contr':
            embed_network = tf.keras.models.load_model(
                os.path.join(workspace_path, exp_group, 'contrastive.h5'))
            self.threshold = -0.35 if pose_given else -0.13
        elif model == 'siam_triplet':
            embed_network = tf.keras.models.load_model(
                os.path.join(workspace_path, exp_group, 'triplet.h5'))
            self.threshold = -0.26 if pose_given else -0.12
        else:
            raise ValueError('Argument \'model\' must be either \'siam_contr\''
                             ' or \'siam_triplet\'')

        if threshold is not None:
            self.threshold = threshold

        self.embed_function = K.function([embed_network.input],
                                         [embed_network.output])

        self.pose_given = pose_given

    def generate_embeddings(self, x):
        x = np.stack(x) if isinstance(x, list) else x
        with eager_learning_phase_scope(value=0):
            return self.embed_function([x])[0]

    def emb_group_similarity(self, embs1, embs2):
        embs1 = np.stack(embs1) if isinstance(embs1, list) else embs1
        embs2 = np.stack(embs2) if isinstance(embs2, list) else embs2
        emb_distances = self.emb_pairwise_similarities(embs1, embs2)
        if self.pose_given:
            return expit(np.mean(np.diag(emb_distances)) - self.threshold)
        else:
            return expit(np.max(emb_distances) - self.threshold)

    def emb_pairwise_similarities(self, embs1, embs2):
        embs1 = np.stack(embs1) if isinstance(embs1, list) else embs1
        embs2 = np.stack(embs2) if isinstance(embs2, list) else embs2
        norms_emb1 = np.square(np.linalg.norm(embs1, axis=1, keepdims=True))
        norms_emb2 = np.square(np.linalg.norm(embs2, axis=1))
        embs_dot = np.matmul(embs1, embs2.T)
        return -(norms_emb1 - 2 * embs_dot + norms_emb2) / 100.0
