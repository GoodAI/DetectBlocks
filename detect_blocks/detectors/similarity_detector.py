import abc
import cv2
import numpy as np

from auto_pose.meshrenderer import meshrenderer_phong
from auto_pose.meshrenderer.pysixd import transform as T


class SimilarityDetector(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, threshold):
        self.threshold = threshold
        self.principal_views = np.array([
            [0.0, 0.0, 0.0],
            [0.5 * np.pi, 0.0, 0.0],
            [np.pi, 0.0, 0.0],
            [1.5 * np.pi, 0.0, 0.0],
            [0.0, 0.5 * np.pi, 0.0],
            [0.0, -0.5 * np.pi, 0.0]
        ])
        self.antialiasing = 8
        self.vertex_scale = 1
        self.h = self.w = 512
        self.K = np.array([
            [1075.65, 0, 512/2],
            [0, 1073.90, 512/2],
            [0, 0, 1]
        ])
        self.clip_near = 10
        self.clip_far = 10000

    def image_similarity(self, screenshot, rendered):
        return self.image_group_similarity([screenshot], [rendered])

    def image_group_similarity(self, screenshot_imgs, rendered_imgs):
        if isinstance(screenshot_imgs, list):
            screenshot_imgs = np.stack(screenshot_imgs)
        if isinstance(rendered_imgs, list):
            rendered_imgs = np.stack(rendered_imgs)
        screenshot_embs = self.generate_embeddings(screenshot_imgs)
        rendered_embs = self.generate_embeddings(rendered_imgs)
        return self.emb_group_similarity(screenshot_embs, rendered_embs)

    def is_model_in_screenshot(self, screenshot, model_path, pose):
        return self.is_model_in_screenshots([screenshot], model_path, pose)

    def is_model_in_screenshots(self, screenshot_imgs, model_path, pose):
        if pose is None:
            rendered_imgs = self.render_principal_views(model_path)
        else:
            rendered_imgs = [self.render_in_pose(model_path, pose)]
        similarity_score = self.image_group_similarity(screenshot_imgs,
                                                       rendered_imgs)
        img_id = np.random.uniform()
        cv2.imwrite('/tmp/{}_scrsht.png'.format(img_id),
                    screenshot_imgs[0]*255)
        cv2.imwrite('/tmp/{}_render.png'.format(img_id),
                    rendered_imgs[0]*255)

        return similarity_score

    def render_in_pose(self, model_path, pose):
        renderer = meshrenderer_phong.Renderer(
            [model_path], self.antialiasing, vertex_scale=self.vertex_scale,
            vertex_tmp_store_folder='/tmp')
        R = pose[:3, :3]
        t = pose[:3, 3]
        rendered_image, _ = renderer.render(
            0, self.w, self.h, self.K, R, t,
            near=self.clip_near, far=self.clip_far)
        mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
        left, top, width, height = cv2.boundingRect(mask)
        bbx_side = int(max(width, height) * 1.2)
        left += width // 2 - bbx_side // 2
        top += height // 2 - bbx_side // 2
        rendered_image = rendered_image[
            max(0, top):min(top + bbx_side, rendered_image.shape[0]),
            max(0, left):min(left + bbx_side, rendered_image.shape[1])
        ]
        return cv2.resize(rendered_image, (128, 128)) / 255.0

    def render_in_pose_generator(self, model_paths, poses):
        renderer = meshrenderer_phong.Renderer(
            model_paths, self.antialiasing, vertex_scale=self.vertex_scale,
            vertex_tmp_store_folder='/tmp')
        for i, pose in enumerate(poses):
            R = pose[:3, :3]
            t = pose[:3, 3]
            rendered_image, _ = renderer.render(
                i, self.w, self.h, self.K, R, t,
                near=self.clip_near, far=self.clip_far)
            mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
            left, top, width, height = cv2.boundingRect(mask)
            bbx_side = int(max(width, height) * 1.2)
            left += width // 2 - bbx_side // 2
            top += height // 2 - bbx_side // 2
            rendered_image = rendered_image[
                max(0, top):min(top + bbx_side, rendered_image.shape[0]),
                max(0, left):min(left + bbx_side, rendered_image.shape[1])
            ]
            yield cv2.resize(rendered_image, (128, 128)) / 255.0

    def render_principal_views(self, model_path):
        renderer = meshrenderer_phong.Renderer(
            [model_path], self.antialiasing, vertex_scale=self.vertex_scale,
            vertex_tmp_store_folder='/tmp')
        rendered_imgs = []
        t = np.array([0.0, 0.0, 1500.0])
        for view_euler in self.principal_views:
            R = T.euler_matrix(*view_euler)[:3, :3]

            rendered_full_model = False
            while not rendered_full_model:
                rendered_image, _ = renderer.render(
                    0, self.w, self.h, self.K, R, t,
                    near=self.clip_near, far=self.clip_far)
                mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
                left, top, width, height = cv2.boundingRect(mask)
                rendered_full_model = width < self.w and height < self.h
                if not rendered_full_model:
                    t[2] *= 1.5

            bbx_side = int(max(width, height) * 1.2)
            left += width // 2 - bbx_side // 2
            top += height // 2 - bbx_side // 2
            rendered_image = rendered_image[
                max(0, top):min(top + bbx_side, rendered_image.shape[0]),
                max(0, left):min(left + bbx_side, rendered_image.shape[1])
            ]
            rendered_image = cv2.resize(rendered_image, (128, 128)) / 255.0

            for _ in range(4):
                rendered_image = cv2.rotate(
                    rendered_image, cv2.ROTATE_90_CLOCKWISE)
                rendered_imgs.append(rendered_image)

        return rendered_imgs

    @abc.abstractmethod
    def generate_embeddings(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def emb_group_similarity(self, embs1, embs2):
        raise NotImplementedError

    @abc.abstractmethod
    def emb_pairwise_similarities(self, embs1, embs2):
        raise NotImplementedError
