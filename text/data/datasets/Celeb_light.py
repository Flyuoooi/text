import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json

logging.getLogger().setLevel(logging.ERROR)   # 不打印属性缺少warning

class Celeb_light(object):
    """ Celeb-reID-light

    Reference:
        Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'Celeb-reID-light'
    def __init__(self, root='data', aux_info=False, meta_dir='PAR_PETA_105.txt',meta_dims=105, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.meta_dir = meta_dir
        self.meta_dims = meta_dims
        self.aux_info = aux_info

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir, aux_info=self.aux_info)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir, aux_info=self.aux_info)
        
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> Celeb loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes  # 9021
        self.num_test_clothes = num_test_clothes  # 1821
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _lookup_attr(self, img_abs_path, imgdir2attribute):
        """将绝对路径转换为相对路径查找属性"""
        rel = img_abs_path.replace(self.dataset_dir + '/', '')
        if rel in imgdir2attribute:
            return imgdir2attribute[rel]
        # 有的文件里不带 train/query/test 前缀
        rel2 = '/'.join(img_abs_path.split('/')[-2:])  # 只取最后两级
        if rel2 in imgdir2attribute:
            return imgdir2attribute[rel2]
        logging.warning(f"[WARN] Attribute not found for {rel}, using zeros.")
        return [0 for _ in range(self.meta_dims)]

    def _process_dir_train(self, dir_path, aux_info=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)

        imgdir2attribute = {}
        if aux_info:
            with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
                for line in f:
                    imgdir, attribute_id, is_present = line.split()
                    if imgdir not in imgdir2attribute:
                        imgdir2attribute[imgdir] = [0 for _ in range(self.meta_dims)]
                    imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            pid_label = pid2label[pid]
            clothes_id = clothes2label[clothes]

            if aux_info:
                attr = self._lookup_attr(img_path, imgdir2attribute)
                dataset.append((img_path, pid_label, camid, clothes_id, attr))
            else:
                dataset.append((img_path, pid_label, camid, clothes_id))

            pid2clothes[pid_label, clothes_id] = 1

        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes


    def _process_dir_test(self, query_path, gallery_path, aux_info=False):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()

        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)

        imgdir2attribute = {}
        if aux_info:
            with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
                for line in f:
                    imgdir, attribute_id, is_present = line.split()
                    if imgdir not in imgdir2attribute:
                        imgdir2attribute[imgdir] = [0 for _ in range(self.meta_dims)]
                    imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []

        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_label = pid2label[pid]
            clothes_label = clothes2label[clothes_id]

            if aux_info:
                attr = self._lookup_attr(img_path, imgdir2attribute)
                query_dataset.append((img_path, pid_label, camid, clothes_label, attr))
            else:
                query_dataset.append((img_path, pid_label, camid, clothes_label))

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_label = pid2label[pid]
            clothes_label = clothes2label[clothes_id]

            if aux_info:
                attr = self._lookup_attr(img_path, imgdir2attribute)
                gallery_dataset.append((img_path, pid_label, camid, clothes_label, attr))
            else:
                gallery_dataset.append((img_path, pid_label, camid, clothes_label))

        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)
        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes

if __name__ =='__main__':
    dataset=Celeb_light('/home/jwt/code/MADE-master/data/Data/ReIDData')
    print(dataset.num_train_clothes)
    print(dataset.num_test_clothes)