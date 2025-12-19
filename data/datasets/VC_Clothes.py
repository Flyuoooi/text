import os
import re
import glob
import random
import logging
import numpy as np
import os.path as osp

logging.getLogger().setLevel(logging.ERROR)   # 不打印属性缺少 warning


class VC_Clothes(object):
    """
    当 aux_info=True 时，样本格式为：(img_path, pid, camid, clothes_id, attr)
    当 aux_info=False 时，样本格式为：(img_path, pid, camid, clothes_id)
    """
    dataset_dir = 'VC-Clothes'

    def __init__(self, root='data', aux_info=False,
                 meta_dir='PAR_PETA_105.txt', meta_dims=105,
                 mode='all', **kwargs):
        """
        args:
            mode:
                'all' 使用所有相机
                'sc'  只用 same-clothes 相机 (常见是 cam 2,3)
                'cc'  只用 clothes-changing 相机 (常见是 cam 3,4)
        """
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.aux_info = aux_info
        self.meta_dir = meta_dir
        self.meta_dims = meta_dims
        self.mode = mode

        self._check_before_run()
        self.imgdir2attribute = self._load_attr_dict()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> VC-Clothes loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset      | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train       | {:5d} | {:8d} | {:9d}".format(
            num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test        | {:5d} | {:8d} | {:9d}".format(
            num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query       | {:5d} | {:8d} |".format(
            num_test_pids, num_query_imgs))
        logger.info("  gallery     | {:5d} | {:8d} |".format(
            num_test_pids, num_gallery_imgs))
        logger.info("  --------------------------------------------")
        logger.info("  total       | {:5d} | {:8d} | {:9d}".format(
            num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
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

    def _load_attr_dict(self):
        """读取类似 PAR_PETA_105.txt 的属性文件；
           如果找不到，则返回空 dict，后面用全 0 向量代替。
        """
        meta_path = osp.join(self.dataset_dir, self.meta_dir)
        if not osp.exists(meta_path):
            logging.warning(f"[VC-Clothes] meta file {meta_path} not found, use zeros as attr.")
            return {}

        imgdir2attribute = {}
        with open(meta_path, 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for _ in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)
        return imgdir2attribute

    def _lookup_attr(self, img_abs_path):
        """根据 img_abs_path 查询属性；如果没有就返回全 0 向量。"""
        if not self.imgdir2attribute:
            return [0 for _ in range(self.meta_dims)]

        # 相对路径匹配
        rel = img_abs_path.replace(self.dataset_dir + '/', '')
        if rel in self.imgdir2attribute:
            return self.imgdir2attribute[rel]

        rel2 = '/'.join(img_abs_path.split('/')[-2:])
        if rel2 in self.imgdir2attribute:
            return self.imgdir2attribute[rel2]

        logging.warning(f"[VC-Clothes] Attribute not found for {rel}, using zeros.")
        return [0 for _ in range(self.meta_dims)]


    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')  # pid-camid-clothes-seq

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid_container.add(int(pid))
            clothes_container.add(clothes_id)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {cid: label for label, cid in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            pid = int(pid)
            camid = int(camid) - 1  # cam 从 0 开始
            clothes_id = pid + clothes

            pid_label = pid2label[pid]
            clothes_label = clothes2label[clothes_id]

            if self.aux_info:
                attr = self._lookup_attr(img_path)
                dataset.append((img_path, pid_label, camid, clothes_label, attr))
            else:
                dataset.append((img_path, pid_label, camid, clothes_label))

            pid2clothes[pid_label, clothes_label] = 1

        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes


    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

        pid_container = set()
        clothes_container = set()

        # 先统计 pid / clothes_id
        for img_path in query_img_paths + gallery_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            pid = int(pid)
            camid = int(camid)
            if self.mode == 'sc' and camid not in [2, 3]:
                continue
            if self.mode == 'cc' and camid not in [3, 4]:
                continue
            clothes_id = str(pid) + clothes
            pid_container.add(pid)
            clothes_container.add(clothes_id)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {cid: label for label, cid in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        def _filter(paths):
            dataset = []
            for img_path in paths:
                pid, camid, clothes, _ = pattern.search(img_path).groups()
                pid = int(pid)
                camid = int(camid)
                if self.mode == 'sc' and camid not in [2, 3]:
                    continue
                if self.mode == 'cc' and camid not in [3, 4]:
                    continue
                camid -= 1
                clothes_id = str(pid) + clothes

                pid_label = pid2label[pid]
                clothes_label = clothes2label[clothes_id]

                if self.aux_info:
                    attr = self._lookup_attr(img_path)
                    dataset.append((img_path, pid_label, camid, clothes_label, attr))
                else:
                    dataset.append((img_path, pid_label, camid, clothes_label))
            return dataset

        query_dataset = _filter(query_img_paths)
        gallery_dataset = _filter(gallery_img_paths)

        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)
        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes


if __name__ == '__main__':
    dataset = VC_Clothes(root='/home/jwt/code/clip/data/Data/ReIDData', aux_info=True)
    print(dataset.num_train_pids)
    print(dataset.num_train_clothes)
    print(len(dataset.train), len(dataset.query), len(dataset.gallery))
