import logging

import torch
import numpy as np
import os


from utils.reranking import re_ranking


def compute_ap_cmc(index, good_index, junk_index):
    """ Compute AP and CMC for each sample
    """
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision

    return ap, cmc

# def euclidean_distance(qf, gf):
#     qf = qf.float()
#     gf = gf.float()
#     m = qf.shape[0]
#     n = gf.shape[0]

#     dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist_mat.addmm_(1, -2, qf, gf.t())
#     return dist_mat.cpu().numpy().astype(np.float32)  # 强制 float32
def euclidean_distance(qf, gf, block_size=256):
    """
    qf: [num_query, feat_dim]
    gf: [num_gallery, feat_dim]
    实现真正的分块欧式距离计算，防止显存/内存爆炸
    """

    qf = qf.float()
    gf = gf.float()

    nq = qf.size(0)
    ng = gf.size(0)

    # 预先一次性开辟完整矩阵（不会重复累积）
    dist_mat = torch.zeros((nq, ng), dtype=torch.float32)

    # 提前算好 gallery 二范数（节约重复计算）
    gf_norm = torch.sum(gf * gf, dim=1).view(1, -1)  # [1, ng]

    # 分块处理 query
    for i in range(0, nq, block_size):
        q_batch = qf[i:i + block_size]            # [b, dim]
        qb_norm = torch.sum(q_batch * q_batch, 1).view(-1, 1)  # [b, 1]

        # 完整欧式距离公式：||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        block = qb_norm + gf_norm - 2 * torch.mm(q_batch, gf.t())

        # 写回整体矩阵（不会 append）
        dist_mat[i:i + block_size] = block

    return dist_mat.clamp(min=1e-12).sqrt()



def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_clothid = q_clothes_ids[q_idx]

        order = indices[q_idx]
        # CC
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        remove = remove | ((g_pids[order] == q_pid) & (
                    g_clothes_ids[order] == q_clothid))
        # SC
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # remove = remove | ((g_pids[order] == q_pid) & ~(g_clothes_ids[order] == q_clothid))

        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


# class R1_mAP_eval():
#     def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
#         super(R1_mAP_eval, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.reranking = reranking

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []

#     def update(self, output):  # called once for each batch
#         feat, pid, camid = output
#         self.feats.append(feat.cpu())
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))

#     def compute(self):  # called after each epoch
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm:
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])

#         g_camids = np.asarray(self.camids[self.num_query:])
#         if self.reranking:
#             print('=> Enter reranking')
#             # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
#             distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

#         else:
#             print('=> Computing DistMat with euclidean_distance')
#             distmat = euclidean_distance(qf, gf)
#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

#         return cmc, mAP, distmat, self.pids, self.camids, qf, gf

# class R1_mAP_eval_LTCC():
#     def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
#         super(R1_mAP_eval_LTCC, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.reranking = reranking

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#         self.cloth_ids = []

#     def update(self, output):  # called once for each batch
#         feat, pid, camid, cloth_id = output
#         self.feats.append(feat.cpu())
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#         self.cloth_ids.extend(np.asarray(cloth_id))

#     def compute(self):  # called after each epoch
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm:
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         q_clothes_ids = np.asarray(self.cloth_ids[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])

#         g_camids = np.asarray(self.camids[self.num_query:])
#         g_clothes_ids = np.asarray(self.cloth_ids[self.num_query:])
#         if self.reranking:
#             print('=> Enter reranking')
#             # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
#             distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

#         else:
#             print('=> Computing DistMat with euclidean_distance')
#             distmat = euclidean_distance(qf, gf)

#         cmc, mAP = eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids,q_clothes_ids,g_clothes_ids)

#         return cmc, mAP, distmat, self.pids, self.camids, qf, gf

# utils/metrics.py  （只展示修改后的关键类，其余函数保持你原样）
# class R1_mAP_eval():
#     def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=True, block_size=256):
#         super(R1_mAP_eval, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.reranking = reranking
#         self.block_size = block_size

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []

#     def update(self, output):
#         feat, pid, camid = output
#         self.feats.append(feat.cpu())
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))

#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm:
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)

#         qf = feats[:self.num_query]
#         gf = feats[self.num_query:]

#         q_pids = np.asarray(self.pids[:self.num_query])
#         g_pids = np.asarray(self.pids[self.num_query:])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         g_camids = np.asarray(self.camids[self.num_query:])

#         print("=> Computing DistMat with block-wise euclidean distance")

#         all_cmc = []
#         all_AP = []

#         qf = qf.float()
#         gf = gf.float()
#         gf_norm = torch.sum(gf * gf, dim=1).view(1, -1)

#         for i in range(0, qf.size(0), self.block_size):
#             q_batch = qf[i:i + self.block_size]
#             qb_norm = torch.sum(q_batch * q_batch, dim=1).view(-1, 1)

#             dist = qb_norm + gf_norm - 2 * torch.mm(q_batch, gf.t())
#             dist = dist.cpu().numpy()

#             cmc, mAP = eval_func(
#                 dist,
#                 q_pids[i:i + q_batch.size(0)],
#                 g_pids,
#                 q_camids[i:i + q_batch.size(0)],
#                 g_camids,
#                 self.max_rank
#             )

#             all_cmc.append(cmc)
#             all_AP.append(mAP)

#         all_cmc = np.mean(np.asarray(all_cmc), axis=0)
#         mAP = np.mean(all_AP)

#         return all_cmc, mAP, None, self.pids, self.camids, qf, gf
class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, block_size=256):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.block_size = block_size

    def reset(self):
        self.feats, self.pids, self.camids = [], [], []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        qf = feats[:self.num_query].float()
        gf = feats[self.num_query:].float()

        q_pids = np.asarray(self.pids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_camids = np.asarray(self.camids[self.num_query:])

        # rerank 只能全量（会占内存），分块不支持
        if self.reranking:
            print("=> Enter reranking (full-matrix, may be heavy)")
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)
            return cmc, mAP, None, self.pids, self.camids, qf, gf

        print("=> Computing DistMat with block-wise euclidean distance")

        gf_norm = torch.sum(gf * gf, dim=1).view(1, -1)  # [1, ng]

        cmc_sum = None
        map_sum = 0.0
        n_sum = 0

        for i in range(0, qf.size(0), self.block_size):
            q_batch = qf[i:i + self.block_size]
            qb_norm = torch.sum(q_batch * q_batch, dim=1).view(-1, 1)
            dist = (qb_norm + gf_norm - 2 * torch.mm(q_batch, gf.t())).cpu().numpy()

            cmc, mAP = eval_func(
                dist,
                q_pids[i:i + q_batch.size(0)],
                g_pids,
                q_camids[i:i + q_batch.size(0)],
                g_camids,
                self.max_rank
            )

            n = q_batch.size(0)
            if cmc_sum is None:
                cmc_sum = cmc * n
            else:
                cmc_sum += cmc * n
            map_sum += mAP * n
            n_sum += n

        all_cmc = cmc_sum / max(n_sum, 1)
        mAP = map_sum / max(n_sum, 1)
        return all_cmc, mAP, None, self.pids, self.camids, qf, gf




class R1_mAP_eval_LTCC():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=True, block_size=256):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.block_size = block_size

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        qf = feats[:self.num_query]
        gf = feats[self.num_query:]

        q_pids = np.asarray(self.pids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_camids = np.asarray(self.camids[self.num_query:])

        print("=> Computing DistMat with block-wise euclidean distance")

        all_cmc = []
        all_AP = []

        qf = qf.float()
        gf = gf.float()
        gf_norm = torch.sum(gf * gf, dim=1).view(1, -1)

        for i in range(0, qf.size(0), self.block_size):
            q_batch = qf[i:i + self.block_size]
            qb_norm = torch.sum(q_batch * q_batch, dim=1).view(-1, 1)

            dist = qb_norm + gf_norm - 2 * torch.mm(q_batch, gf.t())
            dist = dist.cpu().numpy()

            cmc, mAP = eval_func_LTCC(
                dist,
                q_pids[i:i + q_batch.size(0)],
                g_pids,
                q_camids[i:i + q_batch.size(0)],
                g_camids,
                self.max_rank
            )

            all_cmc.append(cmc)
            all_AP.append(mAP)

        all_cmc = np.mean(np.asarray(all_cmc), axis=0)
        mAP = np.mean(all_AP)

        # 注意：不再返回 distmat（否则又炸）
        return all_cmc, mAP, None, self.pids, self.camids, qf, gf
