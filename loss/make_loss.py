import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss
from .orthogonal_loss import OrthogonalLoss


def _masked_logsumexp(x: torch.Tensor, mask: torch.Tensor, dim: int):
    x = x.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(x, dim=dim)


# def itc_pid_proto_loss(image_features, text_features, pids, temperature=0.07, eps=1e-6):
#     """
#     Supervised ITC on PID prototypes:
#     - group samples by pid
#     - average img/txt features within each pid -> prototypes
#     - InfoNCE on P x P (P = #unique pid in batch)
#     """
#     if temperature <= 0:
#         raise ValueError(f"temperature must be > 0, got {temperature}")

#     img = F.normalize(image_features.float(), dim=1, eps=eps)
#     txt = F.normalize(text_features.float(), dim=1, eps=eps)

#     uniq, inv = torch.unique(pids, sorted=True, return_inverse=True)  # inv: [B] -> [0..P-1]
#     P = uniq.numel()
#     B, D = img.shape

#     img_sum = torch.zeros((P, D), device=img.device, dtype=img.dtype)
#     img_sum.index_add_(0, inv, img)
#     img_cnt = torch.bincount(inv, minlength=P).clamp_min(1).unsqueeze(1).to(img.dtype)
#     img_proto = img_sum / img_cnt

#     txt_sum = torch.zeros((P, D), device=txt.device, dtype=txt.dtype)
#     txt_sum.index_add_(0, inv, txt)
#     txt_cnt = torch.bincount(inv, minlength=P).clamp_min(1).unsqueeze(1).to(txt.dtype)
#     txt_proto = txt_sum / txt_cnt

#     logits_i2t = (img_proto @ txt_proto.t()) / float(temperature)
#     labels = torch.arange(P, device=img.device)
#     loss_i2t = F.cross_entropy(logits_i2t, labels)
#     loss_t2i = F.cross_entropy(logits_i2t.t(), labels)
#     return 0.5 * (loss_i2t + loss_t2i)
def itc_pid_proto_loss(image_features, text_features, pids, temperature=0.07, eps=1e-6, return_stats=False):
    """
    Supervised ITC on PID prototypes:
    - group samples by pid
    - average img/txt features within each pid -> prototypes
    - InfoNCE on P x P (P = #unique pid in batch)
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    img = F.normalize(image_features.float(), dim=1, eps=eps)
    txt = F.normalize(text_features.float(), dim=1, eps=eps)

    uniq, inv = torch.unique(pids, sorted=True, return_inverse=True)  # inv: [B] -> [0..P-1]
    P = uniq.numel()
    B, D = img.shape

    img_sum = torch.zeros((P, D), device=img.device, dtype=img.dtype)
    img_sum.index_add_(0, inv, img)
    img_cnt = torch.bincount(inv, minlength=P).clamp_min(1).unsqueeze(1).to(img.dtype)
    img_proto = img_sum / img_cnt

    txt_sum = torch.zeros((P, D), device=txt.device, dtype=txt.dtype)
    txt_sum.index_add_(0, inv, txt)
    txt_cnt = torch.bincount(inv, minlength=P).clamp_min(1).unsqueeze(1).to(txt.dtype)
    txt_proto = txt_sum / txt_cnt

    logits_i2t = (img_proto @ txt_proto.t()) / float(temperature)
    labels = torch.arange(P, device=img.device)

    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_i2t.t(), labels)
    loss = 0.5 * (loss_i2t + loss_t2i)

    if not return_stats:
        return loss

    # -------- stats (no grad) --------
    with torch.no_grad():
        pred_i2t = logits_i2t.argmax(dim=1)
        pred_t2i = logits_i2t.t().argmax(dim=1)
        acc_i2t = (pred_i2t == labels).float().mean()
        acc_t2i = (pred_t2i == labels).float().mean()
        itc_acc = 0.5 * (acc_i2t + acc_t2i)

        pos_logit = logits_i2t.diag().mean()
        if P > 1:
            neg_logit = (logits_i2t.sum(dim=1) - logits_i2t.diag()) / (P - 1)
            neg_logit = neg_logit.mean()
        else:
            neg_logit = logits_i2t.new_tensor(0.0)

        stats = {
            "itc_acc": itc_acc,
            "itc_acc_i2t": acc_i2t,
            "itc_acc_t2i": acc_t2i,
            "itc_pos": pos_logit,
            "itc_neg": neg_logit,
            "itc_P": logits_i2t.new_tensor(float(P)),
        }

    return loss, stats


def make_loss(cfg, num_classes, device):
    # -------------------------
    # 1) ID loss
    # -------------------------
    if cfg.MODEL.IF_LABELSMOOTH == "on":
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("Using LabelSmooth CrossEntropy")
    else:
        id_loss_func = F.cross_entropy
        print("Using normal CrossEntropy")

    # -------------------------
    # 2) Triplet loss
    # -------------------------
    if cfg.MODEL.NO_MARGIN:
        triplet_loss_func = TripletLoss()
        print("Using Soft Triplet Loss")
    else:
        margin = cfg.SOLVER.MARGIN
        triplet_loss_func = TripletLoss(margin=margin)
        print(f"Using Triplet Loss margin={margin}")

    # -------------------------
    # 3) SupCon (optional)
    # -------------------------
    use_supcon = hasattr(cfg.MODEL, "USE_SUPCON") and cfg.MODEL.USE_SUPCON
    if use_supcon:
        supcon_loss_func = SupConLoss(device=device)
        print("Using SupConLoss for text-image alignment")
    else:
        supcon_loss_func = None
        print("SupConLoss disabled")

    # -------------------------
    # 4) Weights from cfg
    # -------------------------
    itc_weight = float(getattr(cfg.MODEL, "ITC_LOSS_WEIGHT", 0.0))
    caa_weight = float(getattr(cfg.MODEL, "CAA_LOSS_WEIGHT", 0.0))
    caa_temp = float(getattr(cfg.MODEL, "CAA_T", 0.07))

    # proj supervision (STRICTLY follow cfg; NO auto-enable!)
    id_proj_weight = float(getattr(cfg.MODEL, "ID_PROJ_WEIGHT", 0.0))
    tri_proj_weight = float(getattr(cfg.MODEL, "TRI_PROJ_WEIGHT", 0.0))

    # text consistency (masked vs clean)
    txt_cons_w = float(getattr(cfg.MODEL, "TEXT_CONSIST_WEIGHT", 0.0))
    ortho_w = float(getattr(cfg.MODEL, "ORTHO_LOSS_WEIGHT", 0.0))
    use_vis_cloth_dir = bool(getattr(cfg.MODEL, "VIS_CLOTH_DIR", False))
    ortho_loss_fn = OrthogonalLoss().to(device)

    if itc_weight > 0 and txt_cons_w <= 0 and caa_weight <= 0:
        raise ValueError(
            "ITC_LOSS_WEIGHT > 0 requires enabling TEXT_CONSIST_WEIGHT or CAA_LOSS_WEIGHT."
        )

    if caa_weight > 0:
        print(f"Using CAA loss (from model outputs), weight={caa_weight}")
    else:
        print("CAA loss disabled (CAA_LOSS_WEIGHT <= 0)")

    if itc_weight > 0:
        print(f"Using ITC loss (PID-proto InfoNCE), weight={itc_weight}, T={caa_temp}")
    else:
        print("ITC loss disabled (ITC_LOSS_WEIGHT <= 0)")

    if id_proj_weight > 0 or tri_proj_weight > 0:
        print(f"Using proj-branch supervision: ID_PROJ_WEIGHT={id_proj_weight}, TRI_PROJ_WEIGHT={tri_proj_weight}")
    else:
        print("Proj-branch supervision disabled (cfg weights are 0)")

    if txt_cons_w > 0:
        print(f"Using text consistency loss (masked vs clean), weight={txt_cons_w}")
    else:
        print("Text consistency loss disabled (TEXT_CONSIST_WEIGHT <= 0)")

    if ortho_w > 0:
        dir_source = "visual" if use_vis_cloth_dir else "text"
        print(f"Using orthogonal loss ({dir_source} cloth direction), weight={ortho_w}")
    else:
        print("Orthogonal loss disabled (ORTHO_LOSS_WEIGHT <= 0)")

    # -------------------------
    # loss closure
    # -------------------------
    def loss_fn(outputs, pid):
        if not isinstance(outputs, dict):
            raise TypeError(f"loss_fn expects dict outputs (use return_dict=True). Got: {type(outputs)}")

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(float(x), device=pid.device)

        losses = {}

        # 1) ID loss (main)
        logits = outputs.get("logits", None)
        id_loss = id_loss_func(logits, pid) if logits is not None else to_tensor(0.0)
        losses["id_loss"] = id_loss

        # 2) Triplet loss (main)  —— 注意：features 必须是 pre-BN（我们在 backbone_prompt 里修好了）
        feat_tri = outputs.get("features", None)
        tri_loss = triplet_loss_func(feat_tri, pid)[0] if feat_tri is not None else to_tensor(0.0)
        losses["triplet_loss"] = tri_loss

        # 3) proj supervision (optional, follow cfg only)
        logits_proj = outputs.get("logits_proj", None)
        if id_proj_weight > 0 and logits_proj is not None:
            id_loss_proj = id_loss_func(logits_proj, pid)
        else:
            id_loss_proj = to_tensor(0.0)
        losses["id_loss_proj"] = id_loss_proj

        feat_tri_proj = outputs.get("features_proj", None)
        if tri_proj_weight > 0 and feat_tri_proj is not None:
            tri_loss_proj = triplet_loss_func(feat_tri_proj, pid)[0]
        else:
            tri_loss_proj = to_tensor(0.0)
        losses["triplet_loss_proj"] = tri_loss_proj

        # 4) ITC loss (optional) — use raw proj space
        img_p = outputs.get("img_feat_proj", None)
        txt_p = outputs.get("txt_feat_proj", None)
        if itc_weight > 0 and img_p is not None and txt_p is not None:
            itc, itc_stats = itc_pid_proto_loss(img_p, txt_p, pid, temperature=caa_temp, return_stats=True)
            losses["itc_loss"] = itc
            # NEW: 记录指标（注意 detach）
            losses["itc_acc"] = itc_stats["itc_acc"].detach()
            losses["itc_pos"] = itc_stats["itc_pos"].detach()
            losses["itc_neg"] = itc_stats["itc_neg"].detach()
            losses["itc_P"]   = itc_stats["itc_P"].detach()
        else:
            itc = to_tensor(0.0)
            losses["itc_loss"] = itc
            # NEW: 保持 key 始终存在，避免打印/写日志时 KeyError
            losses["itc_acc"] = to_tensor(0.0)
            losses["itc_pos"] = to_tensor(0.0)
            losses["itc_neg"] = to_tensor(0.0)
            losses["itc_P"]   = to_tensor(0.0)
            
        # 5) text consistency (optional)
        txt_clean = outputs.get("txt_feat_clean", None)
        if txt_cons_w > 0 and txt_clean is not None and txt_p is not None:
            txt_clean_n = F.normalize(txt_clean.float(), dim=1, eps=1e-6)
            txt_p_n = F.normalize(txt_p.float(), dim=1, eps=1e-6)
            txt_cons = (1.0 - (txt_clean_n * txt_p_n).sum(dim=1)).mean()
        else:
            txt_cons = to_tensor(0.0)
        losses["txt_cons_loss"] = txt_cons

        # 6) prompt mask stat (not a true loss)
        mask_stat = to_tensor(outputs.get("prompt_mask_reg", 0.0))
        losses["mask_loss"] = mask_stat
        losses["mask_ratio"] = mask_stat

        # 6.5) orthogonal loss: push img_feat_proj away from cloth direction
        img_proj = outputs.get("img_feat_proj", None)
        cloth_dir = outputs.get("cloth_direction", None)
        if use_vis_cloth_dir and img_proj is not None:
            clothes_id = outputs.get("clothes_id", None)
            if clothes_id is not None:
                uniq = torch.unique(clothes_id)
                centers = []
                for cid in uniq:
                    mask = clothes_id == cid
                    if mask.any():
                        centers.append(img_proj[mask].mean(dim=0))
                if centers:
                    cloth_dir = torch.stack(centers, dim=0).mean(dim=0, keepdim=True).detach()
        if ortho_w > 0:
            ortho_loss, ortho_stats = ortho_loss_fn(img_proj, cloth_dir, fallback=to_tensor(0.0))
            losses["ortho_cos_mean"] = ortho_stats.get("ortho_cos_mean", to_tensor(0.0))
            losses["ortho_cos_max"] = ortho_stats.get("ortho_cos_max", to_tensor(0.0))
        else:
            ortho_loss = to_tensor(0.0)
            losses["ortho_cos_mean"] = to_tensor(0.0)
            losses["ortho_cos_max"] = to_tensor(0.0)
        losses["ortho_loss"] = ortho_loss

        # 7) CAA loss (from model)
        caa_from_model = to_tensor(outputs.get("caa_loss", 0.0))
        losses["caa_loss"] = caa_from_model.detach()

        # total
        total_loss = (
            id_loss
            + tri_loss
            + id_proj_weight * id_loss_proj
            + tri_proj_weight * tri_loss_proj
            + itc_weight * itc
            + txt_cons_w * txt_cons
            + ortho_w * ortho_loss
            + caa_weight * caa_from_model
        )
        return total_loss, losses

    return loss_fn
