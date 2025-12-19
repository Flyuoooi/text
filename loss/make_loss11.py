# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss


# def caa_loss(image_features, text_features, temperature=0.07):
#     """Stable CLIP-style InfoNCE (used as ITC loss here).
#     Compute in fp32 to avoid AMP overflow/NaNs.
#     """
#     # fp32 for stability
#     img = F.normalize(image_features.float(), dim=1, eps=1e-6)
#     txt = F.normalize(text_features.float(), dim=1, eps=1e-6)

#     # similarity logits
#     logits_i2t = img @ txt.t()
#     logits_t2i = txt @ img.t()

#     temperature = float(temperature)
#     if temperature <= 0:
#         raise ValueError(f"temperature must be > 0, got {temperature}")
#     logits_i2t = logits_i2t / temperature
#     logits_t2i = logits_t2i / temperature

#     labels = torch.arange(logits_i2t.size(0), device=logits_i2t.device)
#     loss_i2t = F.cross_entropy(logits_i2t, labels)
#     loss_t2i = F.cross_entropy(logits_t2i, labels)
#     return 0.5 * (loss_i2t + loss_t2i)
def caa_loss(image_features, text_features, temperature=0.07):
    v = F.normalize(image_features.float(), dim=1, eps=1e-6)
    t = F.normalize(text_features.float(), dim=1, eps=1e-6)
    logits = (v @ t.t()) / float(temperature)

    # 防止极端值进一步炸（可选但很有效）
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = torch.arange(v.size(0), device=v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def itc_pid_loss(img_feat, txt_feat, pid, temperature=0.07):
    img = F.normalize(img_feat.float(), dim=1, eps=1e-6)
    txt = F.normalize(txt_feat.float(), dim=1, eps=1e-6)

    uniq_pid, inv = torch.unique(pid, sorted=True, return_inverse=True)  # inv: [B]
    U, D = uniq_pid.numel(), txt.size(1)

    txt_proto = txt.new_zeros((U, D))
    txt_proto.index_add_(0, inv, txt)

    cnt = txt.new_zeros((U, 1))
    cnt.index_add_(0, inv, torch.ones((txt.size(0), 1), device=txt.device, dtype=txt.dtype))
    txt_proto = F.normalize(txt_proto / cnt.clamp_min(1.0), dim=1, eps=1e-6)

    logits = (img @ txt_proto.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    return F.cross_entropy(logits, inv)




def make_loss(cfg, num_classes, device):
    # ============================================================
    # 1. ID Loss（Label Smooth）
    # ============================================================
    if cfg.MODEL.IF_LABELSMOOTH == "on":
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("Using LabelSmooth CrossEntropy")
    else:
        id_loss_func = F.cross_entropy
        print("Using normal CrossEntropy")

    # ============================================================
    # 2. Triplet Loss
    # ============================================================
    if cfg.MODEL.NO_MARGIN:
        triplet_loss_func = TripletLoss()
        print("Using Soft Triplet Loss")
    else:
        margin = cfg.SOLVER.MARGIN
        triplet_loss_func = TripletLoss(margin=margin)
        print(f"Using Triplet Loss margin={margin}")

    # ============================================================
    # 3. SupConLoss（图文对齐，可选）
    # ============================================================
    use_supcon = hasattr(cfg.MODEL, "USE_SUPCON") and cfg.MODEL.USE_SUPCON
    if use_supcon:
        supcon_loss_func = SupConLoss(device=device)
        print("Using SupConLoss for text-image alignment")
    else:
        supcon_loss_func = None
        print("SupConLoss disabled")

    # ============================================================
    # 4. ITC / CAA 权重
    # ============================================================
    # ITC：image-text consistency（可以用 CLIP-style）
    itc_weight = float(getattr(cfg.MODEL, "ITC_LOSS_WEIGHT", 0.0))

    # CAA：你对抗残差循环那条支路的 loss（由 model.forward 给出）
    caa_weight = float(getattr(cfg.MODEL, "CAA_LOSS_WEIGHT", 0.0))
    caa_temp = float(getattr(cfg.MODEL, "CAA_T", 0.07))

    if caa_weight > 0:
        print(f"Using CAA loss (from model outputs), weight={caa_weight}")
    else:
        print("CAA loss disabled (CAA_LOSS_WEIGHT <= 0)")

    if itc_weight > 0:
        print(f"Using ITC loss (InfoNCE / CLIP-style), weight={itc_weight}, T={caa_temp}")
    else:
        print("ITC loss disabled (ITC_LOSS_WEIGHT <= 0)")

    # ============================================================
    #   真正的 loss 函数：返回 total_loss, loss_dict
    # ============================================================
    def loss_fn(outputs, pid):
        """
        outputs 来自 backbone_prompt forward(return_dict=True)，应包含：
            outputs["logits"]
            outputs["features"]
            outputs["img_feat_proj"]    # [B, D]
            outputs["txt_feat_proj"]    # [B, D]
            outputs.get("prompt_mask_reg")
            outputs.get("caa_loss")
        """
        if not isinstance(outputs, dict):
            raise TypeError(
                f"loss_fn expects model outputs as dict (use return_dict=True). Got: {type(outputs)}"
            )

        # print("features shape =", outputs["features"].shape)
        # print("pid shape =", pid.shape)

        losses = {}

        # 小工具：把标量/0.0 转成 tensor
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(float(x), device=pid.device)

        # ---------------------------------------------------------
        # 1) identity loss
        # ---------------------------------------------------------
        if "logits" in outputs and outputs["logits"] is not None:
            id_loss = id_loss_func(outputs["logits"], pid)
        else:
            id_loss = to_tensor(0.0)
        losses["id_loss"] = id_loss

        # ---------------------------------------------------------
        # 2) triplet loss
        #   TripletLoss 一般返回 (loss, prec), 取第一个
        # ---------------------------------------------------------
        if "features" in outputs and outputs["features"] is not None:
            triplet_loss = triplet_loss_func(outputs["features"], pid)[0]
        else:
            triplet_loss = to_tensor(0.0)
        losses["triplet_loss"] = triplet_loss

        # ---------------------------------------------------------
        # 3) ITC / contrastive loss (CLIP-style InfoNCE)
        #    这里用上面定义的 caa_loss 作为 ITC，实现 image-text 对齐
        # ---------------------------------------------------------
        if (
            itc_weight > 0
            and "img_feat_proj" in outputs
            and "txt_feat_proj" in outputs
            and outputs["img_feat_proj"] is not None
            and outputs["txt_feat_proj"] is not None
        ):
            # itc = caa_loss(
            #     outputs["img_feat_proj"],
            #     outputs["txt_feat_proj"],
            #     temperature=caa_temp,
            # )
            itc = itc_pid_loss(outputs["img_feat_proj"], outputs["txt_feat_proj"], pid, temperature=caa_temp)
        else:
            itc = to_tensor(0.0)
        losses["itc_loss"] = itc

        # ---------------------------------------------------------
        # 4) Prompt mask consistency loss
        # ---------------------------------------------------------
        mask_loss = outputs.get("prompt_mask_reg", 0.0)
        mask_loss = to_tensor(mask_loss)
        losses["mask_loss"] = mask_loss

        # ---------------------------------------------------------
        # 5) CAA residual / 对抗循环那条支路的损失
        #    这里假定 model.forward 已经算好一个标量 outputs["caa_loss"]
        # ---------------------------------------------------------
        caa_from_model = outputs.get("caa_loss", 0.0)
        caa_from_model = to_tensor(caa_from_model)
        if caa_weight > 0:
            caa_from_model = to_tensor(outputs.get("caa_loss", 0.0))
        else:
            caa_from_model = to_tensor(0.0)
        losses["caa_loss"] = caa_from_model.detach()
        # ---------------------------------------------------------
        # 6) 总损失（权重来自 cfg）
        # ---------------------------------------------------------
        total_loss = (
            id_loss
            + triplet_loss
            + itc_weight * itc
            # + 0.5 * mask_loss
            + caa_weight * caa_from_model
        )

        return total_loss, losses

    # 返回闭包 loss_fn，在 train.py 里用：
    # loss_fn = make_loss(cfg, num_class, device)
    return loss_fn
