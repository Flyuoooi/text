import torch

def make_optimizer(cfg, model):
    """
    - PromptLearner 参数 → 更大学习率（通常是base_lr的 3~5倍）
    - CAA / proj_head 参数 → 中 LR
    - Backbone ViT 参数 → 低 LR 或冻结
    """
    base_lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    backbone_params = []
    prompt_params = []
    caa_params = []
    other_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        # PromptLearner
        if "prompt_learner" in name:
            prompt_params.append(param)
            continue

        # CAA 相关模块（proj_head / t2v / v2t / 残差 MLP / 判别器）
        if (
            "proj_head" in name or
            "caa_" in name or
            "disc_r_v" in name or
            "disc_r_t" in name
        ):
            caa_params.append(param)
            continue

        # CLIP backbone（vit）
        if "visual." in name or "transformer" in name:
            backbone_params.append(param)
            continue

        # 其他普通模块
        other_params.append(param)

    # ====== 设置不同学习率 ======
    param_groups = [
        {"params": backbone_params, "lr": base_lr * cfg.SOLVER.BACKBONE_LR_SCALE, "weight_decay": weight_decay},
        {"params": prompt_params,   "lr": base_lr * cfg.SOLVER.PROMPT_LR_SCALE,   "weight_decay": weight_decay},
        {"params": caa_params,      "lr": base_lr * cfg.SOLVER.CAA_LR_SCALE,      "weight_decay": weight_decay},
        {"params": other_params,    "lr": base_lr,                                "weight_decay": weight_decay},
    ]

    if cfg.SOLVER.OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)

    return optimizer
