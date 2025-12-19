import torch

def make_optimizer(cfg, model):
    """Create optimizer with sane parameter grouping.

    Groups:
      - backbone (CLIP visual/text encoders, if trainable) -> BACKBONE_LR_SCALE
      - prompt_learner (CoOp ctx vectors)                 -> PROMPT_LR_SCALE
      - caa / discriminator                              -> CAA_LR_SCALE
      - others (bnneck, classifier, etc.)                 -> BASE_LR
    """

    base_lr = float(cfg.SOLVER.BASE_LR)
    weight_decay = float(cfg.SOLVER.WEIGHT_DECAY)

    backbone_scale = float(getattr(cfg.SOLVER, "BACKBONE_LR_SCALE", 1.0))
    prompt_scale   = float(getattr(cfg.SOLVER, "PROMPT_LR_SCALE", 1.0))
    caa_scale      = float(getattr(cfg.SOLVER, "CAA_LR_SCALE", 1.0))

    groups = {"backbone": [], "prompt": [], "caa": [], "other": []}

    def _decay_for(name: str, p: torch.nn.Parameter) -> float:
        # No decay for bias / norm-like params
        n = name.lower()
        if n.endswith(".bias") or p.ndim == 1 or "bn" in n or "ln" in n or "norm" in n:
            return 0.0
        return weight_decay

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()
        if "prompt_learner" in lname:
            groups["prompt"].append({"params": [p], "lr": base_lr * prompt_scale, "weight_decay": _decay_for(name, p)})
        elif lname.startswith("caa_") or ".caa_" in lname or "disc_" in lname or "discriminator" in lname:
            groups["caa"].append({"params": [p], "lr": base_lr * caa_scale, "weight_decay": _decay_for(name, p)})
        elif lname.startswith("image_encoder") or lname.startswith("visual") or "clip" in lname or "transformer" in lname or "token_embedding" in lname:
            groups["backbone"].append({"params": [p], "lr": base_lr * backbone_scale, "weight_decay": _decay_for(name, p)})
        else:
            groups["other"].append({"params": [p], "lr": base_lr, "weight_decay": _decay_for(name, p)})

    # Merge groups (torch optimizer accepts per-param entries, but we can keep them separate for clarity)
    param_groups = groups["other"] + groups["backbone"] + groups["prompt"] + groups["caa"]

    if len(param_groups) == 0:
        raise ValueError("make_optimizer: no trainable parameters found (all requires_grad=False).")

    optim_name = str(cfg.SOLVER.OPTIMIZER_NAME).lower()
    if optim_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        momentum = float(getattr(cfg.SOLVER, "MOMENTUM", 0.9))
        optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")

    # Light sanity log (won't crash if logger not set up)
    try:
        n_other = len(groups["other"])
        n_backbone = len(groups["backbone"])
        n_prompt = len(groups["prompt"])
        n_caa = len(groups["caa"])
        print(f"[make_optimizer] param groups: other={n_other}, backbone={n_backbone}, prompt={n_prompt}, caa={n_caa}")
    except Exception:
        pass

    return optimizer
