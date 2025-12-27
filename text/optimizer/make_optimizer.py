# import torch

# def make_optimizer(cfg, model):
#     """Create optimizer with sane parameter grouping.

#     Groups:
#       - backbone (CLIP visual/text encoders, if trainable) -> BACKBONE_LR_SCALE
#       - prompt_learner (CoOp ctx vectors)                 -> PROMPT_LR_SCALE
#       - caa / discriminator                              -> CAA_LR_SCALE
#       - others (bnneck, classifier, etc.)                 -> BASE_LR
#     """

#     base_lr = float(cfg.SOLVER.BASE_LR)
#     weight_decay = float(cfg.SOLVER.WEIGHT_DECAY)

#     backbone_scale = float(getattr(cfg.SOLVER, "BACKBONE_LR_SCALE", 1.0))
#     prompt_scale   = float(getattr(cfg.SOLVER, "PROMPT_LR_SCALE", 1.0))
#     caa_scale      = float(getattr(cfg.SOLVER, "CAA_LR_SCALE", 1.0))

#     groups = {"backbone": [], "prompt": [], "caa": [], "other": []}

#     def _decay_for(name: str, p: torch.nn.Parameter) -> float:
#         # No decay for bias / norm-like params
#         n = name.lower()
#         if n.endswith(".bias") or p.ndim == 1 or "bn" in n or "ln" in n or "norm" in n:
#             return 0.0
#         return weight_decay

#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue

#         lname = name.lower()
#         if "prompt_learner" in lname:
#             groups["prompt"].append({"params": [p], "lr": base_lr * prompt_scale, "weight_decay": _decay_for(name, p)})
#         elif lname.startswith("caa_") or ".caa_" in lname or "disc_" in lname or "discriminator" in lname:
#             groups["caa"].append({"params": [p], "lr": base_lr * caa_scale, "weight_decay": _decay_for(name, p)})
#         elif lname.startswith("image_encoder") or lname.startswith("visual") or "clip" in lname or "transformer" in lname or "token_embedding" in lname:
#             groups["backbone"].append({"params": [p], "lr": base_lr * backbone_scale, "weight_decay": _decay_for(name, p)})
#         else:
#             groups["other"].append({"params": [p], "lr": base_lr, "weight_decay": _decay_for(name, p)})

#     # Merge groups (torch optimizer accepts per-param entries, but we can keep them separate for clarity)
#     param_groups = groups["other"] + groups["backbone"] + groups["prompt"] + groups["caa"]

#     if len(param_groups) == 0:
#         raise ValueError("make_optimizer: no trainable parameters found (all requires_grad=False).")

#     optim_name = str(cfg.SOLVER.OPTIMIZER_NAME).lower()
#     if optim_name == "adam":
#         optimizer = torch.optim.Adam(param_groups, lr=base_lr)
#     elif optim_name == "adamw":
#         optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
#     elif optim_name == "sgd":
#         momentum = float(getattr(cfg.SOLVER, "MOMENTUM", 0.9))
#         optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, nesterov=True)
#     else:
#         raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")

#     # Light sanity log (won't crash if logger not set up)
#     try:
#         n_other = len(groups["other"])
#         n_backbone = len(groups["backbone"])
#         n_prompt = len(groups["prompt"])
#         n_caa = len(groups["caa"])
#         print(f"[make_optimizer] param groups: other={n_other}, backbone={n_backbone}, prompt={n_prompt}, caa={n_caa}")
#     except Exception:
#         pass

#     return optimizer


import torch

def make_optimizer(cfg, model):
    """Create optimizer with sane parameter grouping.

    Groups (8 total):
      - other / other_no_decay
      - backbone / backbone_no_decay
      - prompt / prompt_no_decay
      - caa / caa_no_decay
    """

    base_lr = float(cfg.SOLVER.BASE_LR)
    weight_decay = float(cfg.SOLVER.WEIGHT_DECAY)

    backbone_scale = float(getattr(cfg.SOLVER, "BACKBONE_LR_SCALE", 1.0))
    prompt_scale   = float(getattr(cfg.SOLVER, "PROMPT_LR_SCALE", 1.0))
    caa_scale      = float(getattr(cfg.SOLVER, "CAA_LR_SCALE", 1.0))

    lr_map = {
        "backbone": base_lr * backbone_scale,
        "prompt":   base_lr * prompt_scale,
        "caa":      base_lr * caa_scale,
        "other":    base_lr,
    }

    def _is_no_decay(name: str, p: torch.nn.Parameter) -> bool:
        n = name.lower()
        if n.endswith(".bias"):
            return True
        if p.ndim == 1:  # LN/Bn weight usually 1D
            return True
        if "bn" in n or "ln" in n or "norm" in n:
            return True
        if "prompt_learner" in name or "cls_ctx" in name: # <--- 新增
            return True
        return False

    def _which_group(name: str) -> str:
        lname = name.lower()
        if "prompt_learner" in lname:
            return "prompt"
        if lname.startswith("caa_") or ".caa_" in lname or "disc_" in lname or "discriminator" in lname:
            return "caa"
        # 你的原逻辑：把 clip/transformer/token_embedding 都视为 backbone
        if (lname.startswith("image_encoder") or lname.startswith("visual")
            or "clip" in lname or "transformer" in lname or "token_embedding" in lname):
            return "backbone"
        return "other"

    buckets = {
        "backbone": {"decay": [], "no_decay": []},
        "prompt":   {"decay": [], "no_decay": []},
        "caa":      {"decay": [], "no_decay": []},
        "other":    {"decay": [], "no_decay": []},
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = _which_group(name)
        if _is_no_decay(name, p):
            buckets[g]["no_decay"].append(p)
        else:
            buckets[g]["decay"].append(p)

    param_groups = []
    for gname in ["other", "backbone", "prompt", "caa"]:
        lr = lr_map[gname]

        if len(buckets[gname]["decay"]) > 0:
            param_groups.append({
                "name": gname,
                "params": buckets[gname]["decay"],
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if len(buckets[gname]["no_decay"]) > 0:
            param_groups.append({
                "name": f"{gname}_no_decay",
                "params": buckets[gname]["no_decay"],
                "lr": lr,
                "weight_decay": 0.0,
            })

    if len(param_groups) == 0:
        raise ValueError("make_optimizer: no trainable parameters found (all requires_grad=False).")

    optim_name = str(cfg.SOLVER.OPTIMIZER_NAME).lower()
    if optim_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
    elif optim_name == "adamw":
        # 关键：AdamW 的 weight_decay 要用 param_groups 里的，不要再传全局 weight_decay
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=0.0)
    elif optim_name == "sgd":
        momentum = float(getattr(cfg.SOLVER, "MOMENTUM", 0.9))
        optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, nesterov=True, weight_decay=0.0)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")

    # Print summary
    def _numel(ps):
        return int(sum(p.numel() for p in ps))

    try:
        for gname in ["other", "backbone", "prompt", "caa"]:
            n_d = len(buckets[gname]["decay"])
            n_n = len(buckets[gname]["no_decay"])
            p_d = _numel(buckets[gname]["decay"])
            p_n = _numel(buckets[gname]["no_decay"])
            print(f"[make_optimizer] {gname:8s}: decay={n_d:4d}({p_d})  no_decay={n_n:4d}({p_n})  lr={lr_map[gname]:.2e}")
    except Exception:
        pass

    return optimizer
