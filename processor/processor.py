# import torch
# import time
# import os
# import csv
# import json
# from tqdm import tqdm
# from torch.amp import autocast, GradScaler

# from utils.metrics import (
#     R1_mAP_eval,
#     R1_mAP_eval_LTCC
# )

# # --------------------------
# # Helpers
# # --------------------------
# class AverageMeter:
#     def __init__(self):
#         self.reset()
#     def reset(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0
#     def update(self, val, n=1):
#         val = float(val)
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / max(1, self.count)

# def _get_lr_dict(optimizer):
#     """Try to log per-group LR by name if available; fall back to idx-based keys."""
#     lr_dict = {}
#     for i, pg in enumerate(optimizer.param_groups):
#         name = pg.get("name", None) or pg.get("group", None) or pg.get("tag", None) or f"group{i}"
#         lr_dict[str(name)] = float(pg.get("lr", 0.0))
#     return lr_dict

# def _atomic_torch_save(obj, path):
#     tmp = path + ".tmp"
#     torch.save(obj, tmp)
#     os.replace(tmp, path)

# def _append_jsonl(path, record: dict):
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")

# def _append_csv(path, header, row_dict):
#     write_header = not os.path.exists(path)
#     with open(path, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=header)
#         if write_header:
#             w.writeheader()
#         w.writerow({k: row_dict.get(k, "") for k in header})

# def _pick_primary_mode(metrics_dict):
#     if "General" in metrics_dict:
#         return "General"
#     if "Clothes-Changing" in metrics_dict:
#         return "Clothes-Changing"
#     return next(iter(metrics_dict.keys()))


# def do_train(
#     cfg,
#     model,
#     center_criterion,
#     train_loader,
#     optimizer,
#     optimizer_center,
#     scheduler,
#     loss_fn,
#     local_rank,
#     dataset,
#     val_loader=None,
#     val_loader_same=None,
# ):
#     output_dir = cfg.OUTPUT_DIR
#     os.makedirs(output_dir, exist_ok=True)

#     csv_path = os.path.join(output_dir, "train_log.csv")
#     jsonl_path = os.path.join(output_dir, "train_log.jsonl")

#     # --------------------------
#     # BEST selection configs (original behavior)
#     # --------------------------
#     best_metric_name = getattr(getattr(cfg, "TEST", object()), "BEST_METRIC_NAME", "mAP")
#     best_metric_mode = getattr(getattr(cfg, "TEST", object()), "BEST_METRIC_MODE", None)  # e.g. "General"
#     best_value = -1.0
#     best_epoch = -1

#     # --------------------------
#     # NEW: always save two bests (Rank-1 best & mAP best) for primary_mode
#     # --------------------------
#     best_rank1_value = -1.0
#     best_rank1_epoch = -1
#     best_map_value = -1.0
#     best_map_epoch = -1

#     device = cfg.MODEL.DEVICE
#     model.to(device)
#     scaler = GradScaler(enabled=bool(cfg.SOLVER.AMP_ENABLE))

#     def _count_trainable(params):
#         n = 0
#         for p in params:
#             if p.requires_grad:
#                 n += p.numel()
#         return n

#     print("\n[CHECK] Param groups:")

#     for i, pg in enumerate(optimizer.param_groups):
#         name = pg.get("name", f"group{i}")
#         lr = pg.get("lr", None)
#         num = _count_trainable(pg["params"])
#         print(f"  - {name:10s} | lr={lr} | trainable_params={num}")
#     print()

#     print("====== Training Started ======")

#     for epoch in range(cfg.SOLVER.EPOCHS):
#         if cfg.MODEL.DIST_TRAIN and hasattr(train_loader.sampler, "set_epoch"):
#             train_loader.sampler.set_epoch(epoch)

#         total_meter = AverageMeter()
#         sub_meters = {}

#         model.train()
#         epoch_start = time.time()

#         pbar = tqdm(
#             enumerate(train_loader),
#             total=len(train_loader),
#             desc=f"Epoch[{epoch+1}/{cfg.SOLVER.EPOCHS}]"
#         )

#         for step, batch in pbar:
#             img, pid, camid, clothes_id = batch
#             img = img.to(device, non_blocking=True)
#             pid = pid.to(device, non_blocking=True)
#             camid = camid.to(device, non_blocking=True)
#             clothes_id = clothes_id.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             if optimizer_center is not None:
#                 optimizer_center.zero_grad(set_to_none=True)

#             with autocast(device_type="cuda", enabled=bool(cfg.SOLVER.AMP_ENABLE)):
#                 outputs = model(
#                     x=img, label=pid, cam_label=camid, view_label=None, clothes_id=clothes_id, return_dict=True
#                 )
#                 total_loss, loss_items = loss_fn(outputs, pid)

#             if cfg.SOLVER.AMP_ENABLE:
#                 scaler.scale(total_loss).backward()

#                 # unscale -> clip -> step (important)
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

#                 prev_scale = scaler.get_scale()
#                 scaler.step(optimizer)
#                 scaler.update()

#                 if scaler.get_scale() < prev_scale:
#                     print(f"[WARN] GradScaler scale dropped at epoch={epoch}, iter={step} (inf/nan detected).")
                    
#                 if step < 3:
#                     def grad_norm_of(group_name):
#                         for pg in optimizer.param_groups:
#                             if pg.get("name", "") == group_name:
#                                 s = 0.0; c = 0
#                                 for p in pg["params"]:
#                                     if p.grad is not None:
#                                         s += float(p.grad.data.norm().item()); c += 1
#                                 return s / max(1, c)
#                         return None

#                     print("[GRAD] backbone:", grad_norm_of("backbone"),
#                         "prompt:", grad_norm_of("prompt"),
#                         "caa:", grad_norm_of("caa"),
#                         "other:", grad_norm_of("other"))

#             else:
#                 total_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#                 optimizer.step()

#             if optimizer_center is not None:
#                 for p in center_criterion.parameters():
#                     if p.grad is not None:
#                         p.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
#                 optimizer_center.step()

#             total_meter.update(total_loss.item())
#             for k, v in loss_items.items():
#                 val = v.item() if torch.is_tensor(v) else float(v)
#                 if k not in sub_meters:
#                     sub_meters[k] = AverageMeter()
#                 sub_meters[k].update(val)

#             pbar.set_postfix({"Loss": f"{total_meter.avg:.2f}"})

#             if step < 3:
#                 print("mask_ratio=", loss_items.get("mask_ratio", loss_items.get("mask_loss")))
#                 attn = outputs.get("attn_scores", None)
#                 msk = outputs.get("ctx_mask", None)
#                 if isinstance(attn, torch.Tensor):
#                     print("attn_scores:", float(attn.mean()), float(attn.min()), float(attn.max()))
#                 if isinstance(msk, torch.Tensor):
#                     print("ctx_mask ratio:", float(msk.float().mean()))
#                 dbg = outputs.get("debug", None)
#                 if isinstance(dbg, dict):
#                     for kk in ["attn_scores_sample", "mask_bool_sample"]:
#                         if kk in dbg and isinstance(dbg[kk], torch.Tensor):
#                             t = dbg[kk]
#                             print(kk, float(t.mean()), float(t.min()), float(t.max()))

#         if scheduler is not None:
#             scheduler.step()

#         details = "  ".join([f"{k}: {m.avg:.4f}" for k, m in sub_meters.items()])
#         print(f"Epoch {epoch+1} Done. Avg Loss: {total_meter.avg:.4f}")
#         if details:
#             print("Details: " + details)

#         # --------------------------
#         # Inference
#         # --------------------------
#         metrics = {}
#         if val_loader is not None:
#             metrics = do_inference(cfg, model, dataset, val_loader, val_loader_same)

#         if metrics:
#             primary_mode = best_metric_mode or _pick_primary_mode(metrics)
#             primary = metrics.get(primary_mode, None)
#         else:
#             primary_mode = "N/A"
#             primary = None

#         # --------------------------
#         # Logging
#         # --------------------------
#         lr_dict = _get_lr_dict(optimizer)
#         record = {
#             "epoch": int(epoch + 1),
#             "time_sec": float(time.time() - epoch_start),
#             "avg_loss": float(total_meter.avg),
#             "primary_mode": primary_mode,
#             "lr": lr_dict,
#             "loss_items_avg": {k: float(m.avg) for k, m in sub_meters.items()},
#             "metrics": metrics,
#             # NEW: write best tracking status
#             "best_rank1_value": float(best_rank1_value),
#             "best_rank1_epoch": int(best_rank1_epoch),
#             "best_map_value": float(best_map_value),
#             "best_map_epoch": int(best_map_epoch),
#         }
#         _append_jsonl(jsonl_path, record)

#         rank1_pct = ""
#         map_pct = ""
#         if primary is not None:
#             rank1_pct = primary.get("rank1_pct", "")
#             map_pct = primary.get("mAP_pct", "")

#         def _lk(key):
#             return float(sub_meters[key].avg) if key in sub_meters else ""

#         csv_row = {
#             "epoch": epoch + 1,
#             "avg_loss": float(total_meter.avg),
#             "id_loss": _lk("id_loss"),
#             "triplet_loss": _lk("triplet_loss"),
#             "itc_loss_raw": _lk("itc_loss"),
#             "mask_loss": _lk("mask_loss"),
#             "caa_loss": _lk("caa_loss"),
#             "rank1_pct": rank1_pct,
#             "mAP_pct": map_pct,
#             "lr_backbone": lr_dict.get("backbone", ""),
#             "lr_prompt": lr_dict.get("prompt", ""),
#             "lr_caa": lr_dict.get("caa", ""),
#             "lr_other": lr_dict.get("other", ""),
#         }

#         header = [
#             "epoch", "avg_loss", "id_loss", "triplet_loss", "itc_loss_raw", "mask_loss", "caa_loss",
#             "rank1_pct", "mAP_pct", "lr_backbone", "lr_prompt", "lr_caa", "lr_other"
#         ]
#         _append_csv(csv_path, header, csv_row)

#         # --------------------------
#         # Save last
#         # --------------------------
#         last_path = os.path.join(output_dir, "model_last.pth")
#         _atomic_torch_save({
#             "epoch": epoch + 1,
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
#             "best_metric_name": best_metric_name,
#             "best_metric_mode": (best_metric_mode or primary_mode),
#             "best_value": float(best_value),
#             "best_epoch": int(best_epoch),
#             # NEW: two bests
#             "best_rank1_value": float(best_rank1_value),
#             "best_rank1_epoch": int(best_rank1_epoch),
#             "best_map_value": float(best_map_value),
#             "best_map_epoch": int(best_map_epoch),
#         }, last_path)

#         if (epoch + 1) % int(cfg.SOLVER.CHECKPOINT_PERIOD) == 0:
#             ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
#             _atomic_torch_save({"epoch": epoch + 1, "model": model.state_dict()}, ckpt_path)

#         # --------------------------
#         # Save BEST (original configurable best) + NEW (rank1 best & map best)
#         # --------------------------
#         if primary is not None:
#             # (A) Original best selection (by BEST_METRIC_NAME)
#             current_value = primary.get(best_metric_name + "_pct", None)
#             if current_value is None:
#                 current_value = primary.get("mAP_pct", None)

#             if current_value is not None and float(current_value) > best_value:
#                 best_value = float(current_value)
#                 best_epoch = int(epoch + 1)
#                 best_path = os.path.join(output_dir, "model_best.pth")
#                 _atomic_torch_save({
#                     "epoch": epoch + 1,
#                     "best_metric_name": best_metric_name,
#                     "best_metric_mode": (best_metric_mode or primary_mode),
#                     "best_value": float(best_value),
#                     "best_rank1_pct": float(primary.get("rank1_pct", 0.0)),
#                     "best_mAP_pct": float(primary.get("mAP_pct", 0.0)),
#                     "model": model.state_dict(),
#                 }, best_path)
#                 print(f"[BEST] epoch={epoch+1} mode={best_metric_mode or primary_mode} "
#                       f"Rank-1={primary.get('rank1_pct', 0.0):.2f}% mAP={primary.get('mAP_pct', 0.0):.2f}% -> saved: {best_path}")

#             # (B) NEW: always save best by Rank-1 (primary_mode)
#             r1 = primary.get("rank1_pct", None)
#             if r1 is not None and float(r1) > best_rank1_value:
#                 best_rank1_value = float(r1)
#                 best_rank1_epoch = int(epoch + 1)
#                 best_r1_path = os.path.join(output_dir, "model_best_rank1.pth")
#                 _atomic_torch_save({
#                     "epoch": epoch + 1,
#                     "best_metric_name": "rank1",
#                     "best_metric_mode": (best_metric_mode or primary_mode),
#                     "best_value": float(best_rank1_value),
#                     "best_rank1_pct": float(primary.get("rank1_pct", 0.0)),
#                     "best_mAP_pct": float(primary.get("mAP_pct", 0.0)),
#                     "model": model.state_dict(),
#                 }, best_r1_path)
#                 print(f"[BEST-R1] epoch={epoch+1} mode={best_metric_mode or primary_mode} "
#                       f"Rank-1={primary.get('rank1_pct', 0.0):.2f}% mAP={primary.get('mAP_pct', 0.0):.2f}% -> saved: {best_r1_path}")

#             # (C) NEW: always save best by mAP (primary_mode)
#             mp = primary.get("mAP_pct", None)
#             if mp is not None and float(mp) > best_map_value:
#                 best_map_value = float(mp)
#                 best_map_epoch = int(epoch + 1)
#                 best_map_path = os.path.join(output_dir, "model_best_map.pth")
#                 _atomic_torch_save({
#                     "epoch": epoch + 1,
#                     "best_metric_name": "mAP",
#                     "best_metric_mode": (best_metric_mode or primary_mode),
#                     "best_value": float(best_map_value),
#                     "best_rank1_pct": float(primary.get("rank1_pct", 0.0)),
#                     "best_mAP_pct": float(primary.get("mAP_pct", 0.0)),
#                     "model": model.state_dict(),
#                 }, best_map_path)
#                 print(f"[BEST-mAP] epoch={epoch+1} mode={best_metric_mode or primary_mode} "
#                       f"Rank-1={primary.get('rank1_pct', 0.0):.2f}% mAP={primary.get('mAP_pct', 0.0):.2f}% -> saved: {best_map_path}")

#     print("====== Training Finished ======")



# # --------------------------
# # Inference
# # --------------------------
# @torch.no_grad()
# def do_inference(cfg, model, dataset, val_loader, val_loader_same=None):
#     print("====== Inference ======")
#     device = cfg.MODEL.DEVICE
#     model.to(device)
#     model.eval()

#     ds = cfg.DATA.DATASET.lower()

#     if ds == "ltcc":
#         evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, feat_norm=cfg.TEST.FEAT_NORM)
#         evaluator_general = R1_mAP_eval(dataset.num_query_imgs, feat_norm=cfg.TEST.FEAT_NORM,reranking=cfg.TEST.RERANKING)
#         eval_tasks = [
#             ("Clothes-Changing", val_loader, evaluator_diff, True),
#             ("General", val_loader, evaluator_general, False),
#         ]
#     elif ds == "prcc":
#         evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, feat_norm=cfg.TEST.FEAT_NORM,reranking=cfg.TEST.RERANKING)
#         evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, feat_norm=cfg.TEST.FEAT_NORM,reranking=cfg.TEST.RERANKING)
#         eval_tasks = [
#             ("Clothes-Changing", val_loader, evaluator_diff, False),
#             ("Same-Clothes", val_loader_same, evaluator_same, False),
#         ]
#     else:
#         evaluator = R1_mAP_eval(dataset.num_query_imgs, feat_norm=cfg.TEST.FEAT_NORM,reranking=cfg.TEST.RERANKING)
#         eval_tasks = [
#             ("General", val_loader, evaluator, False),
#         ]

#     metrics = {}

#     for mode_name, loader, evaluator, is_ltcc in eval_tasks:
#         if loader is None:
#             continue

#         print(f"\n--- Evaluating: {mode_name} ---")
#         evaluator.reset()

#         for batch in tqdm(loader, desc=f"Evaluating {mode_name}"):
#             img, pid, camid, clothes_id = batch
#             img = img.to(device, non_blocking=True)

#             with autocast(device_type="cuda", enabled=bool(cfg.SOLVER.AMP_ENABLE)):
#                 feat = model(img)

#             if is_ltcc:
#                 evaluator.update((feat, pid, camid, clothes_id))
#             else:
#                 evaluator.update((feat, pid, camid))

#         cmc, mAP, *_ = evaluator.compute()
#         rank1 = float(cmc[0])   # 0~1
#         mAP = float(mAP)        # 0~1

#         rank1_pct = rank1 * 100.0
#         mAP_pct = mAP * 100.0

#         print(f"[{mode_name}] Rank-1: {rank1_pct:.2f}%, mAP: {mAP_pct:.2f}%")

#         metrics[mode_name] = {
#             "rank1": rank1,
#             "mAP": mAP,
#             "rank1_pct": rank1_pct,
#             "mAP_pct": mAP_pct,
#         }

#         torch.cuda.empty_cache()


#     return metrics


import torch
import time
import os
import csv
import json
from tqdm import tqdm
from contextlib import nullcontext
from torch.amp import autocast, GradScaler

from utils.metrics import (
    R1_mAP_eval,
    R1_mAP_eval_LTCC
)

# --------------------------
# Helpers
# --------------------------
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

def _get_lr_dict(optimizer):
    """Try to log per-group LR by name if available; fall back to idx-based keys."""
    lr_dict = {}
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", None) or pg.get("group", None) or pg.get("tag", None) or f"group{i}"
        lr_dict[str(name)] = float(pg.get("lr", 0.0))
    return lr_dict

def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _append_jsonl(path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _append_csv(path, header, row_dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row_dict.get(k, "") for k in header})

def _pick_primary_mode(metrics_dict):
    if "General" in metrics_dict:
        return "General"
    if "Clothes-Changing" in metrics_dict:
        return "Clothes-Changing"
    return next(iter(metrics_dict.keys()))

def _get_device(cfg):
    dev = str(getattr(getattr(cfg, "MODEL", object()), "DEVICE", "cuda"))
    dev_id = int(getattr(getattr(cfg, "MODEL", object()), "DEVICE_ID", 0))
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{dev_id}")
    return torch.device(dev)

def _best_key_from_name(best_metric_name: str) -> str:
    """Map BEST_METRIC_NAME to our metrics dict keys safely."""
    mk = str(best_metric_name).strip().lower()
    if mk in ["rank-1", "rank1", "r1", "top1", "top-1"]:
        return "rank1_pct"
    if mk in ["map", "m_ap", "meanap", "mAP".lower()]:
        return "mAP_pct"
    # allow direct use like "rank1_pct" / "mAP_pct"
    if mk.endswith("_pct"):
        return mk
    return mk + "_pct"

# --------------------------
# Training
# --------------------------
def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    local_rank,
    dataset,
    val_loader=None,
    val_loader_same=None,
):
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "train_log.csv")
    jsonl_path = os.path.join(output_dir, "train_log.jsonl")

    # BEST selection configs
    best_metric_name = getattr(getattr(cfg, "TEST", object()), "BEST_METRIC_NAME", "mAP")
    best_metric_mode = getattr(getattr(cfg, "TEST", object()), "BEST_METRIC_MODE", None)  # e.g. "General"
    best_value = -1.0
    best_epoch = -1

    # NEW: always save two bests (Rank-1 best & mAP best) for primary_mode
    best_rank1_value = -1.0
    best_rank1_epoch = -1
    best_map_value = -1.0
    best_map_epoch = -1

    device = _get_device(cfg)
    model.to(device)

    amp_enabled = bool(getattr(getattr(cfg, "SOLVER", object()), "AMP_ENABLE", False))
    scaler = GradScaler(enabled=amp_enabled)

    def _count_trainable(params):
        n = 0
        for p in params:
            if p.requires_grad:
                n += p.numel()
        return n

    # ---- Sanity print: param groups ----
    print("\n[CHECK] Param groups:")
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", f"group{i}")
        lr = pg.get("lr", None)
        num = _count_trainable(pg["params"])
        print(f"  - {name:10s} | lr={lr} | trainable_params={num}")
    print()

    # ---- Helper: global grad norm per optimizer group name (robust) ----
    # def _grad_norm_by_group_name(group_name: str):
    #     for pg in optimizer.param_groups:
    #         if pg.get("name", "") == group_name:
    #             s = 0.0
    #             for p in pg["params"]:
    #                 if p.grad is not None:
    #                     g = p.grad.detach()
    #                     s += float(g.norm(2).item() ** 2)
    #             return s ** 0.5
    #     return None

    def grad_norm_of(prefix):
        s = 0.0
        c = 0
        for pg in optimizer.param_groups:
            n = pg.get("name", "")
            if n == prefix or n.startswith(prefix + "_"):
                for p in pg["params"]:
                    if p.grad is not None:
                        s += float(p.grad.data.norm().item())
                        c += 1
        return s / max(1, c)


    print("====== Training Started ======")

    for epoch in range(cfg.SOLVER.EPOCHS):
        if cfg.MODEL.DIST_TRAIN and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        total_meter = AverageMeter()
        sub_meters = {}

        model.train()
        epoch_start = time.time()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch[{epoch+1}/{cfg.SOLVER.EPOCHS}]"
        )

        for step, batch in pbar:
            img, pid, camid, clothes_id = batch
            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            camid = camid.to(device, non_blocking=True)
            clothes_id = clothes_id.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if optimizer_center is not None:
                optimizer_center.zero_grad(set_to_none=True)

            # autocast only on cuda
            cast_ctx = autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with cast_ctx:
                outputs = model(
                    x=img, label=pid, cam_label=camid, view_label=None,
                    clothes_id=clothes_id, return_dict=True
                )
                total_loss, loss_items = loss_fn(outputs, pid)

            # ---- backward (AMP / non-AMP) ----
            if amp_enabled:
                scaler.scale(total_loss).backward()
                # unscale before clip
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            # ---- grad debug for BOTH AMP/non-AMP (关键修复点) ----
            if step < 3:
                print("[GRAD]",
                      "backbone", grad_norm_of("backbone"),
                      "prompt",   grad_norm_of("prompt"),
                      "caa",      grad_norm_of("caa"),
                      "other",    grad_norm_of("other"))
                print("unique pid in batch =", int(torch.unique(pid).numel()))

            # clip + step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)


            # 统计 caa 模块梯度（t2v + v2t）
            def grad_norm(module):
                s = 0.0
                for p in module.parameters():
                    if p.grad is not None:
                        s += p.grad.data.norm(2).item()
                return s

            # 这个打印如果每步都开，会极大拖慢训练（尤其是多卡/大量 iter），
            # 容易造成“同样 epoch 数，实际优化步数更少”的假象，从而看起来跑不过 baseline。
            if step < 3 and hasattr(model, "caa_t2v") and hasattr(model, "caa_v2t"):
                print("grad_caa_t2v:", grad_norm(model.caa_t2v), "grad_caa_v2t:", grad_norm(model.caa_v2t))

            if amp_enabled:
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < prev_scale:
                    print(f"[WARN] GradScaler scale dropped at epoch={epoch}, iter={step} (inf/nan detected).")
            else:
                optimizer.step()

            # center optimizer
            if optimizer_center is not None:
                for p in center_criterion.parameters():
                    if p.grad is not None:
                        p.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                optimizer_center.step()

            total_meter.update(total_loss.item())
            for k, v in loss_items.items():
                val = v.item() if torch.is_tensor(v) else float(v)
                if k not in sub_meters:
                    sub_meters[k] = AverageMeter()
                sub_meters[k].update(val)

            pbar.set_postfix({"Loss": f"{total_meter.avg:.2f}"})

            # ---- quick sanity debug (mask/attn) ----
            if step < 3:
                print("mask_ratio=", loss_items.get("mask_ratio", loss_items.get("mask_loss")))
                attn = outputs.get("attn_scores", None)
                msk = outputs.get("ctx_mask", None)
                if isinstance(attn, torch.Tensor):
                    print("attn_scores:", float(attn.mean()), float(attn.min()), float(attn.max()))
                if isinstance(msk, torch.Tensor):
                    print("ctx_mask ratio:", float(msk.float().mean()))
                dbg = outputs.get("debug", None)
                if isinstance(dbg, dict):
                    for kk in ["attn_scores_sample", "mask_bool_sample"]:
                        if kk in dbg and isinstance(dbg[kk], torch.Tensor):
                            t = dbg[kk]
                            print(kk, float(t.mean()), float(t.min()), float(t.max()))

        if scheduler is not None:
            scheduler.step()

        details = "  ".join([f"{k}: {m.avg:.4f}" for k, m in sub_meters.items()])
        print(f"Epoch {epoch+1} Done. Avg Loss: {total_meter.avg:.4f}")
        if details:
            print("Details: " + details)

        # --------------------------
        # Inference
        # --------------------------
        metrics = {}
        if val_loader is not None:
            metrics = do_inference(cfg, model, dataset, val_loader, val_loader_same)

        if metrics:
            primary_mode = best_metric_mode or _pick_primary_mode(metrics)
            primary = metrics.get(primary_mode, None)
        else:
            primary_mode = "N/A"
            primary = None

        # --------------------------
        # Logging
        # --------------------------
        lr_dict = _get_lr_dict(optimizer)
        record = {
            "epoch": int(epoch + 1),
            "time_sec": float(time.time() - epoch_start),
            "avg_loss": float(total_meter.avg),
            "primary_mode": primary_mode,
            "lr": lr_dict,
            "loss_items_avg": {k: float(m.avg) for k, m in sub_meters.items()},
            "metrics": metrics,
            "best_rank1_value": float(best_rank1_value),
            "best_rank1_epoch": int(best_rank1_epoch),
            "best_map_value": float(best_map_value),
            "best_map_epoch": int(best_map_epoch),
        }
        _append_jsonl(jsonl_path, record)

        rank1_pct = ""
        map_pct = ""
        if primary is not None:
            rank1_pct = primary.get("rank1_pct", "")
            map_pct = primary.get("mAP_pct", "")

        def _lk(key):
            return float(sub_meters[key].avg) if key in sub_meters else ""

        csv_row = {
            "epoch": epoch + 1,
            "avg_loss": float(total_meter.avg),
            "id_loss": _lk("id_loss"),
            "triplet_loss": _lk("triplet_loss"),
            "itc_loss_raw": _lk("itc_loss"),
            "txt_cons_loss": _lk("txt_cons_loss"),
            "mask_loss": _lk("mask_loss"),
            "caa_loss": _lk("caa_loss"),
            "rank1_pct": rank1_pct,
            "mAP_pct": map_pct,
            "lr_backbone": lr_dict.get("backbone", ""),
            "lr_prompt": lr_dict.get("prompt", ""),
            "lr_caa": lr_dict.get("caa", ""),
            "lr_other": lr_dict.get("other", ""),
        }

        header = [
            "epoch", "avg_loss", "id_loss", "triplet_loss", "itc_loss_raw", "txt_cons_loss", "mask_loss", "caa_loss",
            "rank1_pct", "mAP_pct", "lr_backbone", "lr_prompt", "lr_caa", "lr_other"
        ]
        _append_csv(csv_path, header, csv_row)

        # --------------------------
        # Save last
        # --------------------------
        last_path = os.path.join(output_dir, "model_last.pth")
        _atomic_torch_save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "best_metric_name": str(best_metric_name),
            "best_metric_mode": (best_metric_mode or primary_mode),
            "best_value": float(best_value),
            "best_epoch": int(best_epoch),
            "best_rank1_value": float(best_rank1_value),
            "best_rank1_epoch": int(best_rank1_epoch),
            "best_map_value": float(best_map_value),
            "best_map_epoch": int(best_map_epoch),
        }, last_path)

        if (epoch + 1) % int(cfg.SOLVER.CHECKPOINT_PERIOD) == 0:
            ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            _atomic_torch_save({"epoch": epoch + 1, "model": model.state_dict()}, ckpt_path)

        # --------------------------
        # Save BEST (original configurable best) + NEW (rank1 best & map best)
        # --------------------------
        if primary is not None:
            # (A) Original best selection by BEST_METRIC_NAME (robust key mapping)
            key = _best_key_from_name(best_metric_name)
            current_value = primary.get(key, None)
            if current_value is None:
                # fallback: still try mAP
                current_value = primary.get("mAP_pct", None)

            if current_value is not None and float(current_value) > best_value:
                best_value = float(current_value)
                best_epoch = int(epoch + 1)
                best_path = os.path.join(output_dir, "model_best.pth")
                _atomic_torch_save({
                    "epoch": epoch + 1,
                    "best_metric_name": str(best_metric_name),
                    "best_metric_mode": (best_metric_mode or primary_mode),
                    "best_value": float(best_value),
                    "best_rank1_pct": float(primary.get("rank1_pct", 0.0)),
                    "best_mAP_pct": float(primary.get("mAP_pct", 0.0)),
                    "model": model.state_dict(),
                }, best_path)
                print(f"[BEST] epoch={epoch+1} mode={best_metric_mode or primary_mode} "
                      f"Rank-1={primary.get('rank1_pct', 0.0):.2f}% mAP={primary.get('mAP_pct', 0.0):.2f}% -> saved: {best_path}")

            # (B) NEW: always save best by Rank-1 (primary_mode)
            r1 = primary.get("rank1_pct", None)
            if r1 is not None and float(r1) > best_rank1_value:
                best_rank1_value = float(r1)
                best_rank1_epoch = int(epoch + 1)
                best_r1_path = os.path.join(output_dir, "model_best_rank1.pth")
                _atomic_torch_save({
                    "epoch": epoch + 1,
                    "best_metric_name": "rank1",
                    "best_metric_mode": (best_metric_mode or primary_mode),
                    "best_value": float(best_rank1_value),
                    "best_rank1_pct": float(primary.get("rank1_pct", 0.0)),
                    "best_mAP_pct": float(primary.get("mAP_pct", 0.0)),
                    "model": model.state_dict(),
                }, best_r1_path)
                print(f"[BEST-R1] epoch={epoch+1} mode={best_metric_mode or primary_mode} "
                      f"Rank-1={primary.get('rank1_pct', 0.0):.2f}% mAP={primary.get('mAP_pct', 0.0):.2f}% -> saved: {best_r1_path}")

            # (C) NEW: always save best by mAP (primary_mode)
            mp = primary.get("mAP_pct", None)
            if mp is not None and float(mp) > best_map_value:
                best_map_value = float(mp)
                best_map_epoch = int(epoch + 1)
                best_map_path = os.path.join(output_dir, "model_best_map.pth")
                _atomic_torch_save({
                    "epoch": epoch + 1,
                    "best_metric_name": "mAP",
                    "best_metric_mode": (best_metric_mode or primary_mode),
                    "best_value": float(best_map_value),
                    "best_rank1_pct": float(primary.get("rank1_pct", 0.0)),
                    "best_mAP_pct": float(primary.get("mAP_pct", 0.0)),
                    "model": model.state_dict(),
                }, best_map_path)
                print(f"[BEST-mAP] epoch={epoch+1} mode={best_metric_mode or primary_mode} "
                      f"Rank-1={primary.get('rank1_pct', 0.0):.2f}% mAP={primary.get('mAP_pct', 0.0):.2f}% -> saved: {best_map_path}")

    print("====== Training Finished ======")

# --------------------------
# Inference
# --------------------------
@torch.no_grad()
def do_inference(cfg, model, dataset, val_loader, val_loader_same=None):
    print("====== Inference ======")
    device = _get_device(cfg)
    model.to(device)
    model.eval()

    amp_enabled = bool(getattr(getattr(cfg, "SOLVER", object()), "AMP_ENABLE", False))
    ds = cfg.DATA.DATASET.lower()

    if ds == "ltcc":
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
        eval_tasks = [
            ("Clothes-Changing", val_loader, evaluator_diff, True),
            ("General", val_loader, evaluator_general, False),
        ]
    elif ds == "prcc":
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
        eval_tasks = [
            ("Clothes-Changing", val_loader, evaluator_diff, False),
            ("Same-Clothes", val_loader_same, evaluator_same, False),
        ]
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
        eval_tasks = [
            ("General", val_loader, evaluator, False),
        ]

    metrics = {}

    for mode_name, loader, evaluator, is_ltcc in eval_tasks:
        if loader is None:
            continue

        print(f"\n--- Evaluating: {mode_name} ---")
        evaluator.reset()

        for batch in tqdm(loader, desc=f"Evaluating {mode_name}"):
            img, pid, camid, clothes_id = batch
            img = img.to(device, non_blocking=True)

            cast_ctx = autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with cast_ctx:
                feat = model(img)

            # ---- IMPORTANT FIX: move everything to CPU for evaluator stability ----
            feat = feat.float().cpu()
            pid = pid.cpu()
            camid = camid.cpu()
            clothes_id = clothes_id.cpu()

            if is_ltcc:
                evaluator.update((feat, pid, camid, clothes_id))
            else:
                evaluator.update((feat, pid, camid))

        cmc, mAP, *_ = evaluator.compute()
        rank1 = float(cmc[0])   # 0~1
        mAP = float(mAP)        # 0~1

        rank1_pct = rank1 * 100.0
        mAP_pct = mAP * 100.0

        print(f"[{mode_name}] Rank-1: {rank1_pct:.2f}%, mAP: {mAP_pct:.2f}%")

        metrics[mode_name] = {
            "rank1": rank1,
            "mAP": mAP,
            "rank1_pct": rank1_pct,
            "mAP_pct": mAP_pct,
        }

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return metrics
