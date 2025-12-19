import math

class WarmupCosineScheduler:
    """Epoch-level warmup + cosine decay scheduler.

    Call scheduler.step() ONCE per epoch (e.g., at the end of each epoch).
    Works with multiple param groups (each group keeps its own scaled LR).
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr_cfg, warmup_lr_cfg=0.0, min_lr_cfg=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.total_epochs = int(total_epochs)
        self.base_lr_cfg = float(base_lr_cfg)
        self.warmup_lr_cfg = float(warmup_lr_cfg)
        self.min_lr_cfg = float(min_lr_cfg)

        self.epoch = 0
        # Snapshot group base LRs (already include per-group scaling)
        self.group_base_lrs = [float(g.get("lr", self.base_lr_cfg)) for g in optimizer.param_groups]

    def _lr_at(self, epoch, group_base_lr):
        # Map cfg-level warmup/min to this group (preserve scaling)
        if self.base_lr_cfg > 0:
            group_warmup_lr = group_base_lr * (self.warmup_lr_cfg / self.base_lr_cfg)
            group_min_lr = group_base_lr * (self.min_lr_cfg / self.base_lr_cfg)
        else:
            group_warmup_lr = self.warmup_lr_cfg
            group_min_lr = self.min_lr_cfg

        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # linear warmup
            t = float(epoch + 1) / float(self.warmup_epochs)
            return group_warmup_lr + t * (group_base_lr - group_warmup_lr)

        # cosine decay
        if self.total_epochs <= self.warmup_epochs:
            return group_min_lr

        t = float(epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
        t = min(max(t, 0.0), 1.0)
        return group_min_lr + 0.5 * (group_base_lr - group_min_lr) * (1.0 + math.cos(math.pi * t))

    def step(self):
        epoch = self.epoch
        for g, base_lr in zip(self.optimizer.param_groups, self.group_base_lrs):
            g["lr"] = self._lr_at(epoch, base_lr)
        self.epoch += 1

def create_scheduler(cfg, optimizer):
    warmup_epochs = int(getattr(cfg.SOLVER, "WARMUP_EPOCHS", 0))
    total_epochs  = int(getattr(cfg.SOLVER, "EPOCHS", 100))
    base_lr_cfg   = float(getattr(cfg.SOLVER, "BASE_LR", 1e-3))
    warmup_lr_cfg = float(getattr(cfg.SOLVER, "WARMUP_LR", 0.0))
    min_lr_cfg    = float(getattr(cfg.SOLVER, "MIN_LR", 0.0))
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        base_lr_cfg=base_lr_cfg,
        warmup_lr_cfg=warmup_lr_cfg,
        min_lr_cfg=min_lr_cfg,
    )
