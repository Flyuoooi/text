import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .CLIP import clip
from .discriminator import CAADiscriminator
from timm.models.layers import trunc_normal_
from collections import OrderedDict


_tokenizer = _Tokenizer()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    """
    使用CLIP的文本transformer，将 PromptLearner生成的token embedding编码为文本特征 text_features
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Args:
            prompts: (N, L, C)  PromptLearner 生成的 token embedding
            tokenized_prompts: (N, 77)  对应的 token id（用于找到 EOT）
        Returns:
            text_features: (N, C_txt)
        """
        x = prompts + self.positional_embedding.type(self.dtype)

        # NLD -> LND
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # LND -> NLD
        x = x.permute(1, 0, 2)

        x = self.ln_final(x).type(self.dtype)

        # 找到每句话的 EOT 位置
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        return x


def load_clip_to_cpu(backbone_name):
    """
    从clip._MODELS中加载预训练模型，根据输入分辨率构建visual encoder
    """
    if backbone_name not in clip._MODELS:
        raise KeyError(
            f"Backbone '{backbone_name}' not found in clip._MODELS. "
            f"可检查cfg.MODEL.NAME是否与clip._MODELS的key一致"
        )
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, mask_prob=0.5):
        super().__init__()

        self.dtype = dtype
        self.num_class = num_class
        self.mask_prob = mask_prob
        self.token_embedding = token_embedding

        # ====== CoOp-style template with placeholders (CRITICAL for EOT alignment) ======
        n_ctx = 4
        n_cls_ctx = 16

        ctx_init = "a photo of a person " + " ".join(["X"] * n_cls_ctx) + " ."
        tokenized_prompts = clip.tokenize(ctx_init).to(token_embedding.weight.device)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)  # (1,77,C)

        ctx_dim = embedding.size(-1)

        # Find placeholder "X" positions to split prefix/suffix robustly
        with torch.no_grad():
            x_tok = clip.tokenize("X").to(token_embedding.weight.device)[0, 1].item()
            x_pos = (tokenized_prompts[0] == x_tok).nonzero(as_tuple=False).squeeze(-1)
            if x_pos.numel() < n_cls_ctx:
                raise RuntimeError(
                    f"PromptLearner: expected {n_cls_ctx} placeholder X tokens, "
                    f"but found {x_pos.numel()}. Template='{ctx_init}'"
                )
            prefix_len = int(x_pos[0].item())
            suffix_start = int(x_pos[-1].item() + 1)

        # ====== learnable class-specific ctx (N, n_cls_ctx, C) ======
        cls_vectors = torch.empty(
            num_class, n_cls_ctx, ctx_dim,
            dtype=dtype, device=token_embedding.weight.device
        )
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # Save init ctx for debugging/visualization (CoOp learned vs init)
        self.register_buffer("cls_ctx_init", cls_vectors.detach().clone())

        # ====== prefix / suffix tokens (keep EOT in suffix!) ======
        self.register_buffer("token_prefix", embedding[:, :prefix_len, :].to(token_embedding.weight.device))
        self.register_buffer("token_suffix", embedding[:, suffix_start:, :].to(token_embedding.weight.device))
        
        # # ====== 4. Cloth Direction（必须 GPU）=====
        # cloth_words = [
        #     "shirt","t-shirt","coat","jacket","jeans","pants","skirt",
        #     "dress","hoodie","sweater","vest","shorts","long-sleeved",
        #     "short-sleeved","striped","plain","red","blue","green",
        #     "black","white","yellow","pink","brown","gray"
        # ]

        # with torch.no_grad():
        #     cloth_ids = clip.tokenize(cloth_words).to(token_embedding.weight.device)
        #     cloth_emb = token_embedding(cloth_ids).type(dtype)  # (N,77,512)
        #     cloth_emb = cloth_emb.mean(dim=1)
        #     cloth_direction = cloth_emb.mean(dim=0, keepdim=True)
        #     cloth_direction = F.normalize(cloth_direction, dim=-1)

        # self.register_buffer("cloth_direction", cloth_direction)
        # ====== 4. Cloth Direction（必须 GPU）=====
        cloth_words = [
            "shirt","t-shirt","coat","jacket","jeans","pants","skirt",
            "dress","hoodie","sweater","vest","shorts","long-sleeved",
            "short-sleeved","striped","plain","red","blue","green",
            "black","white","yellow","pink","brown","gray"
        ]

        with torch.no_grad():
            cloth_ids = clip.tokenize(cloth_words).to(token_embedding.weight.device)
            emb = token_embedding(cloth_ids).type(dtype) # [N, 77, 512]

            valid_emb_list = []
            for i in range(len(cloth_words)):
                # 找到 EOT (End of Text) 的位置
                eot_idx = cloth_ids[i].argmax().item() 
                
                # 取 SOT 之后、EOT 之前的向量作为该词的特征
                # 通常单词在 index 1，如果是复合词(如t-shirt)会有多个token
                valid_tokens = emb[i, 1:eot_idx, :] 
                valid_emb_list.append(valid_tokens.mean(dim=0)) 
            
            valid_emb_stack = torch.stack(valid_emb_list) # [N, 512]
            
            # 计算所有衣服词的平均方向
            cloth_direction = valid_emb_stack.mean(dim=0, keepdim=True)
            cloth_direction = F.normalize(cloth_direction, dim=-1)

        self.register_buffer("cloth_direction", cloth_direction)

        
    def forward(self, labels, mask_mode="random"):
        """
        labels: LongTensor, (B,)
        mask_mode: "none" / "always" / "random"
        返回: prompts (B, L, C)
        """
        B = labels.size(0)
        device = labels.device

        # 1) prefix / suffix
        prefix = self.token_prefix.expand(B, -1, -1)    # (B, n_ctx, C)
        suffix = self.token_suffix.expand(B, -1, -1)    # (B, L_suf, C)

        # 2) 根据类别取 cls_ctx
        cls_ctx = self.cls_ctx[labels]                  # (B, n_cls_ctx, C)

        # 3) 拼完整 prompt
        prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)  # (B, L, C)

        # if mask_mode == "none":
        #     return prompts

        # # 4) 使用 cloth_direction 做“衣物方向”的投影 mask
        # L, C = prompts.shape[1], prompts.shape[2]
        # P = prompts.reshape(B * L, C)             # (B*L, C)
        # sim = torch.matmul(P, self.cloth_direction.t())  # (B*L, 1)
        # projection = sim * self.cloth_direction          # (B*L, C)

        # if torch.rand(1).item() < 0.01: # 偶尔打印
        #      print(f"[DEBUG] Projection Norm: {projection.norm(dim=-1).mean().item():.4f}, Prompt Norm: {P.norm(dim=-1).mean().item():.4f}")

        # P_masked = P - projection                        # (B*L, C)
        # prompts_masked = P_masked.reshape(B, L, C)

        # if mask_mode == "always":
        #     return prompts_masked

        # if mask_mode == "random":
        #     mask_flag = (torch.rand(B, device=device) < self.mask_prob).float()
        #     mask_flag = mask_flag.view(B, 1, 1)
        #     prompts_final = prompts * (1.0 - mask_flag) + prompts_masked * mask_flag
        #     return prompts_final


        # 3) 先拿出 cls_ctx（只动它）
# cls_ctx: (B, n_cls_ctx, C)

        if mask_mode == "none":
            return torch.cat([prefix, cls_ctx, suffix], dim=1)

        # 4) 仅在 cls_ctx 上做 cloth_direction 投影删除（不要动 suffix/EOT）
        cloth_dir = self.cloth_direction.to(cls_ctx.dtype)          # (1, C)
        B, K, C = cls_ctx.shape
        ctx_flat = cls_ctx.reshape(B * K, C)                        # (B*K, C)

        sim = ctx_flat @ cloth_dir.t()                              # (B*K, 1)
        proj = sim * cloth_dir                                      # (B*K, C)
        ctx_masked = (ctx_flat - proj).reshape(B, K, C)             # (B, K, C)

        prompts_ori = torch.cat([prefix, cls_ctx, suffix], dim=1)
        prompts_masked = torch.cat([prefix, ctx_masked, suffix], dim=1)

        if mask_mode == "always":
            return prompts_masked

        if mask_mode == "random":
            mask_flag = (torch.rand(B, device=labels.device) < self.mask_prob).float().view(B, 1, 1)
            return prompts_ori * (1.0 - mask_flag) + prompts_masked * mask_flag

        return prompts_ori


    def visualize_prompts(self):
        """
        可视化Prompt
        """
        tokenizer = _Tokenizer()
        with torch.no_grad():
            device = self.cls_ctx.device
            N = self.num_class

            prefix = self.token_prefix          # [1,5,512]
            cls_ctx = self.cls_ctx              # [N,4,512]
            suffix = self.token_suffix          # [1,68,512]

            prompts = torch.cat(
                [
                    prefix.expand(N, -1, -1),   
                    cls_ctx,                    
                    suffix.expand(N, -1, -1)    
                ],
                dim=1
            )
            print("=== Final Prompt Tensor Shape ===")
            print(prompts.shape)
            print("\n=== Human-Readable Tokens ===")
            readable_list = []
            raw_template = tokenizer.decode(self.tokenized_prompts[0].cpu().numpy())
            for i in range(N):
                prompt_str = raw_template.replace("person", f"person class {i}")
                ctx_marks = " ".join([f"<CTX-{i}-{j}>" for j in range(cls_ctx.shape[1])])
                prompt_str = prompt_str.replace(" .", f" {ctx_marks} .")
                readable_list.append(prompt_str)
                print(f"\nClass {i}:")
                print(prompt_str)

            return prompts, readable_list
        
    
    @torch.no_grad()
    def visualize_masked_prompts(self, backbone_model, num_classes=5):
        print("\n=== Attention-Guided Masked Prompt Visualization ===")
        print(f"num_classes = {num_classes}")
        device = next(backbone_model.parameters()).device
        dtype = next(backbone_model.parameters()).dtype
        dummy_img = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        _, img_feat, _, feat_proj = backbone_model.dummy_forward_image(dummy_img)

        for cls_id in range(num_classes):
            print(f"Class {cls_id}:")
            prompts = self.forward(
                torch.tensor([cls_id], device=device),
                mask_mode="none"
            )
            prompts_masked, attn_scores = backbone_model.apply_attn_guided_mask(
                prompts, feat_proj
            )
            prefix_len = self.token_prefix.shape[1]
            ctx_len = self.cls_ctx.shape[1]
            suffix_len = self.token_suffix.shape[1]
            readable = []
            for i in range(prefix_len):
                readable.append(f"<P-{i}>")
            for i in range(ctx_len):
                token_emb = prompts_masked[0, prefix_len + i]
                is_masked = torch.all(token_emb == 0)
                if is_masked:
                    readable.append("<MASK>")
                else:
                    readable.append(f"<CTX-{cls_id}-{i}>")
            for i in range(suffix_len):
                readable.append(f"<S-{i}>")
            print(" ".join(readable) + "\n")


    def dump_coop_debug(self, labels, topk=8, max_labels=2, max_ctx_tokens=4):
        import torch
        import torch.nn.functional as F
        if labels is None: return {"error": "labels is None"}
        labels = labels.detach().long().view(-1)
        if labels.numel() == 0: return {"error": "empty labels"}
        labels_show = labels[:max_labels]
        raw_template = _tokenizer.decode(self.tokenized_prompts[0].cpu().numpy())
        ctx_len = self.cls_ctx.shape[1]

        with torch.no_grad():
            ctx_now = self.cls_ctx[labels_show].float()
            ctx_init = self.cls_ctx_init[labels_show].float()
            delta = ctx_now - ctx_init
            delta_norm_mean = delta.norm(dim=-1).mean().item()
            ctx_norm_mean = ctx_now.norm(dim=-1).mean().item()
            cloth_cos = (F.normalize(ctx_now, dim=-1) * self.cloth_direction.float()).sum(dim=-1)
            cloth_cos_max = cloth_cos.max(dim=1).values.detach().cpu().tolist()
            cloth_cos_mean = cloth_cos.mean(dim=1).detach().cpu().tolist()

        with torch.no_grad():
            W = self.token_embedding.weight.float()
            Wn = F.normalize(W, dim=1)
            def topk_tokens(vec):
                v = F.normalize(vec.float(), dim=0)
                sim = Wn @ v
                ids = torch.topk(sim, k=topk).indices
                toks = []
                for tid in ids.detach().cpu().tolist():
                    s = _tokenizer.decode([tid]).replace("\\n", " ").strip()
                    toks.append(s)
                return toks

            out_items = []
            for bi, lab in enumerate(labels_show.detach().cpu().tolist()):
                ctx_vecs = self.cls_ctx[lab].float()
                ctx_mean = ctx_vecs.mean(dim=0)
                mean_top = topk_tokens(ctx_mean)
                per_token = []
                for j in range(min(max_ctx_tokens, ctx_len)):
                    per_token.append({
                        "ctx_token_index": j,
                        "top_tokens": topk_tokens(ctx_vecs[j]),
                    })
                out_items.append({
                    "label": int(lab),
                    "ctx_len": int(ctx_len),
                    "ctx_mean_top_tokens": mean_top,
                    "ctx_tokens_top_tokens": per_token,
                    "cloth_cos_max": float(cloth_cos_max[bi]),
                    "cloth_cos_mean": float(cloth_cos_mean[bi]),
                })

        prompt_structure = raw_template.strip() + " " + " ".join([f"<CTX-{j}>" for j in range(ctx_len)])
        return {
            "template": raw_template,
            "prompt_structure": prompt_structure,
            "stats": {
                "ctx_norm_mean": float(ctx_norm_mean),
                "ctx_delta_norm_mean": float(delta_norm_mean),
            },
            "items": out_items,
        }


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.cat_feat_proj_test = bool(getattr(getattr(cfg, 'TEST', cfg), 'CAT_FEAT_PROJ', False))
        self.mask_prob = float(getattr(cfg.MODEL, "MASK_PROB", 0.5))

        if self.model_name == "ViT-B/16":
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == "RN50":
            self.in_planes = 2048
            self.in_planes_proj = 1024
        else:
            raise ValueError(f"Unsupported CLIP backbone: {self.model_name}")

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        # ---- 分类头 ----
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # ---- BNNeck ----
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        device_id = int(getattr(cfg.MODEL, "DEVICE_ID", 0))
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        clip_model = load_clip_to_cpu(self.model_name)
        clip_model.float()
        clip_model.to(device)
        self.clip_dtype = clip_model.dtype

        self.visual = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.image_encoder = clip_model.visual

        # ---- SIE ----
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_((self.cv_embed), std=0.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        else:
            self.cv_embed = None

        # ---- PromptLearner + TextEncoder ----
        dataset_name = cfg.DATA.DATASET
        self.prompt_learner = PromptLearner(
            num_classes,
            dataset_name,
            clip_model.dtype,
            clip_model.token_embedding,
            mask_prob=self.mask_prob,
        )
        self.text_encoder = TextEncoder(clip_model)

        self.use_attn_mask = bool(getattr(cfg.MODEL, "USE_ATTN_MASK", True))
        self.attn_mask_ratio = float(getattr(cfg.MODEL, "ATTN_MASK_RATIO", 0.5))

        # Attention-guided mask is powerful but can easily become too aggressive.
        # We gate it by probability and allow different scoring strategies.
        self.attn_mask_prob = float(getattr(cfg.MODEL, "ATTN_MASK_PROB", 1.0))
        self.attn_mask_strategy = str(getattr(cfg.MODEL, "ATTN_MASK_STRATEGY", "img_sim")).lower()
        self.attn_mask_select = str(getattr(cfg.MODEL, "ATTN_MASK_SELECT", "top")).lower()  # top / bottom


        # ---- CAA ----
        self.caa_t2v = nn.Linear(self.in_planes_proj, self.in_planes_proj)
        self.caa_v2t = nn.Linear(self.in_planes_proj, self.in_planes_proj)
        self.caa_gamma = float(getattr(cfg.MODEL, "CAA_GAMMA", 0.5))
        self.disc_r_v = CAADiscriminator(dim=self.in_planes_proj)
        self.disc_r_t = CAADiscriminator(dim=self.in_planes_proj)

    def apply_attn_guided_mask(self, prompts, image_feat=None):
        """Mask a subset of *ctx tokens* in the prompt.

        Why this exists in your project:
        - 你的核心目标是“衣物不变身份表征”。Prompt 的 cls_ctx 里会逐渐形成“身份/衣物”子语义。
        - 这里的 masking 应该尽量**只打击衣物相关的 ctx token**，避免把身份锚点一起抹掉。
        - 因此我们提供多种打分策略（img_sim / cloth_sim / hybrid），并支持 top/bottom 选择。

        Args:
            prompts: (B, L, C) full prompt embeddings = [prefix, cls_ctx, suffix]
            image_feat: (B, C) image feature (typically feat_proj) used only when strategy needs it.
        Returns:
            prompts_masked: (B, L, C) masked prompt (only some ctx tokens zeroed)
            scores: (B, L) token scores used for masking (float)
            mask_bool: (B, L) bool mask positions (True means masked)
        """
        B, L, C = prompts.shape
        device = prompts.device

        prefix_len = int(self.prompt_learner.token_prefix.shape[1])
        ctx_len = int(self.prompt_learner.cls_ctx.shape[1])

        # Safety: no ctx -> no mask
        if ctx_len <= 0:
            scores = torch.zeros((B, L), device=device, dtype=torch.float32)
            mask_bool = torch.zeros((B, L), device=device, dtype=torch.bool)
            return prompts, scores, mask_bool

        strategy = getattr(self, "attn_mask_strategy", "img_sim")
        select = getattr(self, "attn_mask_select", "top")

        # --- 1) compute token scores in fp32 for stability ---
        prompts_f = prompts.float()
        prompts_norm = F.normalize(prompts_f, dim=-1, eps=1e-6)  # (B,L,C)

        if strategy in ("img", "img_sim", "image", "image_sim"):
            if image_feat is None:
                raise ValueError("apply_attn_guided_mask: image_feat is required for strategy='img_sim'.")
            img_norm = F.normalize(image_feat.float(), dim=-1, eps=1e-6).unsqueeze(1)  # (B,1,C)
            scores = (prompts_norm * img_norm).sum(dim=-1)  # (B,L)

        elif strategy in ("cloth", "cloth_sim"):
            # Use a stable clothing direction to find clothing-related ctx tokens.
            cloth_dir = self.prompt_learner.cloth_direction.to(device=device).float()  # (1,C)
            cloth_norm = F.normalize(cloth_dir, dim=-1, eps=1e-6).view(1, 1, C)       # (1,1,C)
            scores = (prompts_norm * cloth_norm).sum(dim=-1)  # (B,L)

        elif strategy in ("hybrid", "mix", "img+cloth"):
            if image_feat is None:
                raise ValueError("apply_attn_guided_mask: image_feat is required for strategy='hybrid'.")
            img_norm = F.normalize(image_feat.float(), dim=-1, eps=1e-6).unsqueeze(1)  # (B,1,C)

            cloth_dir = self.prompt_learner.cloth_direction.to(device=device).float()
            cloth_norm = F.normalize(cloth_dir, dim=-1, eps=1e-6).view(1, 1, C)

            s_img = (prompts_norm * img_norm).sum(dim=-1)
            s_clo = (prompts_norm * cloth_norm).sum(dim=-1)

            # normalize each score map to comparable scale (per-sample)
            s_img = (s_img - s_img.mean(dim=1, keepdim=True)) / (s_img.std(dim=1, keepdim=True).clamp_min(1e-6))
            s_clo = (s_clo - s_clo.mean(dim=1, keepdim=True)) / (s_clo.std(dim=1, keepdim=True).clamp_min(1e-6))
            scores = 0.5 * s_img + 0.5 * s_clo
        else:
            # fallback: behave like img_sim
            if image_feat is None:
                raise ValueError(f"apply_attn_guided_mask: unknown strategy='{strategy}', and image_feat is None.")
            img_norm = F.normalize(image_feat.float(), dim=-1, eps=1e-6).unsqueeze(1)
            scores = (prompts_norm * img_norm).sum(dim=-1)

        # --- 2) select ctx range only ---
        ctx_scores = scores[:, prefix_len: prefix_len + ctx_len]  # (B, ctx_len)

        k = max(1, int(float(self.attn_mask_ratio) * ctx_len))
        if select in ("bottom", "low", "min"):
            sel_idx = torch.topk(-ctx_scores, k=k, dim=1).indices
        else:
            sel_idx = torch.topk(ctx_scores, k=k, dim=1).indices

        sel_idx_global = sel_idx + prefix_len  # map to full prompt indices

        # --- 3) build mask and apply ---
        mask_bool = torch.zeros((B, L), device=device, dtype=torch.bool)
        mask_bool.scatter_(1, sel_idx_global, True)

        mask = mask_bool.unsqueeze(-1).type_as(prompts)  # (B,L,1)
        prompts_masked = prompts * (1.0 - mask)
        return prompts_masked, scores, mask_bool


    def forward(
        self,
        x=None,
        label=None,
        clothes_id=None, 
        get_image=False,
        get_text=False,
        cam_label=None,
        view_label=None,
        return_dict=False,
        debug=False,
        debug_topk=8,
        debug_n=2,
        **kwargs
    ):
        # if self.training and x is not None:
        #     if torch.rand(1).item() < 0.7: 
        #         if torch.rand(1).item() < 0.5:
        #             # 变灰
        #             x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        #         else:
        #             # 简单的通道扰动：随机打乱 RGB 通道顺序
        #             # 比如把红衣服变成蓝衣服，强迫 ID Loss 彻底放弃颜色
        #             perm = torch.randperm(3, device=x.device)
        #             x = x[:, perm, :, :]

        if get_text:
            prompts = self.prompt_learner(label, mask_mode="none")
            tokenized = self.prompt_learner.tokenized_prompts.expand(prompts.size(0), -1).to(prompts.device)
            text_features = self.text_encoder(prompts, tokenized)
            return text_features

        if get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == "RN50":
                return image_features_proj[0]
            elif self.model_name == "ViT-B/16":
                return image_features_proj[:, 0]

        # ReID正常前向
        if self.model_name == "RN50":
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]
        elif self.model_name == "ViT-B/16":
            if (cam_label is not None) and (view_label is not None) and (self.cv_embed is not None):
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif (cam_label is not None) and (self.cv_embed is not None):
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif (view_label is not None) and (self.cv_embed is not None):
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
        else:
            raise ValueError(f"Unsupported backbone: {self.model_name}")

        # BNNeck
        feat = self.bottleneck(img_feature)                 # (B, in_planes)
        feat_proj = self.bottleneck_proj(img_feature_proj)  # (B, in_planes_proj)

        # 分类
        cls_score = self.classifier(feat)
        cls_score_proj = self.classifier_proj(feat_proj)
        # ====== Test-time feature selection (robust) ======
        # Why:
        # 1) "cat" 直接拼接会让 main/proj 尺度漂移，导致距离被某一支主导 -> Rank-1 抖动/掉点
        # 2) proj 分支更多受 ITC/CAA 约束，不一定对 ReID 最优；cat 时必须做分支内归一化+尺度平衡
        if not self.training:
            try:
                feat_source = str(getattr(getattr(self.cfg, "TEST", None), "FEAT_SOURCE", "main")).lower()
            except Exception:
                feat_source = "main"

            # pick pre/post-BN features according to cfg.TEST.NECK_FEAT
            if self.neck_feat == "after":
                main_feat = feat              # BN后的 main (768)
                proj_feat = feat_proj         # BN后的 proj (512)
            else:
                main_feat = img_feature       # BN前 main (768)
                proj_feat = img_feature_proj  # BN前 proj (512)

            # --- helper: branch-wise normalize + balance ---
            def _cat_balanced(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # ensure fp32 for stable norm (esp. when amp/autocast)
                a = a.float()
                b = b.float()
                a = F.normalize(a, dim=1, eps=1e-12)
                b = F.normalize(b, dim=1, eps=1e-12)

                # balance two branches so that one doesn't dominate simply due to dim/scale
                # (after concat, evaluator will normalize again if cfg.TEST.FEAT_NORM == "yes")
                da = float(a.size(1))
                db = float(b.size(1))
                a = a * (da ** 0.5)
                b = b * (db ** 0.5)
                return torch.cat([a, b], dim=1)

            # Optional warning: using proj/cat while proj-supervision is off
            # (proj still gets gradients via ITC/CAA, but it may not be optimal for retrieval)
            if feat_source in ["proj", "cat"]:
                try:
                    id_pw = float(getattr(getattr(self.cfg, "MODEL", None), "ID_PROJ_WEIGHT", 0.0))
                    tri_pw = float(getattr(getattr(self.cfg, "MODEL", None), "TRI_PROJ_WEIGHT", 0.0))
                except Exception:
                    id_pw, tri_pw = 0.0, 0.0
                if (id_pw <= 0 and tri_pw <= 0) and not hasattr(self, "_warned_unsup_proj"):
                    print(f"[WARN][FEAT_SOURCE={feat_source}] proj-branch has no ReID supervision "
                        f"(ID_PROJ_WEIGHT/TRI_PROJ_WEIGHT are 0). "
                        f"Using '{feat_source}' may hurt retrieval. Consider FEAT_SOURCE='main' for fair baseline check.")
                    self._warned_unsup_proj = True

            if feat_source == "proj":
                return proj_feat.float()
            if feat_source == "cat":
                # robust cat: branch-wise norm + balanced concat
                return _cat_balanced(main_feat, proj_feat)
            if feat_source == "cat_raw":
                # keep old behavior for debugging
                return torch.cat([main_feat, proj_feat], dim=1)

            # default: main
            return main_feat.float()
        
        # ====== 训练阶段 ======
        if not return_dict:
            # 原始模式
            return [cls_score, cls_score_proj], [
                img_feature_last,
                img_feature,
                img_feature_proj,
            ]
        else:
            # ==========================================================
            # - CE/ID loss 用 BN 后特征 feat / feat_proj
            # - Triplet loss 必须用 BN 前特征 img_feature / img_feature_proj
            #   否则度量空间会被 BN 扭曲，最典型表现就是：
            #     Same-Clothes 很高，但 Clothes-Changing 上不去
            # ==========================================================

            outputs = {
                # ---- main branch ----
                "logits": cls_score,                  # CE 用这个（BN后）
                "bn_features": feat,                  # BN后 768
                "features": img_feature,              # Triplet 用这个（BN前 768）

                # ---- proj branch (optional supervision) ----
                "logits_proj": cls_score_proj,        # CE_proj 用这个（BN后）
                "bn_features_proj": feat_proj,        # BN后 512
                "features_proj": img_feature_proj,    # Triplet_proj 用这个（BN前 512）

                # ---- cross-modal space ----
                # ITC/CAA 用 raw CLIP projection 空间（更贴近 text encoder 输出分布）
                "img_feat_proj": img_feature_proj,    # raw 512
                "txt_feat_proj": None,                # masked text feat (filled later)
                "txt_feat_clean": None,               # clean text feat (optional, for text consistency)

                # ---- raw (debug/analysis) ----
                "img_feature_last": img_feature_last,
                "img_feature": img_feature,
                "img_feature_proj": img_feature_proj,
                "feat_proj": feat_proj,               # keep alias if you used it elsewhere
            }
            outputs["clothes_id"] = clothes_id

            if label is None:
                raise ValueError("return_dict=True requires `label` (pid) for prompt/text branch.")

            # --------------------------
            # Text branch: clean + masked
            # --------------------------
            tokenized = self.prompt_learner.tokenized_prompts.expand(img_feature.size(0), -1).to(img_feature.device)

            # (A) clean prompt (no cloth-direction removal, no attn-mask)
            prompts_clean = self.prompt_learner(label, mask_mode="none")
            txt_clean = self.text_encoder(prompts_clean, tokenized)
            outputs["txt_feat_clean"] = txt_clean

            # (B) masked prompt (your designed cloth-direction suppression + optional attn-guided ctx mask)
            prompts = self.prompt_learner(label, mask_mode="random")

            attn_scores = None
            mask_bool = None

            if self.use_attn_mask:
                prob = float(getattr(self, "attn_mask_prob", 1.0))
                do_mask = (prob >= 1.0) or (torch.rand((), device=prompts.device) < prob)
                if bool(do_mask):
                    # ====== 关键修复：guidance 用 raw img_feature_proj，而不是 BN 后 feat_proj ======
                    prompts, attn_scores, mask_bool = self.apply_attn_guided_mask(
                        prompts,
                        img_feature_proj.detach(),   # <-- IMPORTANT
                    )

            txt_masked = self.text_encoder(prompts, tokenized)
            outputs["txt_feat_proj"] = txt_masked
            outputs["attn_scores"] = attn_scores.detach() if isinstance(attn_scores, torch.Tensor) else None
            outputs["ctx_mask"] = mask_bool.detach() if isinstance(mask_bool, torch.Tensor) else None

            # --------------------------
            # Prompt mask stat (not a real loss)
            # --------------------------
            if self.use_attn_mask and (mask_bool is not None):
                prefix_len = self.prompt_learner.token_prefix.shape[1]
                ctx_len = self.prompt_learner.cls_ctx.shape[1]
                ctx_mask = mask_bool[:, prefix_len:prefix_len + ctx_len]
                outputs["prompt_mask_reg"] = ctx_mask.float().mean().detach()  # should be ~ATTN_MASK_RATIO
            else:
                outputs["prompt_mask_reg"] = torch.tensor(0.0, device=feat.device)

            # --------------------------
            # CAA loss (keep your structure, but use raw proj space for consistency)
            # --------------------------
            feat_v = img_feature_proj          # raw proj
            feat_t = txt_masked                # text proj

            t_to_v = self.caa_t2v(feat_t)
            r_v = feat_v - t_to_v
            v_corr = feat_v - self.caa_gamma * r_v

            v_to_t = self.caa_v2t(feat_v)
            r_t = feat_t - v_to_t
            t_corr = feat_t - self.caa_gamma * r_t

            caa_loss = 0.5 * (
                F.mse_loss(F.normalize(v_corr, dim=1).float(), F.normalize(feat_v, dim=1).float()) +
                F.mse_loss(F.normalize(t_corr, dim=1).float(), F.normalize(feat_t, dim=1).float())
            )
            caa_loss = caa_loss.to(feat.dtype)
            outputs["caa_loss"] = caa_loss

            if debug:
                outputs["debug"] = {
                    "attn_scores": attn_scores.detach() if (attn_scores is not None) else None,
                    "ctx_mask": mask_bool.detach() if (mask_bool is not None) else None,
                    "t_to_v": t_to_v.detach(),
                    "r_v": r_v.detach(),
                    "v_corr": v_corr.detach(),
                    "v_to_t": v_to_t.detach(),
                    "r_t": r_t.detach(),
                    "t_corr": t_corr.detach(),
                }

            return outputs


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))

    def get_text_embeddings(self, masked=False):
        with torch.no_grad():
            N = self.num_classes
            device = next(self.parameters()).device
            prefix = self.prompt_learner.token_prefix.expand(N, -1, -1)
            cls_ctx = self.prompt_learner.cls_ctx
            suffix = self.prompt_learner.token_suffix.expand(N, -1, -1)
            prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)
            if masked:
                mask = torch.rand_like(prompts[:,:,0]) < self.prompt_learner.mask_prob
                prompts = prompts.clone()
                prompts[mask] = 0.0
            tokenized = self.prompt_learner.tokenized_prompts.expand(N, -1)
            text_features = self.text_encoder(prompts, tokenized)
            return text_features.cpu().numpy()
        
    @torch.no_grad()
    def dummy_forward_image(self, img):
        if self.model_name == "RN50":
            image_features_last, image_features, image_features_proj = self.image_encoder(img)
            img_feature_last = F.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(img.shape[0], -1)
            img_feature = F.avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(img.shape[0], -1)
            img_feature_proj = image_features_proj[0]
        elif self.model_name == "ViT-B/16":
            cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(
                img, cv_embed
            )
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
        else:
            raise ValueError(f"Unsupported backbone: {self.model_name}")

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        return img_feature, img_feature_proj, feat, feat_proj

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model