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

#  分词器
_tokenizer = _Tokenizer()


def weights_init_kaiming(m):
    """
    Kaiming 初始化：通常用于卷积层和 BN 层，帮助模型在训练初期快速收敛。
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        # 线性层使用 kaiming_normal，适用于ReLU激活函数
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        # 卷积层使用 fan_in 模式
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        # BN 层权重初始化为 1，偏置为 0
        if m.affine:  # 如果使用仿射变换（可学习的缩放和偏移）
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """
    权重初始化函数，分类器初始化：通常用于最后一层全连接层 (ID Loss 的分类头)
    使用较小的标准差 (0.001) 防止初始 Loss 过大
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)  # 使用正态分布初始化，标准差较小
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    """
    对 CLIP 的 Text Transformer 进行封装
    作用：将 PromptLearner 生成的 Prompt Embedding 编码为最终的文本特征向量
    """
    def __init__(self, clip_model):
        super().__init__()
        # 直接复用 CLIP 预训练模型的组件
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding   # 位置编码
        self.ln_final = clip_model.ln_final  # 最后的LayerNorm层
        self.text_projection = clip_model.text_projection  # 文本投影层
        self.dtype = clip_model.dtype  # 数据类型

    def forward(self, prompts, tokenized_prompts):
        """
        Args:
            prompts: (N, L, C)  PromptLearner 生成的 token embedding
            tokenized_prompts: (N, 77)  对应的 token id（用于找到 EOT）
        Returns:
            text_features: (N, C_txt)
        """
        x = prompts + self.positional_embedding.type(self.dtype)  # 添加位置编码

        # NLD -> LND(batch, seq_len, dim) -> (seq_len, batch, dim)transformer标准输入格式
        x = x.permute(1, 0, 2)
        x = self.transformer(x)   # 通过transformer编码
        # LND -> NLD
        x = x.permute(1, 0, 2)

        x = self.ln_final(x).type(self.dtype)  # 应用最后的LayerNorm

        eot_indices = tokenized_prompts.argmax(dim=-1)  # 找到每句话的 EOT 位置
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection  # 提取EOT位置的向量并通过文本投影层
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
    url = clip._MODELS[backbone_name]  # 获取模型URL
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    try:
        # 尝试加载JIT编译的模型
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:  # 如果JIT加载失败，则直接加载state_dict
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # 构建CLIP模型

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, mask_prob=0.5):
        super().__init__()

        self.dtype = dtype
        self.num_class = num_class
        self.mask_prob = mask_prob
        self.token_embedding = token_embedding  # CLIP的token embedding层

        # ====== CoOp风格的模板，包含占位符（对于EOT对齐至关重要）======
        n_ctx = 4  # 上下文token数量（前缀/后缀中的固定token）
        n_cls_ctx = 16  # 类别特定的上下文token数量（可学习的）

        ctx_init = "a photo of a person " + " ".join(["X"] * n_cls_ctx) + " ."
        tokenized_prompts = clip.tokenize(ctx_init).to(token_embedding.weight.device)   # token化模板
        self.register_buffer("tokenized_prompts", tokenized_prompts)   # 注册为buffer（不参与训练）

        # 获取模板的embedding
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)  # (1,77,C)

        ctx_dim = embedding.size(-1)   # embedding维度

        # 找到占位符"X"的位置，以便稳健地分割前缀/后缀
        with torch.no_grad():
            x_tok = clip.tokenize("X").to(token_embedding.weight.device)[0, 1].item()  # 获取X对应的token id
            x_pos = (tokenized_prompts[0] == x_tok).nonzero(as_tuple=False).squeeze(-1)   # 找到所有X的位置
            if x_pos.numel() < n_cls_ctx:    # 检查找到的占位符数量是否正确
                raise RuntimeError(
                    f"PromptLearner: expected {n_cls_ctx} placeholder X tokens, "
                    f"but found {x_pos.numel()}. Template='{ctx_init}'"
                )
            prefix_len = int(x_pos[0].item())    # 前缀长度（第一个X之前的所有token）
            suffix_start = int(x_pos[-1].item() + 1)   # 后缀开始位置（最后一个X之后）

        # ====== 可学习的类别特定上下文向量 (N, n_cls_ctx, C) ======
        # cls_vectors = torch.empty(
        #     num_class, n_cls_ctx, ctx_dim,
        #     dtype=dtype, device=token_embedding.weight.device
        # )
        # nn.init.normal_(cls_vectors, std=0.02)   # 使用正态分布初始化
        # self.cls_ctx = nn.Parameter(cls_vectors)   # 注册为可学习参数
        # ====== learnable class-specific ctx (N, n_cls_ctx, C) ======

        # 用语义短语初始化，避免纯随机导致 ITC/CAA 初期不稳定
        ctx_init_str = "a photo of a person"
        with torch.no_grad():
            init_ids = clip.tokenize(ctx_init_str).to(token_embedding.weight.device)  # (1,77)
            init_emb = token_embedding(init_ids).type(dtype)[0]                       # (77,C)
            eot = init_ids[0].argmax().item()
            init_tokens = init_emb[1:eot]  # 去掉SOT，到EOT前

            # 如果 token 数不足 n_cls_ctx，则循环补齐；多了就截断
            if init_tokens.size(0) >= n_cls_ctx:
                init_ctx = init_tokens[:n_cls_ctx]
            else:
                repeat = (n_cls_ctx + init_tokens.size(0) - 1) // init_tokens.size(0)
                init_ctx = init_tokens.repeat(repeat, 1)[:n_cls_ctx]

        # 给每个类复制同一份语义 init，再加一点小噪声打破对称
        cls_vectors = init_ctx.unsqueeze(0).repeat(num_class, 1, 1).contiguous()
        cls_vectors = cls_vectors + 0.02 * torch.randn_like(cls_vectors)

        self.cls_ctx = nn.Parameter(cls_vectors)


        # 保存初始上下文向量用于调试/可视化（对比学习前后的变化）
        self.register_buffer("cls_ctx_init", cls_vectors.detach().clone())

        # ====== 保存固定的前缀和后缀 (Prefix / Suffix) ======
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

        # ====== 4. 计算衣物方向向量 (Cloth Direction) ======
        # 为了找到 embedding 空间中代表“衣物”的方向，后续用于正交投影去除衣物信息
        cloth_words = [
            "shirt","t-shirt","coat","jacket","jeans","pants","skirt",
            "dress","hoodie","sweater","vest","shorts","long-sleeved",
            "short-sleeved","striped","plain","red","blue","green",
            "black","white","yellow","pink","brown","gray"
        ]

        with torch.no_grad():
            cloth_ids = clip.tokenize(cloth_words).to(token_embedding.weight.device)  # 将单词 token 化
            emb = token_embedding(cloth_ids).type(dtype) # [N, 77, 512]

            valid_emb_list = []
            for i in range(len(cloth_words)):
                # 找到 EOT (End of Text) 的位置
                eot_idx = cloth_ids[i].argmax().item() 
                
                # 取 SOT 之后、EOT 之前的向量作为该词的特征
                # 通常单词在 index 1，如果是复合词(如t-shirt)会有多个token
                valid_tokens = emb[i, 1:eot_idx, :] 
                valid_emb_list.append(valid_tokens.mean(dim=0))   # 对多 token 的词取平均
            
            valid_emb_stack = torch.stack(valid_emb_list) # [N, 512]
            
            # 计算所有衣物词向量的均值，作为全局衣物方向
            cloth_direction = valid_emb_stack.mean(dim=0, keepdim=True)
            cloth_direction = F.normalize(cloth_direction, dim=-1)  # 归一化

        self.register_buffer("cloth_direction", cloth_direction)  # 注册为 buffer (不参与梯度更新，随模型保存)

        
    def forward(self, labels, mask_mode="random"):
        """
        生成 Prompt Embeddings
        Args:
            labels: (B,) 当前 batch 的行人 ID 标签，用于选取对应的 cls_ctx
            mask_mode: "none" / "always" / "random" -> 控制是否去除衣物特征
        Returns:
            prompts: (B, L, C)
        """
        B = labels.size(0)
        device = labels.device

        # 1) 扩展前缀和后缀到 batch size
        prefix = self.token_prefix.expand(B, -1, -1)    # (B, n_ctx, C)
        suffix = self.token_suffix.expand(B, -1, -1)    # (B, L_suf, C)

        # 2) 根据类别ID labels取对应Learnable Context cls_ctx
        cls_ctx = self.cls_ctx[labels]                  # (B, n_cls_ctx, C)
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


        # 3) 如果不需要 mask，直接拼接返回
        if mask_mode == "none":  
            return torch.cat([prefix, cls_ctx, suffix], dim=1)

        # 4) 正交投影，loth_direction投影删除，去除衣物
        # 仅对 cls_ctx 进行操作，保留 prefix/suffix 语义
        cloth_dir = self.cloth_direction.to(cls_ctx.dtype)          # (1, C)
        B, K, C = cls_ctx.shape
        ctx_flat = cls_ctx.reshape(B * K, C)                        # (B*K, C)

        # 计算 ctx 与衣物方向的相似度投影
        sim = ctx_flat @ cloth_dir.t()                              # (B*K, 1)
        proj = sim * cloth_dir                                      # (B*K, C)
        # 减去衣物分量 -> 得到去除衣物特征后的 ctx
        ctx_masked = (ctx_flat - proj).reshape(B, K, C)             # (B, K, C)

        # 原始prompt和掩码后的prompt
        prompts_ori = torch.cat([prefix, cls_ctx, suffix], dim=1)
        prompts_masked = torch.cat([prefix, ctx_masked, suffix], dim=1)

        if mask_mode == "always":
            return prompts_masked

        if mask_mode == "random":
            mask_flag = (torch.rand(B, device=labels.device) < self.mask_prob).float().view(B, 1, 1) # 生成掩码标志
            return prompts_ori * (1.0 - mask_flag) + prompts_masked * mask_flag  # 根据掩码标志混合原始prompt和掩码prompt

        return prompts_ori  # 默认返回原始prompt（兜底

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
                    prefix.expand(N, -1, -1),   # 扩展前缀
                    cls_ctx,                    # 类别特定上下文
                    suffix.expand(N, -1, -1)    # 扩展后缀
                ],
                dim=1
            )
            print("=== Final Prompt Tensor Shape ===")
            print(prompts.shape)
            print("\n=== Human-Readable Tokens ===")

           
            raw_template = tokenizer.decode(self.tokenized_prompts[0].cpu().numpy())  # 解码原始模板
            readable_list = []
            for i in range(N):   # 为每个类别生成可读的prompt字符串
                prompt_str = raw_template.replace("person", f"person class {i}")
                ctx_marks = " ".join([f"<CTX-{i}-{j}>" for j in range(cls_ctx.shape[1])])
                prompt_str = prompt_str.replace(" .", f" {ctx_marks} .")
                readable_list.append(prompt_str)
                print(f"\nClass {i}:")
                print(prompt_str)

            return prompts, readable_list
        
    
    @torch.no_grad()
    def visualize_masked_prompts(self, backbone_model, num_classes=5):
        """
        可视化注意力引导的掩码prompt
        """
        print("\n=== Attention-Guided Masked Prompt Visualization ===")
        print(f"num_classes = {num_classes}")
        device = next(backbone_model.parameters()).device
        dtype = next(backbone_model.parameters()).dtype
        dummy_img = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)  # 使用虚拟图像进行前向传播
        _, img_feat, _, feat_proj = backbone_model.dummy_forward_image(dummy_img)

        for cls_id in range(num_classes):
            print(f"Class {cls_id}:")
            prompts = self.forward(   # 获取原始prompt
                torch.tensor([cls_id], device=device),
                mask_mode="none"
            )
            prompts_masked, attn_scores = backbone_model.apply_attn_guided_mask(   # 应用注意力引导的掩码
                prompts, feat_proj
            )
            prefix_len = self.token_prefix.shape[1]
            ctx_len = self.cls_ctx.shape[1]
            suffix_len = self.token_suffix.shape[1]

            readable = []  # 生成可读的表示
            for i in range(prefix_len):
                readable.append(f"<P-{i}>")    # 前缀token
            for i in range(ctx_len):
                token_emb = prompts_masked[0, prefix_len + i]
                is_masked = torch.all(token_emb == 0)   # 检查是否被掩码
                if is_masked:
                    readable.append("<MASK>")   # 掩码token
                else:
                    readable.append(f"<CTX-{cls_id}-{i}>")   # 上下文token
            for i in range(suffix_len):
                readable.append(f"<S-{i}>")   # 后缀token
            print(" ".join(readable) + "\n")


    def dump_coop_debug(self, labels, topk=8, max_labels=2, max_ctx_tokens=4):
        """
        调试函数：输出CoOp学习的相关信息
        """
        import torch
        import torch.nn.functional as F
        if labels is None: return {"error": "labels is None"}
        labels = labels.detach().long().view(-1)
        if labels.numel() == 0: return {"error": "empty labels"}
        labels_show = labels[:max_labels]   # 只显示前max_labels个类别

        raw_template = _tokenizer.decode(self.tokenized_prompts[0].cpu().numpy())  # 解码原始模板
        ctx_len = self.cls_ctx.shape[1]

        with torch.no_grad():  # 获取当前和初始的上下文向量
            ctx_now = self.cls_ctx[labels_show].float()
            ctx_init = self.cls_ctx_init[labels_show].float()
            delta = ctx_now - ctx_init   # 变化量

            # 计算统计信息
            delta_norm_mean = delta.norm(dim=-1).mean().item()
            ctx_norm_mean = ctx_now.norm(dim=-1).mean().item()

             # 计算与衣物方向的余弦相似度
            cloth_cos = (F.normalize(ctx_now, dim=-1) * self.cloth_direction.float()).sum(dim=-1)
            cloth_cos_max = cloth_cos.max(dim=1).values.detach().cpu().tolist()
            cloth_cos_mean = cloth_cos.mean(dim=1).detach().cpu().tolist()

        with torch.no_grad():  # 获取CLIP的token embedding权重
            W = self.token_embedding.weight.float()
            Wn = F.normalize(W, dim=1)
            def topk_tokens(vec):
                """找到与给定向量最相似的前k个token"""
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
                mean_top = topk_tokens(ctx_mean)    # 平均向量的topk token
                per_token = []
                for j in range(min(max_ctx_tokens, ctx_len)):
                    # 每个上下文token的topk token
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

        prompt_structure = raw_template.strip() + " " + " ".join([f"<CTX-{j}>" for j in range(ctx_len)])  # 构建prompt结构字符串
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
        self.cos_layer = cfg.MODEL.COS_LAYER   # 是否使用余弦层
        self.neck = cfg.MODEL.NECK  # neck类型
        self.neck_feat = cfg.TEST.NECK_FEAT   # 测试时使用的neck特征
        # 测试时是否拼接投影特征
        self.cat_feat_proj_test = bool(getattr(getattr(cfg, 'TEST', cfg), 'CAT_FEAT_PROJ', False))
        self.mask_prob = float(getattr(cfg.MODEL, "MASK_PROB", 0.5))

        if self.model_name == "ViT-B/16":
            self.in_planes = 768  # 主分支维度
            self.in_planes_proj = 512  # 投影分支维度
        elif self.model_name == "RN50":
            self.in_planes = 2048
            self.in_planes_proj = 1024
        else:
            raise ValueError(f"Unsupported CLIP backbone: {self.model_name}")

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE    # SIE系数

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

        # 计算特征图分辨率
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        device_id = int(getattr(cfg.MODEL, "DEVICE_ID", 0))
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        clip_model = load_clip_to_cpu(self.model_name)
        clip_model.float()
        clip_model.to(device)
        self.clip_dtype = clip_model.dtype

        # 提取CLIP组件
        self.visual = clip_model.visual   # 视觉编码器
        self.token_embedding = clip_model.token_embedding  # token embedding层
        self.ln_final = clip_model.ln_final   # 最后的LayerNorm
        self.text_projection = clip_model.text_projection   # 文本投影层
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.image_encoder = clip_model.visual   # 图像编码器（视觉编码器）

         # ---- SIE（空间交互嵌入）----
        # SIE用于处理相机和视角信息
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_((self.cv_embed), std=0.02)  # 截断正态分布初始化
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

         # 冻结文本编码器和token embedding的参数
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.token_embedding.parameters():
            p.requires_grad = False

        # 注意力掩码相关参数
        self.use_attn_mask = bool(getattr(cfg.MODEL, "USE_ATTN_MASK", True))
        self.attn_mask_ratio = float(getattr(cfg.MODEL, "ATTN_MASK_RATIO", 0.5))

        self.attn_mask_prob = float(getattr(cfg.MODEL, "ATTN_MASK_PROB", 1.0))
        self.attn_mask_strategy = str(getattr(cfg.MODEL, "ATTN_MASK_STRATEGY", "img_sim")).lower()
        self.attn_mask_select = str(getattr(cfg.MODEL, "ATTN_MASK_SELECT", "top")).lower()  # top / bottom


        # ---- CAA ----
        self.caa_t2v = nn.Linear(self.in_planes_proj, self.in_planes_proj)   # 文本到视觉的转换
        self.caa_v2t = nn.Linear(self.in_planes_proj, self.in_planes_proj)   # 视觉到文本的转换
        self.caa_gamma = float(getattr(cfg.MODEL, "CAA_GAMMA", 0.5))   # CAA系数
        self.disc_r_v = CAADiscriminator(dim=self.in_planes_proj)    # 残差鉴别器（视觉）
        self.disc_r_t = CAADiscriminator(dim=self.in_planes_proj)    # 残差鉴别器（文本）

        # ---- Residual correction (identity semantic recovery) ----
        self.txt_resid_mlp = nn.Sequential(
            nn.Linear(self.in_planes_proj, self.in_planes_proj),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_planes_proj, self.in_planes_proj),
        )


    def apply_attn_guided_mask(self, prompts, image_feat=None):
        """
        对prompt中的上下文token用注意力引导的掩码
        
        Args:
            prompts: (B, L, C) 完整的prompt embedding = [prefix, cls_ctx, suffix]
            image_feat: (B, C) 图像特征（通常为feat_proj），仅当策略需要时使用
        Returns:
            prompts_masked: (B, L, C) 掩码后的prompt（只有部分ctx token被置零）
            scores: (B, L) 用于掩码的token分数
            mask_bool: (B, L) 布尔掩码位置（True表示被掩码）
        """
        B, L, C = prompts.shape
        device = prompts.device

         # 获取前缀和上下文长度
        prefix_len = int(self.prompt_learner.token_prefix.shape[1])
        ctx_len = int(self.prompt_learner.cls_ctx.shape[1])

        # 没有上下文则返回原始prompt
        if ctx_len <= 0:
            scores = torch.zeros((B, L), device=device, dtype=torch.float32)
            mask_bool = torch.zeros((B, L), device=device, dtype=torch.bool)
            return prompts, scores, mask_bool

        # 获取掩码策略和选择方式
        strategy = getattr(self, "attn_mask_strategy", "img_sim")
        select = getattr(self, "attn_mask_select", "top")

        # --- 1) 在fp32精度下计算token分数以确保稳定性 ---
        prompts_f = prompts.float()
        prompts_norm = F.normalize(prompts_f, dim=-1, eps=1e-6)  # (B,L,C)

        # 根据策略计算分数
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

        elif strategy in ("hybrid", "mix", "img+cloth"):  # 混合策略：结合图像和衣物相似度
            if image_feat is None:
                raise ValueError("apply_attn_guided_mask: image_feat is required for strategy='hybrid'.")
            img_norm = F.normalize(image_feat.float(), dim=-1, eps=1e-6).unsqueeze(1)  # (B,1,C)

            cloth_dir = self.prompt_learner.cloth_direction.to(device=device).float()
            cloth_norm = F.normalize(cloth_dir, dim=-1, eps=1e-6).view(1, 1, C)

            s_img = (prompts_norm * img_norm).sum(dim=-1)
            s_clo = (prompts_norm * cloth_norm).sum(dim=-1)

            # 对每个分数图进行标准化（按样本）
            s_img = (s_img - s_img.mean(dim=1, keepdim=True)) / (s_img.std(dim=1, keepdim=True).clamp_min(1e-6))
            s_clo = (s_clo - s_clo.mean(dim=1, keepdim=True)) / (s_clo.std(dim=1, keepdim=True).clamp_min(1e-6))
            scores = 0.5 * s_img + 0.5 * s_clo
        else:
            # 回退策略：使用图像相似度
            if image_feat is None:
                raise ValueError(f"apply_attn_guided_mask: unknown strategy='{strategy}', and image_feat is None.")
            img_norm = F.normalize(image_feat.float(), dim=-1, eps=1e-6).unsqueeze(1)
            scores = (prompts_norm * img_norm).sum(dim=-1)

          # --- 2) 仅选择上下文范围 ---
        ctx_scores = scores[:, prefix_len: prefix_len + ctx_len]  # (B, ctx_len)

        k = max(1, int(float(self.attn_mask_ratio) * ctx_len))   # 计算要掩码的token数量
        if select in ("bottom", "low", "min"):   # 根据选择方式选择token
            sel_idx = torch.topk(-ctx_scores, k=k, dim=1).indices   # 选择分数最低的
        else:
            sel_idx = torch.topk(ctx_scores, k=k, dim=1).indices   # 选择分数最高的

        sel_idx_global = sel_idx + prefix_len   # 映射到完整的prompt索引

        # --- 3) 构建掩码并应用 ---
        mask_bool = torch.zeros((B, L), device=device, dtype=torch.bool)
        mask_bool.scatter_(1, sel_idx_global, True)   # 将选中的位置设为True

        mask = mask_bool.unsqueeze(-1).type_as(prompts)  # (B,L,1)
        prompts_masked = prompts * (1.0 - mask)   # 将掩码位置的embedding置零
        return prompts_masked, scores, mask_bool


    def forward(
        self,
        x=None,
        label=None,  # 标签（行人ID）
        clothes_id=None, 
        get_image=False,
        get_text=False,
        cam_label=None,
        view_label=None,
        return_dict=False,   # 是否返回字典格式的结果
        debug=False,
        debug_topk=8,  # 调试时的topk值
        debug_n=2,  # 调试时的样本数量
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
            img_feature_last = nn.functional.avg_pool2d(   # 池化特征
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

            # 使用SIE嵌入
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            # 取[CLS] token作为特征
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
        else:
            raise ValueError(f"Unsupported backbone: {self.model_name}")

        # BNNeck处理
        feat = self.bottleneck(img_feature)                 # (B, in_planes)
        feat_proj = self.bottleneck_proj(img_feature_proj)  # (B, in_planes_proj)BN后投影特征

        # 分类得分
        cls_score = self.classifier(feat)
        cls_score_proj = self.classifier_proj(feat_proj)
        # ====== Test-time feature selection (robust) ======
        # Why:
        # 1) "cat" 直接拼接会让 main/proj 尺度漂移，导致距离被某一支主导 -> Rank-1 抖动/掉点
        # 2) proj 分支更多受 ITC/CAA 约束，不一定对 ReID 最优；cat 时必须做分支内归一化+尺度平衡
        if not self.training:  # 测试模式
            try:
                feat_source = str(getattr(getattr(self.cfg, "TEST", None), "FEAT_SOURCE", "main")).lower()  # 从配置中获取特征来源
            except Exception:
                feat_source = "main"

            # 根据配置选择使用BN前还是BN后的特征
            if self.neck_feat == "after":
                main_feat = feat              # BN后的 main (768)
                proj_feat = feat_proj         # BN后的 proj (512)
            else:
                main_feat = img_feature       # BN前 main (768)
                proj_feat = img_feature_proj  # BN前 proj (512)

            # 辅助函数：分支级归一化 + 平衡拼接
            def _cat_balanced(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # 转换为fp32以确保稳定的归一化
                a = a.float()
                b = b.float()
                a = F.normalize(a, dim=1, eps=1e-12)
                b = F.normalize(b, dim=1, eps=1e-12)

                # 平衡两个分支，避免因维度/尺度导致一个分支主导
                da = float(a.size(1))
                db = float(b.size(1))
                a = a * (da ** 0.5)
                b = b * (db ** 0.5)
                return torch.cat([a, b], dim=1)

            # 可选警告：使用proj/cat而proj分支没有监督
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

             # 根据特征来源返回相应的特征
            if feat_source == "proj":
                return proj_feat.float()
            if feat_source == "cat":
                # 稳健的拼接：分支级归一化 + 平衡拼接
                return _cat_balanced(main_feat, proj_feat)
            if feat_source == "cat_raw":
                # 保持旧行为用于调试
                return torch.cat([main_feat, proj_feat], dim=1)

            # 默认：返回主特征
            return main_feat.float()
        
        # ====== 训练阶段 ======
        if not return_dict:
            # 原始模式 返回分类得分和特征
            return [cls_score, cls_score_proj], [
                img_feature_last,
                img_feature,
                img_feature_proj,
            ]
        else:
            # 字典模式：返回更详细的信息
            # - CE/ID loss 用 BN 后特征 feat / feat_proj
            # - Triplet loss 必须用 BN 前特征 img_feature / img_feature_proj
            #   否则度量空间会被 BN 扭曲，最典型表现就是：
            #     Same-Clothes 很高，但 Clothes-Changing 上不去
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

            # 文本分支：干净 + 掩码版本
            tokenized = self.prompt_learner.tokenized_prompts.expand(img_feature.size(0), -1).to(img_feature.device)

            # (A) 干净prompt（无衣物方向移除，无注意力掩码）
            prompts_clean = self.prompt_learner(label, mask_mode="none")
            txt_clean = self.text_encoder(prompts_clean, tokenized)
            outputs["txt_feat_clean"] = txt_clean

            # (B) 掩码prompt（衣物方向抑制 + 可选的注意力引导上下文掩码）
            prompts = self.prompt_learner(label, mask_mode="random")

            attn_scores = None
            mask_bool = None

            if self.use_attn_mask:   # 应用注意力引导的掩码
                prob = float(getattr(self, "attn_mask_prob", 1.0))
                do_mask = (prob >= 1.0) or (torch.rand((), device=prompts.device) < prob)
                if bool(do_mask):
                    # 使用原始img_feature_proj作为引导，而不是BN后的feat_proj
                    prompts, attn_scores, mask_bool = self.apply_attn_guided_mask(
                        prompts,
                        img_feature_proj.detach(),    # 使用原始特征
                    )

            txt_masked = self.text_encoder(prompts, tokenized)
            outputs["txt_feat_proj"] = txt_masked
            outputs["attn_scores"] = attn_scores.detach() if isinstance(attn_scores, torch.Tensor) else None
            outputs["ctx_mask"] = mask_bool.detach() if isinstance(mask_bool, torch.Tensor) else None

            # --------------------------
            # Residual correction (masked text -> recovered identity semantics)
            # --------------------------
            txt_resid = txt_masked + self.txt_resid_mlp(txt_masked)
            outputs["txt_feat_resid"] = txt_resid

            # Prompt掩码统计（不是真正的损失）
            if self.use_attn_mask and (mask_bool is not None):
                prefix_len = self.prompt_learner.token_prefix.shape[1]
                ctx_len = self.prompt_learner.cls_ctx.shape[1]
                ctx_mask = mask_bool[:, prefix_len:prefix_len + ctx_len]
                outputs["prompt_mask_reg"] = ctx_mask.float().mean().detach()  # should be ATTN_MASK_RATIO
            else:
                outputs["prompt_mask_reg"] = torch.tensor(0.0, device=feat.device)

            # CAA损失（保持原有结构，但使用原始投影空间以确保一致性）
            feat_v = img_feature_proj          # raw proj
            feat_t = txt_masked                # text proj

            # 文本到视觉的转换
            t_to_v = self.caa_t2v(feat_t)
            r_v = feat_v - t_to_v
            v_corr = feat_v - self.caa_gamma * r_v

             # 视觉到文本的转换
            v_to_t = self.caa_v2t(feat_v)
            r_t = feat_t - v_to_t
            t_corr = feat_t - self.caa_gamma * r_t

            # CAA损失：均方误差
            caa_loss = 0.5 * (
                F.mse_loss(F.normalize(v_corr, dim=1).float(), F.normalize(feat_v, dim=1).float()) +
                F.mse_loss(F.normalize(t_corr, dim=1).float(), F.normalize(feat_t, dim=1).float())
            )
            caa_loss = caa_loss.to(feat.dtype)  # 转换为正确的数据类型
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
        # 加载预训练模型参数
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])   # 移除"module."前缀（如果有多GPU训练的前缀）
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        # 微调时加载参数
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))

    def get_text_embeddings(self, masked=False):
        # 获取文本embedding
        with torch.no_grad():
            N = self.num_classes
            device = next(self.parameters()).device
            # 构建所有类别的prompt
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
        

    # 虚拟前向传播（用于调试）
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

        # BN处理
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        return img_feature, img_feature_proj, feat, feat_proj

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model