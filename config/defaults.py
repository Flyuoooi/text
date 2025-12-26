from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
_C.MODEL.PRETRAIN_PATH = ""
# Name of backbone
_C.MODEL.TYPE = ''
# Model name
_C.MODEL.NAME = "ViT-B/16"


# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Dimension of the attribute list
_C.MODEL.META_DIMS = []
_C.MODEL.CLOTH_XISHU = 3
# Add attributes in model, options: 'True', 'False'
_C.MODEL.ADD_META = False
# Mask cloth attributes, options: 'True', 'False'
_C.MODEL.MASK_META = False
# Add cloth embedding only, options: 'True', 'False'
_C.MODEL.CLOTH_ONLY = False
# ID number of GPU
_C.MODEL.DEVICE_ID = 0
# use bnneck
_C.MODEL.NECK = "bnneck"
# 是否使用 SIE（camera/view embedding）
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
_C.MODEL.SIE_COE = 3.0  # SIE 系数
# CLIP visual 的 stride 设置，用来计算特征图大小
# 对 ViT-B-16 / RN50 一般写成 (16, 16)
_C.MODEL.STRIDE_SIZE = (16, 16)
# Prompt 随机 mask 概率（我们实现的模式2）
_C.MODEL.MASK_PROB = 0.5 
# CAA 残差校正强度 γ
_C.MODEL.CAA_GAMMA = 0.5
_C.MODEL.CAA_LOSS_WEIGHT = 0.0
_C.MODEL.ITC_LOSS_WEIGHT = 0.0
_C.MODEL.CAA_T = 0.07

# Text consistency: enforce masked text embedding stay close to clean embedding
#（mask 掉衣物语义但保留身份锚点的必要约束；设为 0 即完全关闭）
_C.MODEL.TEXT_CONSIST_WEIGHT = 0.0
_C.MODEL.ORTHO_LOSS_WEIGHT = 0.0
_C.MODEL.RESID_LOSS_WEIGHT = 0.1

_C.MODEL.USE_ATTN_MASK = True     # attention-guided masking
_C.MODEL.ATTN_MASK_RATIO = 0.5

_C.MODEL.ID_PROJ_WEIGHT=0.0
_C.MODEL.TRI_PROJ_WEIGHT=0.0
_C.MODEL.VIS_CLOTH_DIR = False

# _C.MODEL.TYPE= "no_proj"

_C.MODEL.ATTN_MASK_PROB = 1.0
_C.MODEL.ATTN_MASK_STRATEGY = "img_sim"
_C.MODEL.ATTN_MASK_SELECT = "top"



# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1



# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_HEIGHT = 256
_C.DATA.IMG_WIDTH = 128
_C.DATA.AMPLER = "pk"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4
# Data root
_C.DATA.ROOT = '/data/Data/ReIDData'
# Number of instances
_C.DATA.NUM_INSTANCES = 2 #8
# Batch size during testing
_C.DATA.TEST_BATCH = 128
# Data sampling strategy
_C.DATA.SAMPLER = 'softmax_triplet'
# Extract data containing attributes during data processing, options: 'True', 'False'
_C.DATA.AUX_INFO = False
# Filename containing attributes
# _C.DATA.META_DIR = 'PAR_PETA_105.txt'
_C.DATA.RANDOM_NOISE = False
_C.DATA.RANDOM_PROP = 0.05
# _C.DATA.CLOTH_BALANCE=True

# 兼容backbone_prompt中的 INPUT.SIZE_TRAIN
_C.INPUT = CN()
# 训练图像尺寸与 DATA 保持一致
_C.INPUT.SIZE_TRAIN = (_C.DATA.IMG_HEIGHT, _C.DATA.IMG_WIDTH)


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()

# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is normalized before test
_C.TEST.FEAT_NORM = "yes"
# Test using images only
_C.TEST.TYPE = "image_only"
# 是否在测试阶段取 BNNeck前/后特征：'before' or 'after'
# backbone_prompt在测试时会根据这个字段决定返回feat/img_feature
_C.TEST.NECK_FEAT = "after"
# Whether to use center crop when testing
_C.TEST.CROP = True
# baseline(backbone.py) 测试默认是 cat([main, proj])。
# 为了避免“模型其实没差，只是评估特征选错了”这种坑，这里把默认改成 cat。
# 如需只用主分支/投影分支，可在 yml 里显式设置 TEST.FEAT_SOURCE: "main" / "proj"。
_C.TEST.FEAT_SOURCE="cat"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of epoches
_C.SOLVER.EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.WARMUP_LR = 1e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.AMP_ENABLE = True      # 混合精度开关
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

_C.SOLVER.BACKBONE_LR_SCALE = 0.1
_C.SOLVER.PROMPT_LR_SCALE = 10.0
_C.SOLVER.CAA_LR_SCALE = 1.0
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 60)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.PRINT_FREQ = 20     # 每多少 step 打印一次
_C.SOLVER.AMP_ENABLE = True



# # ---------------------------------------------------------------------------- #
# # TEST
# # ---------------------------------------------------------------------------- #

# _C.TEST = CN()
# # Path to trained model
# _C.TEST.WEIGHT = ""
# # Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
# _C.TEST.FEAT_NORM = 'yes'
# # Test using images only
# _C.TEST.TYPE = 'image_only'
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""


_C.TEST.RERANKING=True