import torch
from torchvision.transforms import v2 as T
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset
# from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.samplers import RandomIdentitySampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.Celeb_light import Celeb_light
from data.datasets.last import LaST
from data.datasets.VC_Clothes import VC_Clothes
from torch.utils.data import ConcatDataset, DataLoader
from data.samplers import FastPKSampler


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'celeb_light': Celeb_light,
    'last': LaST,
    'VC_Clothes':VC_Clothes,
}



def get_names():
    """返回支持的数据集名称列表"""
    return list(__factory.keys())

def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():  # 校验数据集名称
        print(config.DATA.DATASET)
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))
    dataset = __factory[config.DATA.DATASET](   # 实例化具体的数据集类（如 PRCC），通常会解析目录结构生成图像列表
        root=config.DATA.ROOT,
        aux_info=config.DATA.AUX_INFO,
        # meta_dir=config.DATA.META_DIR,
        # meta_dims=config.MODEL.META_DIMS[0]
        )

    return dataset


def build_img_transforms(config):
    """构建图像预处理流程 (使用 V2 API)"""
    transform_train = T.Compose([
        T.ToImage(),   # V2 新特性：将输入转为 Tensor (但不缩放 float)，替代 ToTensor
        T.Resize((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH)),   # 强制缩放到指定尺寸
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),  # 随机水平翻转
        T.RandomGrayscale(p=config.AUG.RC_PROB),  # 随机灰度化
        T.RandomResizedCrop((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH),
                            scale=(0.8, 1.0)),  # 随机裁剪缩放：增强增加对尺度变化鲁棒性
        T.ToDtype(torch.float32, scale=True),  # 类型转换并归一化至 [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),   # ImageNet 均值方差归一化
        T.RandomErasing(p=config.AUG.RE_PROB),   # 随机擦除
    ])

    transform_test = T.Compose([
        T.Resize((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    # image dataset
    transform_train, transform_test = build_img_transforms(config)
    train_sampler = FastPKSampler(dataset.train,num_instances=config.DATA.NUM_INSTANCES)
    # 构建训练 DataLoader
    trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train,aux_info=config.DATA.AUX_INFO),
                             sampler=train_sampler,
                             batch_size=config.DATA.BATCH_SIZE, 
                             num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=False, drop_last=True)  # drop_last=True 避免最后一个batch尺寸不一致导致 BatchNorm 出错

    galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                               sampler=RandomIdentitySampler(dataset.gallery),
                               batch_size=config.DATA.TEST_BATCH, 
                               num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=False, drop_last=False, shuffle=False)
    if config.DATA.DATASET == 'prcc':
        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                                 sampler=RandomIdentitySampler(dataset.query_same),
                                batch_size=config.DATA.TEST_BATCH,
                                  num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=False,drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                                 sampler=RandomIdentitySampler(dataset.query_diff),
                                 batch_size=config.DATA.TEST_BATCH,
                                   num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=False, drop_last=False, shuffle=False)

        # 拼接 Query(Diff) 和 Gallery 作为一个大的验证集
        combined_dataset = ConcatDataset([queryloader_diff.dataset, galleryloader.dataset])
        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False)

        # 拼接 Query(Same) 和 Gallery 作为另一个验证集
        combined_dataset_same = ConcatDataset([queryloader_same.dataset, galleryloader.dataset])
        val_loader_same = DataLoader(
            dataset=combined_dataset_same,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same
    else:
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test,aux_info=config.DATA.AUX_INFO),
                                 sampler=RandomIdentitySampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, 
                                 num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=False, drop_last=False, shuffle=False)

        combined_dataset = ConcatDataset([queryloader.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )



        return trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader



    

    
