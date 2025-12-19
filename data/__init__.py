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
    return list(__factory.keys())

def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        print(config.DATA.DATASET)
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))
    dataset = __factory[config.DATA.DATASET](
        root=config.DATA.ROOT,
        aux_info=config.DATA.AUX_INFO,
        # meta_dir=config.DATA.META_DIR,
        # meta_dims=config.MODEL.META_DIMS[0]
        )

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.ToImage(), 
        T.Resize((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH)),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.RandomGrayscale(p=config.AUG.RC_PROB),
        T.RandomResizedCrop((config.DATA.IMG_HEIGHT, config.DATA.IMG_WIDTH),
                            scale=(0.8, 1.0)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=config.AUG.RE_PROB), 
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
    trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train,aux_info=config.DATA.AUX_INFO),
                             sampler=train_sampler,
                             batch_size=config.DATA.BATCH_SIZE, 
                             num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=False, drop_last=True)

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


        combined_dataset = ConcatDataset([queryloader_diff.dataset, galleryloader.dataset])

        val_loader = DataLoader(
            dataset=combined_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )

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



    

    
