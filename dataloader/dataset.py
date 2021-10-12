import os
import random
import torch
import dataloader.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
            self,
            img_dir,
            trimap_dir=None,
            seg_img_dir=None,
            seg_mask_dir=None,
            trans=transforms.Compose([transforms.ToTensor()]),
            seg_trans=transforms.Compose([transforms.ToTensor()]),
            sample_size=-1,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.trimap_dir = trimap_dir
        self.seg_img_dir = seg_img_dir
        self.seg_mask_dir = seg_mask_dir
        self.trans = trans
        self.seg_trans = seg_trans
        self.img_names = []
        self.seg_names = []

        for name in os.listdir(self.img_dir):
            self.img_names.append(name)

        if seg_img_dir is not None:
            for name in os.listdir(self.seg_img_dir):
                self.seg_names.append(name)

        if sample_size != -1:
            random.shuffle(self.img_names)
            self.img_names = self.img_names[:sample_size]
            if seg_img_dir is not None:
                random.shuffle(self.seg_names)
                self.seg_names = self.seg_names[:sample_size]
        else:
            if seg_img_dir is not None:
                random.shuffle(self.seg_names)
                self.seg_names = self.seg_names[:len(self.img_names)]  # seg dataset is larger than mat dataset.

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        sample = {'name': img_name}

        # for inference.
        if not self.trimap_dir:
            sample['img'] = self.trans([img])
            return sample

        trimap_path = os.path.join(self.trimap_dir, img_name)
        trimap = Image.open(trimap_path).convert('L')
        sample['img'], sample['trimap'] = self.trans([img, trimap])

        # get 3-channels trimap.
        trimap_3 = sample['trimap'].repeat(3, 1, 1)
        trimap_3[0, :, :] = (trimap_3[0, :, :] <= 0.1).float()
        trimap_3[1, :, :] = ((trimap_3[1, :, :] < 0.9) & (trimap_3[1, :, :] > 0.1)).float()
        trimap_3[2, :, :] = (trimap_3[2, :, :] >= 0.9).float()

        sample['trimap_3'] = trimap_3

        # for val & test.
        if not self.seg_img_dir or not self.seg_mask_dir:
            return sample

        seg_name = self.seg_names[index]

        seg_img_path = os.path.join(self.seg_img_dir, seg_name)
        seg_img = Image.open(seg_img_path).convert('RGB')

        seg_mask_path = os.path.join(self.seg_mask_dir, seg_name.replace('.jpg', '.png'))
        seg_mask = Image.open(seg_mask_path).convert('L')

        sample['seg_img'], sample['seg_mask'] = self.seg_trans([seg_img, seg_mask])

        # get 2-channels trimap.
        trimap_2 = sample['trimap'].repeat(2, 1, 1)
        trimap_2[0, :, :] = (trimap_2[0, :, :] < 0.9).float()   # B & U
        trimap_2[1, :, :] = (trimap_2[1, :, :] >= 0.9).float()  # F

        sample['trimap_2'] = trimap_2

        # get 2-channels seg mask.
        seg_mask_2 = sample['seg_mask'].repeat(2, 1, 1)
        seg_mask_2[0, :, :] = (seg_mask_2[0, :, :] <= 0.1).float()  # B
        seg_mask_2[1, :, :] = (seg_mask_2[1, :, :] > 0.1).float()   # U & F

        sample['seg_mask_2'] = seg_mask_2

        return sample

    def __len__(self):
        return len(self.img_names)


class TrainDataset(BaseDataset):
    def __init__(self, args):
        self.mode = args.mode
        self.patch_size = args.patch_size
        self.hr = args.hr
        super().__init__(
            img_dir=args.img,
            trimap_dir=args.trimap,
            seg_img_dir=args.seg_img,
            seg_mask_dir=args.seg_mask,
            trans=self.__create_transforms(),
            seg_trans=self.__create_seg_transforms(),
            sample_size=args.sample
        )

    def __create_seg_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __create_transforms(self):
        return transforms.Compose([
            transforms.ResizeIfBiggerThan(1600),
            transforms.ResizeIfShortBiggerThan(1080),
            transforms.RandomCrop([640, 960, 1280]),
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class ValDataset(BaseDataset):
    def __init__(self, args):
        self.mode = args.mode
        self.patch_size = args.patch_size
        super().__init__(
            img_dir=args.val_img,
            trimap_dir=args.val_trimap,
            trans=self.__create_transforms(),
        )

    def __create_transforms(self):
        return transforms.Compose([
            # transforms.ResizeIfBiggerThan(self.patch_size),
            transforms.Resize4(1600, 1080),
            # transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class TestDataset(BaseDataset):
    def __init__(self, args):
        self.mode = args.mode
        self.patch_size = args.patch_size
        super().__init__(
            img_dir=args.img,
            trimap_dir=args.trimap,
            trans=self.__create_transforms()
        )

    def __create_transforms(self):
        return transforms.Compose([
            transforms.ResizeIfBiggerThan(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
