from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,unpaired_paths_from_folder)
from data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation, paired_resize,paired_scale,paired_scale_BAID
from utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
import random
import numpy as np
import torch
import cv2
from os import path as osp
import os
import random

@DATASET_REGISTRY.register()
class DeblurPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(DeblurPairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        if self.opt['dataset_type'] == 'chaos':
            gt_size = self.opt.get('gt_size', None)
            if gt_size is not None:
                img_gt, img_lq = paired_resize(img_gt, img_lq, gt_size)
            if self.opt['phase'] == 'train':
                if self.geometric_augs:
                    img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['dataset_type'] == 'crop_or_resize':
            gt_size = self.opt['gt_size']
            img_h, img_w = img_gt.shape[:2]
            if img_h < gt_size and img_w < gt_size:
                # crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                    gt_path)
            else:
                # resize
                img_gt, img_lq = paired_resize(img_gt, img_lq, gt_size)

        if self.opt['dataset_type'] == 'scale_and_crop':
            scale_rate = self.opt['scale_rate']
            img_gt, img_lq = paired_scale(img_gt, img_lq, scale_rate)

        if self.opt['dataset_type'] == 'BAID':
            img_gt, img_lq = paired_scale_BAID(img_gt, img_lq)

        # augmentation for training
        if self.opt['phase'] == 'train' and self.opt['dataset_type'] != 'chaos' and self.opt[
            'dataset_type'] != 'crop_or_resize':
            try:
                gt_size = self.opt['gt_size']
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                    gt_path)
            except:
                pass
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class UnPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(UnPairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.lq_folder = opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder], ['lq'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder], ['lq'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = unpaired_paths_from_folder(
                self.lq_folder, 'lq',
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq,
                                    bgr2rgb=True,
                                    float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'lq_path': lq_path,
        }

    def __len__(self):
        return len(self.paths)

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        if self.opt['dataset_type'] == 'chaos':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_resize(img_gt, img_lq, gt_size)
            if self.opt['phase'] == 'train':
                if self.geometric_augs:
                    img_gt, img_lq = random_augmentation(img_gt, img_lq)

        if self.opt['dataset_type'] == 'scale_and_crop':
            scale_rate = self.opt['scale_rate']
            img_gt, img_lq = paired_scale(img_gt, img_lq, scale_rate)

        # augmentation for training
        if self.opt['phase'] == 'train' and self.opt['dataset_type'] != 'chaos':
            if self.opt['gt_size'] is not None:
                gt_size = self.opt['gt_size']
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        print(img_lq.size(), img_gt.size())

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class SICEDataset(data.Dataset):
    """SICE dataset for low-light image enhancement.
    
    Train structure:
    - Input images: dataroot_lq/1/, dataroot_lq/2/, etc. (multiple images per scene)
    - GT images: dataroot_gt/1.JPG, dataroot_gt/2.JPG, etc. (one per scene)
    
    Eval structure:  
    - Input images: dataroot_lq/1.jpg, dataroot_lq/2.jpg, etc. (one per scene)
    - GT images: dataroot_gt/1.JPG, dataroot_gt/2.JPG, etc. (one per scene)
    """

    def __init__(self, opt):
        super(SICEDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        
        # Build file pairs
        self.file_pairs = []
        import os
        
        if self.opt['phase'] == 'train':
            # Train: scan numbered subdirectories in lq_folder, match with gt files
            if osp.exists(self.lq_folder) and osp.exists(self.gt_folder):
                lq_subdirs = [d for d in os.listdir(self.lq_folder) 
                             if osp.isdir(osp.join(self.lq_folder, d)) and d.isdigit()]
                
                for subdir in sorted(lq_subdirs, key=int):
                    subdir_path = osp.join(self.lq_folder, subdir)
                    gt_file = osp.join(self.gt_folder, f'{subdir}.JPG')
                    
                    if osp.exists(gt_file):
                        lq_files = [f for f in os.listdir(subdir_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        for lq_file in lq_files:
                            self.file_pairs.append({
                                'lq_path': osp.join(subdir_path, lq_file),
                                'gt_path': gt_file
                            })
        else:
            # Validation: direct 1:1 mapping between test and target files
            if osp.exists(self.lq_folder) and osp.exists(self.gt_folder):
                lq_files = [f for f in os.listdir(self.lq_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for lq_file in sorted(lq_files):
                    base_name = osp.splitext(lq_file)[0]
                    lq_path = osp.join(self.lq_folder, lq_file)
                    
                    # Try different extensions for GT file
                    gt_extensions = ['.JPG', '.jpg', '.png', '.PNG', '.jpeg', '.JPEG']
                    gt_path = None
                    
                    for ext in gt_extensions:
                        candidate_path = osp.join(self.gt_folder, f'{base_name}{ext}')
                        if osp.exists(candidate_path):
                            gt_path = candidate_path
                            break
                    
                    if gt_path:
                        self.file_pairs.append({
                            'lq_path': lq_path,
                            'gt_path': gt_path
                        })

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.file_pairs)
        
        # Load images
        pair = self.file_pairs[index]
        lq_path = pair['lq_path']
        gt_path = pair['gt_path']

        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # Apply dataset-specific processing
        if self.opt.get('dataset_type') == 'SICE':
            img_gt, img_lq = paired_scale_BAID(img_gt, img_lq)

        # Training augmentations
        if self.opt['phase'] == 'train':
            if self.opt.get('gt_size') is not None:
                gt_size = self.opt['gt_size']
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # Convert to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.file_pairs)


@DATASET_REGISTRY.register()
class SSIDDataset(data.Dataset):
    """SSID (Sony Smartphone Image Dataset) for low-light image enhancement.
    
    Dataset structure:
    - Short exposure: dataroot_lq/*.ARW
    - Long exposure: dataroot_gt/*.ARW
    - Split files map short to long exposures directly
    """

    def __init__(self, opt):
        super(SSIDDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.short_folder = opt['dataroot_lq']
        self.long_folder = opt['dataroot_gt']
        
        # Parse split files to get file pairs
        if opt['phase'] == 'train':
            split_file = opt.get('train_list_file', 'Sony_train_list.txt')
        else:
            split_file = opt.get('val_list_file', 'Sony_val_list.txt')
            
        if not osp.isabs(split_file):
            split_file = osp.join(osp.dirname(osp.dirname(opt['dataroot_lq'])), split_file)
        
        self.file_pairs = self._parse_split_file(split_file)
        
        # Validate that we found file pairs
        if len(self.file_pairs) == 0:
            print(f"WARNING: No valid file pairs found for SSID dataset")
            print(f"Short folder: {self.short_folder}")
            print(f"Long folder: {self.long_folder}")
            print(f"Split file: {split_file}")

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def _parse_split_file(self, split_file):
        """Parse Sony split file to get short-long file pairs"""
        file_pairs = []
        import os
        
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}")
            return file_pairs
            
        with open(split_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    print(f"Invalid line {line_num} in split file: {line}")
                    continue
                    
                # Extract full paths directly from split file
                short_path = parts[0].strip()
                long_path = parts[1].strip()
                
                # Check if both files exist before adding to pairs
                if osp.exists(short_path) and osp.exists(long_path):
                    file_pairs.append({
                        'lq_path': short_path,
                        'gt_path': long_path
                    })
                else:
                    print(f"Missing files for line {line_num}: {short_path} or {long_path}")
        
        return file_pairs

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.file_pairs)
        
        # Get file pair
        pair = self.file_pairs[index]
        lq_path = pair['lq_path']
        gt_path = pair['gt_path']

        # Load PNG images directly
        try:
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception as e:
            raise Exception("gt path {} not working: {}".format(gt_path, str(e)))

        try:
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception as e:
            raise Exception("lq path {} not working: {}".format(lq_path, str(e)))

        # Apply dataset-specific processing
        if self.opt.get('dataset_type') == 'SSID':
            img_gt, img_lq = paired_scale_BAID(img_gt, img_lq)

        # Training augmentations
        if self.opt['phase'] == 'train':
            if self.opt.get('gt_size') is not None:
                gt_size = self.opt['gt_size']
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # Convert to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.file_pairs)