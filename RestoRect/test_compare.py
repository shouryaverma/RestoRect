# flake8: noqa
import os.path as osp
import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

import thop

import time
import archs
import data
import models

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_compare_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # Test with different step counts for RF
    step_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fid_results = {}
    
    for steps in step_counts:
        logger.info(f'Testing with {steps} RF steps...')
        
        # Set the number of steps for this run
        model.set_rf_steps(steps)
        
        start_time = time.time()
        
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name} with {steps} steps...')
            
            fid_score = model.validation_with_fid(test_loader, current_iter=f"{opt['name']}_{steps}steps", 
                                                tb_logger=None, save_img=opt['val']['save_img'])
            
            if test_set_name not in fid_results:
                fid_results[test_set_name] = {}
            fid_results[test_set_name][steps] = fid_score
        
        end_time = time.time()
        validation_time = end_time - start_time
        logger.info(f'Validation completed for {steps} steps in {validation_time:.2f} seconds')
    
    # Print final results
    logger.info("FID Results Summary:")
    logger.info("-" * 50)
    for dataset_name, results in fid_results.items():
        logger.info(f"\n{dataset_name}:")
        logger.info("Steps\tFID")
        for steps, fid in results.items():
            logger.info(f"{steps}\t{fid:.2f}")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
