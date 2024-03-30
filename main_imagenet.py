import os
import random
import argparse
import yaml
import time
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from methods import __dict__ as all_methods
from utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config', default='configs/base.yaml',
        help='setting of Few-shot CLIP')
    parser.add_argument(
        '--dataset_config', default='configs/imagenet.yaml',
        help='dataset config')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.dataset_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main():

    # Load config file
    cfg = get_arguments()

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    method = all_methods[cfg['method']](args=cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    print(preprocess)
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], preprocess)
    classnames = imagenet.classnames

    domain_shift_data = ["imagenetv2", "imagenet_sketch", "imagenet_rendition", "imagenet_adversarial"]
    print(cfg["dataset"])
    if cfg["dataset"] in domain_shift_data:
        print("Preparing target dataset.")
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

        test_loader = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(
            imagenet.test, batch_size=100, num_workers=8, shuffle=False)
        # val_loader = torch.utils.data.DataLoader(
        #     imagenet.val, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(
        imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(
        imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(
        imagenet.classnames, imagenet.template, clip_model)

    # Pre-load test features
    f_test_time = time.time()
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(
        cfg, "test", clip_model, test_loader)
    # val_features, val_labels = pre_load_features(
    #     cfg, "val", clip_model, val_loader)

    total_acc = 0
    predictions = []
    for i in range(cfg['tasks']):
        random.seed(i+1)
        torch.manual_seed(i+1)
        print("Start Training Task:{}".format(str(i+1)))
        few_shot_train_data, few_shot_val_data  = imagenet.generate_fewshot_dataset(cfg['shots'])
        
        if cfg['finetune']:
            train_loader = torch.utils.data.DataLoader(
                few_shot_train_data, batch_size=cfg["batch_size"], num_workers=8, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                few_shot_train_data, batch_size=cfg["batch_size"], num_workers=8, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
                few_shot_val_data, batch_size=cfg["batch_size"], num_workers=8, shuffle=False)
        loss, acc = method(train_loader=train_loader,
                            val_loader=val_loader,
                            test_features=test_features,
                            test_labels=test_labels,
                            text_weights=clip_weights,
                            model=clip_model,
                            classnames=classnames)
        print('Final Accuracy on task {}: {}'.format(str(i+1), acc))
        predictions.append(acc)
    tasks_acc, tasks_std = compute_confidence_interval(predictions)
    test_stats = {}
    test_stats['acc'] = tasks_acc
    test_stats['std'] = tasks_std

    print('Total Accuracy and std on {} tasks: {:.4f} , {:.4f}'.format(
        str(cfg['tasks']), tasks_acc, tasks_std))
    print('test time: {}'.format(str(time.time() - f_test_time)))
    if not os.path.exists(cfg['output_dir']):
        os.mkdir(cfg['output_dir'])
    csv_path = os.path.join(cfg['output_dir'], cfg['dataset']+".csv")
    write_to_csv(cfg, csv_path, test_stats)

def write_to_csv(cfg, path, test_stats):
    
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame()
    records = res.to_dict('records')
    if cfg['method'] == "TIPAdapter" and cfg["finetune"]:
        test_stats['method'] = "TIPAdapter-F"
    else:
        test_stats['method'] = cfg['method']
    test_stats['acc'] = round(test_stats['acc'],4)
    test_stats['std'] = round(test_stats['std'],4)
    test_stats['num_shots'] = cfg['shots']
    test_stats['tasks'] = cfg['tasks']

    records.append(test_stats)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    main()

