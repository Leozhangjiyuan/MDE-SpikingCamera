import os
import json
import logging
import argparse
import torch
from model.model import *
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.loss import *
from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.SpikesDENSE_dataset import *
from trainer.spiket_trainer import SpikeTTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import bisect

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format='')

parser = argparse.ArgumentParser(
        description='Spike Transformer')
parser.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
parser.add_argument('-f', '--datafolder', default=None, type=str,
                    help='datafolder root path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-i', '--initial_checkpoint', default=None, type=str,
                    help='path to the checkpoint with which to initialize the model weights (default: None)')
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1241')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)

args = parser.parse_args()

class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


def concatenate_subfolders(base_folder, dataset_type, spike_folder, depth_folder, frame_folder, sequence_length,
                           transform=None, proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=1,
                           clip_distance=100.0, every_x_rgb_frame=1, normalize=True, scale_factor=1.0,
                           use_phased_arch=False, baseline=False, loss_composition=False, reg_factor=5.7,
                           dataset_idx_flag=False, recurrency=True):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """

    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))

    train_datasets = []
    for dataset_name in subfolders:
        train_datasets.append(eval(dataset_type)(base_folder=join(base_folder, dataset_name),
                                                 spike_folder=spike_folder,
                                                 depth_folder=depth_folder,
                                                 frame_folder=frame_folder,
                                                 sequence_length=sequence_length,
                                                 transform=transform,
                                                 proba_pause_when_running=proba_pause_when_running,
                                                 proba_pause_when_paused=proba_pause_when_paused,
                                                 step_size=step_size,
                                                 clip_distance=clip_distance,
                                                 every_x_rgb_frame=every_x_rgb_frame,
                                                 normalize=normalize,
                                                 scale_factor=scale_factor,
                                                 use_phased_arch=use_phased_arch,
                                                 baseline=baseline,
                                                 loss_composition=loss_composition,
                                                 reg_factor=reg_factor,
                                                 recurrency=recurrency))

    if dataset_idx_flag == False:
        concat_dataset = ConcatDataset(train_datasets)
    elif dataset_idx_flag == True:
        concat_dataset = ConcatDatasetCustom(train_datasets)

    return concat_dataset


def main_worker(gpu, ngpus_per_node, args):
# def main_worker(config, resume, initial_checkpoint=None, DeviceIds=None):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    config = args.config 
    resume = args.resume
    initial_checkpoint = args.initial_checkpoint


    train_logger = None

    L = config['trainer']['sequence_length']
    assert (L > 0)

    dataset_type, base_folder, spike_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size = {}
    clip_distance = {}
    scale_factor = {}
    every_x_rgb_frame = {}
    reg_factor = {}
    baseline = {}
    recurrency = {}

    # this will raise an exception is the env variable is not set
    # preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    use_phased_arch = config['use_phased_arch']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = config['data_loader'][split]['base_folder']
        spike_folder[split] = config['data_loader'][split]['spike_folder']
        depth_folder[split] = config['data_loader'][split]['depth_folder']
        frame_folder[split] = config['data_loader'][split]['frame_folder']
        proba_pause_when_running[split] = config['data_loader'][split]['proba_pause_when_running']
        proba_pause_when_paused[split] = config['data_loader'][split]['proba_pause_when_paused']
        scale_factor[split] = config['data_loader'][split]['scale_factor']
        recurrency[split] = True

        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

        try:
            every_x_rgb_frame[split] = config['data_loader'][split]['every_x_rgb_frame']
        except KeyError:
            every_x_rgb_frame[split] = 1

        try:
            baseline[split] = config['data_loader'][split]['baseline']
        except KeyError:
            baseline[split] = False

        try:
            reg_factor[split] = config['data_loader'][split]['reg_factor']
        except KeyError:
            reg_factor[split] = 5.7

    loss_composition = config['trainer']['loss_composition']
    loss_weights = config['trainer']['loss_weights']
    normalize = config['data_loader'].get('normalize', True)

    train_dataset = concatenate_subfolders(join(args.datafolder,base_folder['train']),
                                           dataset_type['train'],
                                           spike_folder['train'],
                                           depth_folder['train'],
                                           frame_folder['train'],
                                           sequence_length=L,
                                           transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                                              RandomCrop(224)]),
                                           proba_pause_when_running=proba_pause_when_running['train'],
                                           proba_pause_when_paused=proba_pause_when_paused['train'],
                                           step_size=step_size['train'],
                                           clip_distance=clip_distance['train'],
                                           every_x_rgb_frame=every_x_rgb_frame['train'],
                                           normalize=normalize,
                                           scale_factor=scale_factor['train'],
                                           use_phased_arch=use_phased_arch,
                                           baseline=baseline['train'],
                                           loss_composition=loss_composition,
                                           reg_factor=reg_factor['train'],
                                           recurrency=recurrency['train']
                                           )

    validation_dataset = concatenate_subfolders(join(args.datafolder,base_folder['validation']),
                                                dataset_type['validation'],
                                                spike_folder['validation'],
                                                depth_folder['validation'],
                                                frame_folder['validation'],
                                                sequence_length=L,
                                                transform=CenterCrop(224),
                                                proba_pause_when_running=proba_pause_when_running['validation'],
                                                proba_pause_when_paused=proba_pause_when_paused['validation'],
                                                step_size=step_size['validation'],
                                                clip_distance=clip_distance['validation'],
                                                every_x_rgb_frame=every_x_rgb_frame['validation'],
                                                normalize=normalize,
                                                scale_factor=scale_factor['train'],
                                                use_phased_arch=use_phased_arch,
                                                baseline=baseline['validation'],
                                                loss_composition=loss_composition,
                                                reg_factor=reg_factor['validation'],
                                                recurrency=recurrency['validation']
                                                )

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    data_loader = DataLoader(train_dataset, batch_size=int(config['data_loader']['batch_size']/ ngpus_per_node),
                            #  shuffle=config['data_loader']['shuffle'],
                             sampler=train_sampler,
                             **kwargs)

    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    valid_data_loader = DataLoader(validation_dataset, batch_size=int(config['data_loader']['batch_size']/ ngpus_per_node),
                                #    shuffle=config['data_loader']['shuffle'],
                                   sampler=validation_sampler,
                                   **kwargs)

    config['model']['gpu'] = args.gpu
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']

    torch.manual_seed(111)
    model = eval(config['arch'])(config['model'])
    model.summary()
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(config['data_loader']['batch_size'] / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        else:
            model.cuda()
            model = DataParallelModel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")
    

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        # model.load_state_dict(checkpoint['state_dict'])
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)  # tag="events"
        model.load_state_dict(checkpoint['state_dict'])
    
    cudnn.benchmark = True
    

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = SpikeTTrainer(model, args, loss, loss_params, metrics,
                              resume=resume,
                              config=config,
                              train_sampler=train_sampler,
                              data_loader=data_loader, 
                              ngpus_per_node=ngpus_per_node,
                              valid_data_loader=valid_data_loader,
                              train_logger=train_logger)

    trainer.train()


def main():
    logger = logging.getLogger()

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None
    args.config = config

    if args.multiprocessing_distributed:
        print("---- Distributed Training ----")
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':

    main()
