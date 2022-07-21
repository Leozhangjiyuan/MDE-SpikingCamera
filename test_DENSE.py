import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from model.metric import *
from model.S2DepthNet import S2DepthTransformerUNetConv
from data_loader.SpikesDENSE_dataset import *
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import bisect


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


logging.basicConfig(level=logging.INFO, format='')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_metrics(output, target):
    metrics = [mse, abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics


def make_colormap(img, color_mapper):
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = np.nan_to_num(color_map_inv, nan=1)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    color_map_inv = color_mapper.to_rgba(color_map_inv)
    color_map_inv[:, :, 0:3] = color_map_inv[:, :, 0:3][..., ::-1]
    return color_map_inv

def main(config, initial_checkpoint, output_folder, data_folder):
    train_logger = None

    # L = config['trainer']['sequence_length']
    # assert(L > 0)
    L = 1
    calculate_scale = True

    dataset_type, base_folder, spike_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size = {}
    clip_distance = {}
    scale_factor = {}
    every_x_rgb_frame = {}
    baseline = {}
    recurrency = {}
    total_metrics = []

    # this will raise an exception is the env variable is not set
    # preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    if output_folder:
        ensure_dir(output_folder)
        depth_dir = join(output_folder, "depth")
        npy_dir = join(output_folder, "npy")
        color_map_dir = join(output_folder, "color_map")
        gt_dir_grey = join(output_folder, "ground_truth/grey")
        gt_dir_color_map = join(output_folder, "ground_truth/color_map")
        gt_dir_npy = join(output_folder, "ground_truth/npy")
        semantic_seg_dir_npy = join(output_folder, "semantic_seg/npy")
        semantic_seg_dir_frames = join(output_folder, "semantic_seg/frames")
        video_pred = join(output_folder, "video/predictions")
        video_gt = join(output_folder, "video/gt")
        video_inputs = join(output_folder, "video/inputs")
        ensure_dir(depth_dir)
        ensure_dir(npy_dir)
        ensure_dir(color_map_dir)
        ensure_dir(gt_dir_grey)
        ensure_dir(gt_dir_color_map)
        ensure_dir(gt_dir_npy)
        ensure_dir(semantic_seg_dir_npy)
        ensure_dir(semantic_seg_dir_frames)
        ensure_dir(video_pred)
        ensure_dir(video_gt)
        ensure_dir(video_inputs)
        print('Will write images to: {}'.format(depth_dir))

    use_phased_arch = config['use_phased_arch']

    dataset_type['validation'] = config['data_loader']['validation']['type']
    # change path here if test.py needs to be run on validation or training data
    if data_folder is not None:
        base_folder['validation'] = data_folder
    else:
        # base_folder['validation'] =  config['data_loader']['validation']['base_folder']
        print("base_folder not loaded properly!")

    spike_folder['validation'] = config['data_loader']['validation']['spike_folder']
    depth_folder['validation'] = config['data_loader']['validation']['depth_folder']
    frame_folder['validation'] = config['data_loader']['validation']['frame_folder']
    proba_pause_when_running['validation'] = config['data_loader']['validation']['proba_pause_when_running']
    proba_pause_when_paused['validation'] = config['data_loader']['validation']['proba_pause_when_paused']
    scale_factor['validation'] = config['data_loader']['validation']['scale_factor']
    recurrency['validation'] = True

    try:
        step_size['validation'] = 1
    except KeyError:
        step_size['validation'] = 1

    try:
        clip_distance['validation'] = config['data_loader']['validation']['clip_distance']
    except KeyError:
        clip_distance['validation'] = 100.0
        print("Clip distance not loaded properly!")

    try:
        every_x_rgb_frame['validation'] = config['data_loader']['validation']['every_x_rgb_frame']
    except KeyError:
        every_x_rgb_frame['validation'] = 1
        print("Every_x_rgb_frame not loaded properly!")

    try:
        baseline['validation'] = config['data_loader']['validation']['baseline']
    except KeyError:
        baseline['validation'] = False
        print("Baseline not loaded properly!")

    normalize = config['data_loader'].get('normalize', True)
    loss_composition = config['trainer']['loss_composition']

    test_dataset = concatenate_subfolders(base_folder['validation'],
                                          dataset_type['validation'],
                                          spike_folder['validation'],
                                          depth_folder['validation'],
                                          frame_folder['validation'],
                                          sequence_length=L,
                                          # change transform to evaluate on different input sizes.
                                          transform=CenterCrop(224),
                                          proba_pause_when_running=proba_pause_when_running['validation'],
                                          proba_pause_when_paused=proba_pause_when_paused['validation'],
                                          step_size=step_size['validation'],
                                          clip_distance=clip_distance['validation'],
                                          every_x_rgb_frame=every_x_rgb_frame['validation'],
                                          normalize=normalize,
                                          scale_factor=scale_factor['validation'],
                                          use_phased_arch=use_phased_arch,
                                          baseline=baseline['validation'],
                                          loss_composition=loss_composition,
                                          dataset_idx_flag=True,
                                          recurrency=recurrency['validation']
                                          )

    config['model']['gpu'] = config['gpu']
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']

    reg_factor = None

    try:
        reg_factor = config['data_loader']['train']['reg_factor']
    except KeyError:
        reg_factor = 5.7
    
    model = eval(config['arch'])(config['model'])
    model.summary()


    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model = torch.nn.DataParallel(model).cuda()
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)
        model.load_state_dict(checkpoint['state_dict'])

    # model.summary()

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)

    model.eval()

    video_idx = 0

    N = len(test_dataset)
    print(N)
    if calculate_scale:
        scale = np.empty(N)

    # construct color mapper, such that same color map is used for all outputs.
    # get groundtruth that is not at the beginning
    item, dataset_idx = test_dataset[20]
    frame = item[0]['depth_image'].cpu().numpy()

    print("======================================")
    print(frame.shape)
    print("======================================")

    color_map_inv = np.ones_like(frame[0]) * np.amax(frame[0]) - frame[0]
    color_map_inv = np.nan_to_num(color_map_inv, nan=1)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    vmax = np.percentile(color_map_inv, 95)
    normalizer = mpl.colors.Normalize(vmin=color_map_inv.min(), vmax=vmax)
    color_mapper_overall = cm.ScalarMappable(norm=normalizer, cmap='magma')

    with torch.no_grad():
        idx = 0
        prev_dataset_idx = -1
        # for batch_idx, sequence in enumerate(data_loader):

        while idx < N:
            item, dataset_idx = test_dataset[idx]

            if dataset_idx > prev_dataset_idx:
                # reset internal states for new sequence
                prev_super_states = {'image': None}
                prev_states_lstm = {}
                for k in range(0, every_x_rgb_frame['validation']):
                    prev_states_lstm['events{}'.format(k)] = None
                    prev_states_lstm['depth{}'.format(k)] = None
                prev_states_lstm['image'] = None
                sequence_idx = 0

            # new_events, new_image, new_target, times = to_input_and_target(item[0], gpu, use_phased_arch)
            # the output of the network is a [N x 1 x H x W] tensor containing the image prediction
            input = {}
            for key, value in item[0].items():
                input[key] = value[None, :]
            new_predicted_targets, new_super_states, new_states_lstm = model(input,
                                                                             prev_super_states['image'],
                                                                             prev_states_lstm)

            if idx > 20 and idx < 24:
                print("test preview of index ", idx)
                fig, ax = plt.subplots(ncols=every_x_rgb_frame['validation']+1, nrows=4)
                for i, key in enumerate(new_predicted_targets.keys()):
                    ax[0, i].imshow(new_predicted_targets[key][0].cpu().numpy()[0])
                    ax[0, i].set_title("prediction " + key)
                index_1 = 0
                index_2 = 0
                for i, key in enumerate(input.keys()):
                    if "depth" in key:
                        ax[1, index_1].imshow(input[key][0].cpu().numpy()[0])
                        ax[1, index_1].set_title("groundtruth " + key)
                        index_1 += 1
                    else:
                        #ax[2, index_2].imshow(torch.sum(input[key], dim=1)[0].cpu().numpy())  # all
                        ax[2, index_2].imshow(torch.sum(input[key][0][0:-2], dim=0).cpu().numpy())  # events only
                        ax[3, index_2].imshow(input[key][0][-1].cpu().numpy())  # image only
                        ax[2, index_2].set_title("input eventdata" + key)
                        ax[3, index_2].set_title("input imagedata" + key)
                        index_2 += 1
                plt.show()

            # crop prediction output
            # transform = CenterCrop(224)

            if output_folder and sequence_idx > 1:
                #print("save images")
                # don't save the first 2 predictions such that the temporal dependencies of the network are settled.
                for key, img in new_predicted_targets.items():
                    groundtruth = input['depth_' + key]

                    #metrics = eval_metrics(new_predicted_targets[key], groundtruth)
                    metrics = eval_metrics(img, groundtruth)
                    total_metrics.append(metrics)
                    # print("metrics of index ", idx, ": ", metrics)
                    img = img[0].cpu().numpy()

                    # save depth image
                    depth_dir_key = join(depth_dir, key)
                    ensure_dir(depth_dir_key)
                    cv2.imwrite(join(depth_dir_key, 'frame_{:010d}.png'.format(idx)), img[0][:, :, None] * 255.0)

                    # save numpy
                    npy_dir_key = join(npy_dir, key)
                    ensure_dir(npy_dir_key)
                    data = img
                    np.save(join(npy_dir_key, 'depth_{:010d}.npy'.format(idx)), data)

                    # save color map
                    color_map_dir_key = join(color_map_dir, key)
                    ensure_dir(color_map_dir_key)
                    color_map = make_colormap(img, color_mapper_overall)
                    cv2.imwrite(join(color_map_dir_key, 'frame_{:010d}.png'.format(idx)), color_map * 255.0)

                for key, value in input.items():
                    if 'depth' in key:
                        # save GT images grey
                        gt_dir_grey_key = join(gt_dir_grey, key)
                        ensure_dir(gt_dir_grey_key)
                        img = value[0].cpu().numpy()
                        cv2.imwrite(join(gt_dir_grey_key, 'frame_{:010d}.png'.format(idx)), img[0][:, :, None] * 255.0)

                        # save GT images color map
                        gt_dir_cm_key = join(gt_dir_color_map, key)
                        ensure_dir(gt_dir_cm_key)
                        color_map = make_colormap(img, color_mapper_overall)
                        cv2.imwrite(join(gt_dir_cm_key, 'frame_{:010d}.png'.format(idx)), color_map * 255.0)

                        # save GT to numpy array
                        gt_dir_npy_key = join(gt_dir_npy, key)
                        ensure_dir(gt_dir_npy_key)
                        np.save(join(gt_dir_npy_key, 'frame_{:010d}.npy'.format(idx)), img)
                    elif 'semantic' in key:
                        # save semantic seg numpy array
                        img = value[0].cpu().numpy()[0]
                        semantic_seg_dir_npy_key = join(semantic_seg_dir_npy, key)
                        ensure_dir(semantic_seg_dir_npy_key)
                        np.save(join(semantic_seg_dir_npy_key, 'frame_{:010d}.npy'.format(idx)), img)
                        # save semantic seg frame
                        semantic_seg_dir_frames_key = join(semantic_seg_dir_frames, key)
                        ensure_dir(semantic_seg_dir_frames_key)
                        cv2.imwrite(join(semantic_seg_dir_frames_key, 'frame_{:010d}.png'.format(idx)), img)

                # save data for video of consecutive inputs
                if baseline['validation'] == "rgb" or config['arch'] == "ERGB2Depth":
                    keys = ["image"]
                else:
                    keys = []
                    for i in range(every_x_rgb_frame['validation']-1):
                        keys.append("events{}".format(i))
                    if not baseline['validation']:
                        keys.append("events{}".format(every_x_rgb_frame['validation']-1))
                    keys.append("image")

                for key in keys:
                    prediction = new_predicted_targets[key].cpu().numpy()
                    gt_data = item[0]["depth_" + key].cpu().numpy()
                    input_data = item[0][key].cpu().numpy()

                    # save data
                    cm_prediction = make_colormap(prediction[0], color_mapper_overall)
                    cv2.imwrite(join(video_pred, 'frame_{:010d}.png'.format(video_idx)), cm_prediction * 255.0)
                    cm_gt_data = make_colormap(gt_data, color_mapper_overall)
                    cv2.imwrite(join(video_gt, 'frame_{:010d}.png'.format(video_idx)), cm_gt_data * 255.0)
                    input_data = np.sum(input_data, axis=0)
                    if "event" in key:
                        #input_data = np.ones_like(input_data) * 0.5
                        #negativ_input = np.where(input_data <= 0, np.ones_like(input_data), np.zeros_like(input_data))
                        negativ_input = np.where(input_data <= -0.5, 1.0, 0.0)
                        positiv_input = np.where(input_data > 0.9, 1.0, 0.0)
                        zeros_input = np.zeros_like(input_data)
                        total_image = np.concatenate((negativ_input[:, :, None], zeros_input[:, :, None], positiv_input[:, :, None]), axis=2)
                        '''fig, ax = plt.subplots(ncols=1, nrows=4)
                        ax[0].imshow(negativ_input)
                        ax[0].set_title("negativ input")
                        ax[1].imshow(positiv_input)
                        ax[1].set_title("positive input")
                        ax[2].imshow(zeros_input)
                        ax[2].set_title("zeros input")
                        ax[3].imshow(total_image)
                        ax[3].set_title("total input")
                        plt.show()'''
                        cv2.imwrite(join(video_inputs, 'frame_{:010d}.png'.format(video_idx)), total_image * 255.0)
                    else:
                        cv2.imwrite(join(video_inputs, 'frame_{:010d}.png'.format(video_idx)), input_data[:, :, None] * 255.0)

                    video_idx += 1

                if idx % 100 == 0:
                    print("saved image ", idx)

            if calculate_scale:
                for key, img in new_predicted_targets.items():
                    key_target = f"depth_{key}"

                    prediction = img[0][0].cpu().numpy()
                    target = input[key_target][0][0].cpu().numpy()
                    # change to metric space
                    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]),
                                                                           dtype=np.float32)))
                    target = np.exp(reg_factor * (target - np.ones((target.shape[0], target.shape[1]),
                                                                   dtype=np.float32)))
                    target *= clip_distance['validation']
                    prediction *= clip_distance['validation']
                    scale[idx] = np.sum(prediction * target) / np.sum(prediction * prediction)

            prev_super_states = new_super_states
            prev_states_lstm = new_states_lstm
            sequence_idx += 1
            prev_dataset_idx = dataset_idx
            idx += 1

        if calculate_scale:
            total_scale = np.mean(scale)
            # print(scale)
            print("total scale: ", total_scale)
            print("min scale: ", np.min(scale))
            print("max scale: ", np.max(scale))

        # total metrics:
        print("total metrics: ", np.sum(np.array(total_metrics), 0) / len(total_metrics))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Spike Transformer')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default=None)
    parser.add_argument('--output_path', type=str,
                        help='path to folder for saving outputs',
                        default='')
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default=None)


    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))

    main(config, args.path_to_model, args.output_path, args.data_folder)
