# -*- coding: utf-8 -*-
"""
SpikeDataset classes
"""

from torch.utils.data import Dataset
from skimage import io
from os.path import join
import numpy as np
from utils.util import first_element_greater_than, last_element_less_than
import random
import torch
import glob


class SpikeDataset(Dataset):
    """Loads spike tensors from a folder, with different spike representations."""
    def __init__(self, base_folder, spike_folder, start_time=0, stop_time=0, transform=None, normalize=True):
        self.base_folder = base_folder
        self.spike_folder = join(self.base_folder, spike_folder)
        self.transform = transform

        self.start_time = start_time
        self.stop_time = stop_time

        self.normalize = normalize

        if "mvsec" in base_folder or "javi" in base_folder:
            self.use_mvsec = True
        else:
            self.use_mvsec = False

        self.read_timestamps()

        self.parse_event_folder()

    def read_timestamps(self):
        # Load the timestamps file
        raw_stamps = np.loadtxt(join(self.spike_folder, 'timestamps.txt'))

        if raw_stamps.size == 0:
            raise IOError('Dataset is empty')

        if len(raw_stamps.shape) == 1:
            # if timestamps.txt has only one entry, the shape will be (2,) instead of (1, 2). fix that.
            raw_stamps = raw_stamps.reshape((1, 2))

        self.stamps = raw_stamps[:, 1]
        if self.stamps is None:
            raise IOError('Unable to read timestamp file: '.format(join(self.spike_folder,
                                                                        'timestamps.txt')))

        # Check that the timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)), "timestamps are not unique and monotonically increasing"

        self.initial_stamp = self.stamps[0]
        self.stamps = self.stamps - self.initial_stamp  # offset the timestamps so they start at 0

        # Find the index of the first spike tensor whose timestamp >= start_time
        # If there is none, throw an error
        if self.start_time <= 0.0:
            self.first_valid_idx, self.first_stamp = 1, self.stamps[1]
        else:
            self.first_valid_idx, self.first_stamp = first_element_greater_than(self.stamps, self.start_time)
        assert(self.first_stamp is not None)
        # print('First valid index / stamp = {}, {}'.format(self.first_valid_idx, self.first_stamp))

        # Find the index of the last spike tensor whose timestamp <= end_time
        # If there is None, throw an error
        if self.stop_time <= 0.0:
            self.last_valid_idx, self.last_stamp = len(self.stamps) - 1, self.stamps[-1]
        else:
            self.last_valid_idx, self.last_stamp = last_element_less_than(self.stamps, self.stop_time)
        assert(self.last_stamp is not None)
        # print('Last valid index / stamp = {}, {}'.format(self.last_valid_idx, self.last_stamp))

        assert(self.first_stamp <= self.last_stamp)

        if self.use_mvsec and not "javi" in self.base_folder:
            self.length = self.last_valid_idx - self.first_valid_idx + 1 - 1
        else:
            self.length = self.last_valid_idx - self.first_valid_idx + 1
        assert(self.length > 0)

    def parse_event_folder(self):
        """Parses the event folder to check its validity and read the parameters of the event representation."""
        raise NotImplementedError

    def __len__(self):
        return self.length

    def get_last_stamp(self):
        """Returns the last spike timestamp, in seconds."""
        return self.stamps[self.last_valid_idx]

    def num_channels(self):
        """Returns the number of channels of the spike tensor."""
        raise NotImplementedError

    def get_index_at(self, i):
        """Returns the index of the ith spike tensor"""
        return self.first_valid_idx + i

    def get_stamp_at(self, i):
        """Returns the timestamp of the ith spike tensor"""
        return self.stamps[self.get_index_at(i)]

    def __getitem(self, i):
        """Returns a C x H x W spike tensor for the ith element in the dataset."""
        raise NotImplementedError


class VoxelGridDENSESpikeDataset(SpikeDataset):
    """Load an spike folder containing spike tensors encoded with the VoxelGrid format."""

    def parse_event_folder(self):
        """Check that the passed directory has the following form:

        ├── spike_folder
        |   ├── timestamps.txt
        |   ├── spike_tensor_0000000000.npy
        |   ├── ...
        |   ├── spike_tensor_<N>.npy
        """
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        # if self.use_mvsec:
        #     event_tensor = np.load(join(self.spike_folder, 'event_tensor_{:010d}.npy'.format(self.first_valid_idx + i)))
        # else:
        path_spike = glob.glob(self.spike_folder + '/spike_{:010d}.npy'.format(self.first_valid_idx + i))
        
        spike_tensor = np.load(path_spike[0]).astype(np.float32)

        if self.normalize:
            # normalize the spike tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
            # in the tensor are equal to (0.0, 1.0)
            mask = np.nonzero(spike_tensor)
            if mask[0].size > 0:
                mean, stddev = spike_tensor[mask].mean(), spike_tensor[mask].std()
                if stddev > 0:
                    spike_tensor[mask] = (spike_tensor[mask] - mean) / stddev

        self.num_bins = spike_tensor.shape[0]

        spikes = torch.from_numpy(spike_tensor)  # [C x H x W]
        if self.transform:
            random.seed(transform_seed)
            spikes = self.transform(spikes)

        return {'events': spikes}  # [num_bins x H x W] tensor