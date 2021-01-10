from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

from .. import ImageDataset
from ..utils import mkdir_if_missing, read_json, write_json


class CUHK03_splited(ImageDataset):
    """CUHK03.

    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_

    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'CUHK03'
    #dataset_url = None

    def __init__(self, root='', split_id=0, cuhk03_labeled=False, cuhk03_classic_split=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        # self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        # self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(
            self.dataset_dir, 'splits_classic_detected.json')
        self.split_classic_lab_json_path = osp.join(
            self.dataset_dir, 'splits_classic_labeled.json')

        self.split_new_det_json_path = osp.join(
            self.dataset_dir, 'splits_new_detected.json')
        self.split_new_lab_json_path = osp.join(
            self.dataset_dir, 'splits_new_labeled.json')

        self.split_new_det_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.split_new_lab_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')

        required_files = [
            self.dataset_dir,
            # self.data_dir,
            # self.raw_mat_path,
            self.split_new_det_mat_path,
            self.split_new_lab_mat_path
        ]
        self.check_before_run(required_files)

        # self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(splits), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits))
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']
        new_train_list = []
        new_query_list = []
        new_gallery_list = []
        for item in train:
            new_train_list.append(
                [osp.join(self.dataset_dir, item[0][31:]), item[1], item[2]])
        for item in query:
            new_query_list.append(
                [osp.join(self.dataset_dir, item[0][31:]), item[1], item[2]])
        for item in gallery:
            new_gallery_list.append(
                [osp.join(self.dataset_dir, item[0][31:]), item[1], item[2]])
        super(CUHK03_splited, self).__init__(new_train_list,
                                     new_query_list, new_gallery_list, **kwargs)
