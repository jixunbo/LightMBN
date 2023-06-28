# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET

from .. import ImageDataset


class AICity20(ImageDataset):

    dataset_dir = "AICITY21/AIC21_Track2_ReID"
    dataset_aug_dir = "AICITY21/AIC20_ReID_Simulation"

    def __init__(self, root="", verbose=True, **kwargs):
        # super(AICity20, self).__init__()
        # import ipdb; ipdb.set_trace()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.dataset_aug_dir = osp.join(root, self.dataset_aug_dir)

        self.train_dir = osp.join(self.dataset_dir, "image_train")
        self.query_dir = osp.join(self.dataset_dir, "image_query")
        self.gallery_dir = osp.join(self.dataset_dir, "image_test")
        # self.train_aug_dir = osp.join(self.dataset_aug_dir, 'image_train')
        self.train_track_path = osp.join(self.dataset_dir, "train_track.txt")
        self.test_track_path = osp.join(self.dataset_dir, "test_track.txt")

        self.list_train_path = osp.join(self.dataset_dir, "name_train.txt")
        self.list_query_path = osp.join(self.dataset_dir, "name_query.txt")
        self.list_gallery_path = osp.join(self.dataset_dir, "name_test.txt")

        self.train_label_path = osp.join(self.dataset_dir, "train_label.xml")

        self._check_before_run()

        train = self._process_dir(
            self.train_dir, self.list_train_path, self.train_label_path, relabel=False
        )
        query = self._process_dir(self.query_dir, self.list_query_path, None)
        # query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path, None)
        # gallery = self._process_dir_test(self.gallery_dir, relabel=False, query=False)
        # import ipdb; ipdb.set_trace()
        # train += self._process_dir(self.train_aug_dir, self.list_train_path, self.train_label_path, relabel=False)
        train = self.relabel(train)
        if verbose:
            print("=> AI CITY 2021 data loaded")
            # self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.train_tracks = self._read_tracks(
            os.path.join(self.dataset_dir, "train_track.txt")
        )
        self.test_tracks = self._read_tracks(
            os.path.join(self.dataset_dir, "test_track.txt")
        )

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
        ) = self.get_imagedata_info(self.gallery)
        super(AICity20, self).__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(
        self, img_dir, list_path, label_path, relabel=False, domain="real"
    ):
        dataset = []
        if label_path:
            tree = ET.parse(label_path, parser=ET.XMLParser(encoding="utf-8"))
            objs = tree.find("Items")
            for obj in objs:
                image_name = obj.attrib["imageName"]
                img_path = osp.join(img_dir, image_name)
                pid = int(obj.attrib["vehicleID"])
                camid = int(obj.attrib["cameraID"][1:])
                dataset.append((img_path, pid, camid, domain))
                # dataset.append((img_path, camid, pid))
            if relabel:
                dataset = self.relabel(dataset)
        else:
            with open(list_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    img_path = osp.join(img_dir, line)
                    pid = 0
                    camid = 0
                    dataset.append((img_path, pid, camid, domain))
        return dataset

    def _process_dir_test(self, dir_path, relabel=False, query=True):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))

        dataset = []
        _, _, frame2trackID = self._process_track(path=self.test_track_path)
        for img_path in img_paths:
            camid = 1
            pid = 2
            if False:
                dataset.append((img_path, pid, camid, -1))
            else:
                trackid = frame2trackID[(img_path[-10:])]
                camid = trackid
                dataset.append((img_path, pid, camid, trackid))
        # print(len(dataset), 'len(dataset)')
        return dataset

    def _process_track(self, path):  #### Revised

        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            # curLine = list(map(eval, curLine[-10:-4]))
            tracklet[track_id] = curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
        return tracklet, nums, frame2trackID


if __name__ == "__main__":
    dataset = AICity20(root="/media/data/ai-city/Track2/AIC21_Track2_ReID")
