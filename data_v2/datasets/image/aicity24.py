import os.path as osp
import re

from .. import ImageDataset


class AICity24Balanced(ImageDataset):
    """AICity24 dataset.

    Cropped from the tracking dataset of the AI City Challenge 2024 with balanced
    distribution of identities. For each identity, we choose 100 images for the
    training set. Then, for query and gallery, we choose 108 images for each
    identitiy in the validation set, where 8 images are used for query and 100
    images are used for gallery.

    We choose a "hard" setting here, where there is no shared camera between
    query and gallery, i.e., for one identity in the query set, the same identity
    in the gallery will be from different cameras.

    | subset   | # ids   | # images   | # cameras   |
    |:---------|:--------|:-----------|:------------|
    | train    | 1012    | 101200     | 350         |
    | query    | 518     | 4144       | 20          |
    | gallery  | 518     | 51800      | 154         |

    """

    _junk_pids = [0, -1]
    dataset_dir = "AICITY24"
    dataset_name = "AICity24"

    def __init__(self, root="datasets", **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.seed = 0
        self.root = root
        self.data_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, "bounding_box_train")
        self.query_dir = osp.join(self.data_dir, "bounding_box_query")
        self.gallery_dir = osp.join(self.data_dir, "bounding_box_test")

        self.valid_train_stems = []
        with open(osp.join(self.data_dir, "valid_train_stems.txt"), "r") as f:
            for line in f:
                self.valid_train_stems.append(line.strip())

        self.valid_test_stems = []
        with open(osp.join(self.data_dir, "valid_test_stems.txt"), "r") as f:
            for line in f:
                self.valid_test_stems.append(line.strip())

        self.valid_query_stems = []
        with open(osp.join(self.data_dir, "valid_query_stems.txt"), "r") as f:
            for line in f:
                self.valid_query_stems.append(line.strip())

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        train = self.process_dir(self.train_dir, mode="train")
        query = self.process_dir(self.query_dir, mode="query")
        gallery = self.process_dir(self.gallery_dir, mode="test")

        super(AICity24Balanced, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, mode="train"):
        if mode == "train":
            img_paths = [osp.join(dir_path, f + ".jpg") for f in self.valid_train_stems]
        elif mode == "test":
            img_paths = [osp.join(dir_path, f + ".jpg") for f in self.valid_test_stems]
        elif mode == "query":
            img_paths = [osp.join(dir_path, f + ".jpg") for f in self.valid_query_stems]
        else:
            raise ValueError("Invalid mode")

        pattern = re.compile(r"([-\d]+)_s([-\d]+)c([-\d]+)")

        data = []

        for img_path in img_paths:
            pid, sceneid, camid = map(int, pattern.search(img_path).groups())
            data.append((img_path, pid, camid))

        return data
