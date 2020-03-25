from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
import os
import re


class GTA(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        if dtype == 'train':
            data_path += '/train'
        elif dtype == 'test':
            data_path += '/gallery'
        else:
            data_path += '/query'

        self.imgs = [path for path in self.list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        print('{} classes.'.format(len(self.unique_ids)))

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[4])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[5][-1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))


    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(
            directory), 'dataset is not exists!{}'.format(directory)
        imgs=[]

        for d in os.listdir(directory):
            if os.path.isdir(os.path.join(directory,d)):
                for file in os.listdir(os.path.join(directory,d)):
                    if file.split('.')[-1] == 'jpeg':
                        imgs.append(os.path.join(directory,d,file))
        return imgs

        # return sorted([os.path.join(root, f)
        #                for root, _, files in os.walk(directory) for f in files
        #                if re.match(r'([\w]+\.(?:' + ext + '))', f)])
if __name__ == '__main__':
    dataset = GTA
