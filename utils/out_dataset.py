import numpy as np
import torch
from bisect import bisect_left
import PIL
import os

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=True):

        data_file = open('/nobackup-slow/dataset/80million/tiny_images.bin', "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            # print(np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F"))
            # breakpoint()
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('/nobackup-slow/dataset/80million/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 79302017


class RandomImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=False):

        data_file = np.load('/nobackup-slow/dataset/my_xfdu/300K_random_images.npy')

        def load_image(idx):
            # data_file.seek(idx * 3072)
            # data = data_file.read(3072)
            data = data_file[idx]
            return np.asarray(data, dtype='uint8')#.reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('/nobackup-slow/dataset/80million/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search

    def __getitem__(self, index):
        index = (index + self.offset) % 299999

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(300000)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 300000

class RandomImages42k(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=False):

        data_file = np.load('/nobackup-slow/dataset/my_xfdu/300K_random_images.npy')
        indices = np.random.permutation(len(data_file))
        data_file = data_file[indices[:42000]]
        def load_image(idx):
            # data_file.seek(idx * 3072)
            # data = data_file.read(3072)
            data = data_file[idx]
            return np.asarray(data, dtype='uint8')#.reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('/nobackup-slow/dataset/80million/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search

    def __getitem__(self, index):
        index = (index + self.offset) % 41999

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(42000)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 42000

class RandomImages50k(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=False):

        data_file = np.load('/nobackup-slow/dataset/my_xfdu/300K_random_images.npy')
        indices = np.random.permutation(len(data_file))
        self.data_file = data_file[indices[:50100]]
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

    def __getitem__(self, index):
        index = (index + self.offset) % 50099

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(50100)


        data = self.data_file[index]
        img = np.asarray(data, dtype='uint8')


        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 50100

class SDImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=False):

        data_file = []
        for item in list(os.listdir('/nobackup-slow/dataset/my_xfdu/sd/txt2img-samples/samples')):
            data_file.append(os.path.join('/nobackup-slow/dataset/my_xfdu/sd/txt2img-samples/samples', item))



        def load_image(idx):
            data = data_file[idx]
            data = np.asarray(PIL.Image.open(data))
            return data

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('/nobackup-slow/dataset/80million/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs



    def __getitem__(self, index):
        index = (index + self.offset) % 41999

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(42000)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 42000