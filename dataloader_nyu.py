from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio


# nyu/test/1318_a=0.55_b=1.21.png
class nyu_DataSet(Dataset):
    def __init__(self, transform1, path=None):
        # print(path)
        self.transform1 = transform1
        self.haze_path, self.gt_path, self.d_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.gt_data_list = os.listdir(self.gt_path)
        self.d_data_list = os.listdir(self.d_path)
        self.haze_data_list.sort(key=lambda x: float(x[-8:-4]))
        self.haze_data_list.sort(key=lambda x: int(x[:-18]))

        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        self.d_data_list.sort(key=lambda x: int(x[:-4]))
        self.length = len(os.listdir(self.haze_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            需要传递的信息有：
            有雾图像
            无雾图像
            (深度图)
            (雾度)
            (大气光)
            624, 464
        """
        A_image = np.ones((608, 448, 3), dtype=np.float32)
        A_gt = np.ones((608, 448, 3), dtype=np.float32)
        t_gt = np.ones((608, 448, 1), dtype=np.float32)
        haze_image_name = self.haze_data_list[idx]
        haze_image = cv2.imread(self.haze_path + haze_image_name)
        gt_image = cv2.imread(self.gt_path + haze_image_name[:-18] + '.PNG')
        d_image = np.load(self.d_path + haze_image_name[:-18] + '.npy')
        b = float(haze_image_name[-8:-4])
        A = float(haze_image_name[-15:-11])
        d_image = np.expand_dims(d_image, axis=2)
        d_image = d_image.astype(np.float32)
        t_image = np.exp(-1 * b * d_image)
        A_image = A_image * A
        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
            A_image = self.transform1(A_image)
            A_gt = self.transform1(A_gt)
            t_image = self.transform1(t_image)
            t_gt = self.transform1(t_gt)
        '''
        print(1)
        print(haze_image[0][0][0])
        print(2)
        print(gt_image[0][0][0])
        print(3)
        print(A_image[0][0][0])
        print(4)
        print(A_gt[0][0][0])
        print(5)
        print(t_image[0][0][0])
        print(6)
        print(t_gt[0][0][0])
        '''
        return haze_image.cuda(), gt_image.cuda(), A_image.cuda(), A_gt.cuda(), t_image.cuda(), t_gt.cuda()

# if __name__ == '__main__':
