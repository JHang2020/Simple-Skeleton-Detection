# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import json
import os
from . import tools
try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation='graph-based',
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.crop_resize =True
        self.l_ratio = l_ratio
        self.aug_method = '1234'
        self.input_representation = "graph-based"


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print(self.data.shape,len(self.number_of_frames))
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = min(self.number_of_frames[index], 300)  # 300 is max_len, for pku-mmd

        # apply spatio-temporal augmentations to generate  view 1 

        # temporal crop-resize
        data_numpy_v1 = self.basic_aug(data_numpy,number_of_frames)
        data_numpy_v2 = self.basic_aug(data_numpy,number_of_frames)
        data_numpy_v3 = self.strong_aug(data_numpy,number_of_frames)
        #data_numpy_v4 = self.strong_aug(data_numpy,number_of_frames)
        #data_numpy_v5 = self.strong_aug(data_numpy,number_of_frames)
        #data_numpy_v6 = self.strong_aug(data_numpy,number_of_frames)

        # convert augmented views into input formats based on skeleton-representations
        if self.input_representation == "seq-based" or self.input_representation == "trans-based": 

             #Input for sequence-based representation
             # two person  input ---> shpae (64 X 150)

             #View 1
             input_v1 = data_numpy_v1.transpose(1,2,0,3)
             input_v1 = input_v1.reshape(-1,150).astype('float32')

             #View 2
             input_v2 = data_numpy_v2.transpose(1,2,0,3)
             input_v2 = input_v2.reshape(-1,150).astype('float32')

             return input_v1, input_v2

        elif self.input_representation == "graph-based" or self.input_representation == "image-based": 

             #input for graph-based or image-based representation
             # two person input --->  shape (3, 64, 25, 2)

             #View 1
             input_v1 = data_numpy_v1.astype('float32')
             #View 2
             input_v2 = data_numpy_v2.astype('float32')
             input_v3 = data_numpy_v3.astype('float32')
             #input_v4 = data_numpy_v4.astype('float32')
             #input_v5 = data_numpy_v5.astype('float32')
             #input_v6 = data_numpy_v6.astype('float32')

             return [input_v1, input_v2, input_v3], index#,input_v4,input_v5,input_v6], input_v1
    def basic_aug(self, data_numpy,number_of_frames):
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
        
        return data_numpy_v2

    def strong_aug(self, data_numpy,number_of_frames):
        data_numpy = self.basic_aug(data_numpy,number_of_frames)

        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy

class Feeder_single(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 video_name_path,
                 start_pos_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation='graph-based',
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.video_name_path = video_name_path
        self.start_pos_path = start_pos_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.l_ratio = l_ratio

        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')#N,T,V,C,M
        else:
            self.data = np.load(self.data_path)

        print(self.data.shape)
        N,T,V,C,M = self.data.shape
        #self.data = self.data.reshape(N,T,V,3,2)
        
        # load num of valid frame length
        self.label= np.load(self.label_path)#N,T
        print(self.label.max())
        with open(self.video_name_path, 'r') as f:
            self.video_name_path = f.readlines()
        
        self.start_pos_path = np.loadtxt(self.start_pos_path)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get raw input
        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
       
        label = self.label[index]
        if len(self.label) > 6500:
            start_pos = self.start_pos_path[0]
            video_name = self.video_name_path[0]
        else:
            start_pos = self.start_pos_path[index]
            video_name = self.video_name_path[index]

        input_data = data_numpy
        #label_mask = (label.astype(int)==255)
        #label = 0: no action label -> intervals
        #label = 255: no skeletons -> empty frames
        return input_data, label,start_pos,video_name.strip()


    def basic_aug(self, data_numpy,number_of_frames):
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
        
        return data_numpy_v2

class Feeder_single_sliding_window(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 video_name_path,
                 start_pos_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation='graph-based',
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.video_name_path = video_name_path
        self.start_pos_path = start_pos_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.l_ratio = l_ratio

        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')#N,T,V,C,M
        else:
            self.data = np.load(self.data_path)

        print(self.data.shape)
        N,T,V,C,M = self.data.shape
        #self.data = self.data.reshape(N,T,V,3,2)
        
        # load num of valid frame length
        self.label= np.load(self.label_path)#N,T
        print(self.label.max())
        with open(self.video_name_path, 'r') as f:
            self.video_name_path = f.readlines()
        
        self.start_pos_path = np.loadtxt(self.start_pos_path)

    def __len__(self):
        return self.N*16

    def __iter__(self):
        return self

    def __getitem__(self, index):
        index = index%self.N
        # get raw input
        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        window_size = 32
       
        label = self.label[index]
        if len(self.label) > 6500:
            start_pos = self.start_pos_path[0]
            video_name = self.video_name_path[0]
        else:
            start_pos = self.start_pos_path[index]
            video_name = self.video_name_path[index]

        input_data = data_numpy
        length = (label.astype(int)!=255).astype(float).sum()
        #print(length)
        sample_s = random.randint(0,length-1)
        if sample_s<window_size//2:
            sample_s = 0
            sample_e = window_size
        elif sample_s>length-(window_size//2):
            sample_s = length-window_size
            sample_e = length
        else:
            sample_s = sample_s - (window_size//2)
            sample_e = sample_s +  window_size

        #label = 0: no action label -> intervals
        #label = 255: no skeletons -> empty frames
        sample_s = int(sample_s)
        sample_e = int(sample_e)
        return input_data[sample_s:sample_e], label[(sample_s+sample_e)//2],start_pos,video_name.strip()


    def basic_aug(self, data_numpy,number_of_frames):
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
        
        return data_numpy_v2

class Feeder_single_sliding_window_downsample(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 video_name_path,
                 start_pos_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation='graph-based',
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.video_name_path = video_name_path
        self.start_pos_path = start_pos_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.l_ratio = l_ratio
        self.downsample = 4
        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')#N,T,V,C,M
        else:
            self.data = np.load(self.data_path)

        print(self.data.shape)
        N,T,V,C,M = self.data.shape
        #self.data = self.data.reshape(N,T,V,3,2)
        
        # load num of valid frame length
        self.label= np.load(self.label_path)#N,T
        print(self.label.max())
        with open(self.video_name_path, 'r') as f:
            self.video_name_path = f.readlines()
        
        self.start_pos_path = np.loadtxt(self.start_pos_path)

    def __len__(self):
        return self.N*16

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        self.downsample = random.randint(3,5)

        index = index%self.N
        # get raw input
        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        window_size = 16 * self.downsample
       
        label = self.label[index]
        if len(self.label) > 6500:
            start_pos = self.start_pos_path[0]
            video_name = self.video_name_path[0]
        else:
            start_pos = self.start_pos_path[index]
            video_name = self.video_name_path[index]
        
        label_path = '/mnt/netdisk/Datasets/088-PKUMMD/PKUMMDv1/Train_Label_PKU_final/'
        label_quad = np.loadtxt(os.path.join(label_path, video_name.strip()+'.txt'),delimiter=',').astype(int)#T,4 (label,start,end,confidence)
        
        input_data = data_numpy
        #255只会在label的最后以占位符的形式出现
        length = (label.astype(int)!=255).astype(float).sum()
        
        #print(length)
        sample_s = random.randint(0,length-1)
        if sample_s<window_size//2:
            sample_s = 0
            sample_e = window_size
        elif sample_s>length-(window_size//2):
            sample_s = length-window_size
            sample_e = length
        else:
            sample_s = sample_s - (window_size//2)
            sample_e = sample_s +  window_size

        #label = 0: no action label -> intervals
        #label = 255: no skeletons -> empty frames
        sample_s = int(sample_s)
        sample_e = int(sample_e)

        target = 0 #获取overlap最大的clip的label
        max_ratio = 0.5
        for action_id, start_frame, end_frame, _ in label_quad:
            overlap = get_overlap([sample_s, sample_e - 1],
                                [start_frame, end_frame - 1])
            ratio = overlap / (sample_e - sample_s)
            if ratio > max_ratio:
                target = int(action_id+1)
        
        return input_data[sample_s:sample_e:self.downsample], label[(sample_s+sample_e)//2], start_pos,video_name.strip()


    def basic_aug(self, data_numpy,number_of_frames):
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
        
        return data_numpy_v2

def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

class Feeder_single_eccv(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 video_name_path,
                 start_pos_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation='graph-based',
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.video_name_path = video_name_path
        self.start_pos_path = start_pos_path
        self.num_frame_path= num_frame_path
        self.downsample = 3
        self.n_frames = 10 # frame number in a clip
        self.timestep = 10 # clip number in a video
        self.step_size = 10 # size between two clips
        self.input_size=input_size
        self.input_representation=input_representation
        self.l_ratio = l_ratio

        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')#N,T,V,C,M
        else:
            self.data = np.load(self.data_path)

        print(self.data.shape)
        N,T,V,C,M = self.data.shape
        #self.data = self.data.reshape(N,T,V,3,2)
        
        # load num of valid frame length
        self.label= np.load(self.label_path)#N,T
        print(self.label.max())
        with open(self.video_name_path, 'r') as f:
            self.video_name_path = f.readlines()
        
        self.start_pos_path = np.loadtxt(self.start_pos_path)

    def __len__(self):
        return self.N*16

    def __iter__(self):
        return self

    def __getitem__(self, index):
        index = index%self.N
        # get raw input
        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        window_size = 32
       
        label = self.label[index]
        if len(self.label) > 6500:
            start_pos = self.start_pos_path[0]
            video_name = self.video_name_path[0]
        else:
            start_pos = self.start_pos_path[index]
            video_name = self.video_name_path[index]

        input_data = data_numpy
        length = (label.astype(int)!=255).astype(float).sum()
        #print(length)
        sample_s = random.randint(0,length-1)
        if sample_s<window_size//2:
            sample_s = 0
            sample_e = window_size
        elif sample_s>length-(window_size//2):
            sample_s = length-window_size
            sample_e = length
        else:
            sample_s = sample_s - (window_size//2)
            sample_e = sample_s +  window_size

        #label = 0: no action label -> intervals
        #label = 255: no skeletons -> empty frames
        sample_s = int(sample_s)
        sample_e = int(sample_e)
        return input_data[sample_s:sample_e], label[(sample_s+sample_e)//2],start_pos,video_name.strip()


    def basic_aug(self, data_numpy,number_of_frames):
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
        
        return data_numpy_v2