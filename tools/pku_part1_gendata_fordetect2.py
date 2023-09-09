import os
import numpy as np
import math
from keras.utils import np_utils
from tqdm import tqdm
import json
#不分割
def load_filename_list(path):
    with open(path,'r') as f:
        content = f.readlines()
        assert 'Training' in content[0] and 'Validataion' in content[2]
        train_list = content[1].strip().split(', ')
        val_list = content[3].strip().split(', ')
        train_list[-1] = train_list[-1].strip(',')
        val_list[-1] = val_list[-1].strip(',')
        print('Length of traing data', len(train_list))
        print('Length of validation data', len(val_list))
    return train_list, val_list

def load_seq_lst(spilit_path, data_path, label_path, num_seq, train=False, ovr_num=None, margin_rate=1):
    # the reverse process of load_sequence() method in pku_dataset class 
    # the output is a list of skeleton sequence of variable length
    # for training set, samples clips around intervals of actions
    
    train_list, val_list = load_filename_list(spilit_path)
    
    X = []
    Y = []
    vid_list = []
    start_list = []

    if train:
        keyname_lst = train_list
    else:
        keyname_lst = val_list
    
    if 1: 
        for item in tqdm(keyname_lst):
            skeleton = np.loadtxt(os.path.join(data_path, item+'.txt')) #T, 150
            # skeleton = skeleton.reshape((skeleton.shape[0], self._num_joints, self._dim_point ))
            skeleton1 = skeleton[:, 0:75].reshape((skeleton.shape[0], 25, 3 ))
            skeleton2 = skeleton[:, 75:].reshape((skeleton.shape[0], 25, 3 ))
            skeleton = np.stack((skeleton1, skeleton2), axis=-1)#T,V,C,M
            
            labels = np.loadtxt(os.path.join(label_path, item+'.txt'),delimiter=',').astype(int)#clip_num,4 (label,start,end,confidence)

            labels_pertime = np.zeros((skeleton.shape[0]), dtype=np.int32)

            for clip_idx in range(len(labels)):
                # NOTE: for detection, labels start from 1, 0 represents the empty clips for the input stream
                
                labels_pertime[labels[clip_idx][1]:labels[clip_idx][2]] = labels[clip_idx][0] + 1
            
            labels_pertime = labels_pertime.astype(np.int32)
            #labels_pertime = np_utils.to_categorical(labels_pertime,52)#转为one-hot向量 T,class
            
            X.append(skeleton)
            Y.append(labels_pertime)
            vid_list.append(item)
            start_list.append(0)
    final_X = []
    final_Y = []
    max_frame = 0
    for i in range(len(X)):
        max_frame = max(max_frame,X[i].shape[0])
    for i in range(len(X)):
        X_tem = np.zeros((max_frame,X[i].shape[1],X[i].shape[2],X[i].shape[3]))
        Y_tem = np.zeros((max_frame))+255
        act_length = len(X[i])
        X_tem[0:act_length] = X[i][0:act_length]
        Y_tem[0:act_length] = Y[i][0:act_length]
        final_X.append(X_tem)
        final_Y.append(Y_tem)
    X = np.asarray(final_X).astype(np.float32)
    Y = np.asarray(final_Y)
    print (X.shape, Y.shape)
    return X, Y, vid_list, start_list


if __name__ =='__main__':
    spilit_path = '/mnt/netdisk/zhangjh/Code/skeleton_detection/detection/tools/pkuv1_cross_subject.txt'
    data_path = '/mnt/netdisk/Datasets/088-PKUMMD/PKUMMDv1/PKU_Skeleton_Renew/'
    label_path = '/mnt/netdisk/Datasets/088-PKUMMD/PKUMMDv1/Train_Label_PKU_final/'
    
    X, Y, vid_list, start_list = load_seq_lst(spilit_path=spilit_path,data_path=data_path,label_path=label_path,num_seq=200,train=True)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/train_data.npy',X)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/train_label.npy',Y)
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/train_video_name.txt', 'w') as  f:
        for i in vid_list:
            f.write(i + '\n')
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/train_start_list.txt', 'w') as  f:
        for i in start_list:
            f.write(str(i))
            f.write('\n')
    
    '''
    X, Y, vid_list, start_list = load_seq_lst(spilit_path=spilit_path,data_path=data_path,label_path=label_path,num_seq=200,train=False)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/val_data.npy',X)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/val_label.npy',Y)
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/val_video_name.txt', 'w') as  f:
        for i in vid_list:
            f.write(i + '\n')
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/untrimmed/val_start_list.txt', 'w') as  f:
        for i in start_list:
            f.write(str(i))
            f.write('\n')
    '''
    
    

