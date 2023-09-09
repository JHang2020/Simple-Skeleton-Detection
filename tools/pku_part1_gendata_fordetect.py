import os
import numpy as np
import math
from keras.utils import np_utils
from tqdm import tqdm
import json
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
    
    if ovr_num == None:
        ovr_num = num_seq//2
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
            skeleton = np.concatenate((skeleton1, skeleton2), axis=-1)#N,T,V,C,M
            
            labels = np.loadtxt(os.path.join(label_path, item+'.txt'),delimiter=',').astype(int)#T,4 (label,start,end,confidence)

            labels_pertime = np.zeros((skeleton.shape[0]), dtype=np.int32)

            for clip_idx in range(len(labels)):
                # NOTE: for detection, labels start from 1, 0 represents the empty clips for the input stream
                
                labels_pertime[labels[clip_idx][1]:labels[clip_idx][2]] = labels[clip_idx][0] + 1
            
            labels_pertime = labels_pertime.astype(np.int32)
            #labels_pertime = np_utils.to_categorical(labels_pertime,52)#转为one-hot向量 T,class

            
            if train:#train_set:
                for clip_idx in range(len(labels)):
                    # only sample clips centered at each action
                    if labels[clip_idx][1] > labels[clip_idx][2]:
                        temp = labels[clip_idx][2]
                        labels[clip_idx][2] = labels[clip_idx][1]
                        labels[clip_idx][1] = temp
                    pos1 = np.max([labels[clip_idx][1] - margin_rate*num_seq, 0])
                    pos2 = np.min([labels[clip_idx][2] + margin_rate*num_seq, skeleton.shape[0] ])
                    #从目标动作范围两侧各扩出一段
                    assert(skeleton.shape[0] >= labels[clip_idx][2] and pos1 < pos2)
                    skt = skeleton[pos1:pos2]
                    label_pt = labels_pertime[pos1:pos2]
                    if skt.shape[0] > num_seq:#如果这一段长度大于num_seq
                        start = 0
                        while start + num_seq < skt.shape[0]:
                            X.append(skt[start:start+num_seq])
                            Y.append(label_pt[start:start+num_seq])
                            vid_list.append(item)
                            start_list.append(start + pos1)
                            start = start + ovr_num
                            
                        X.append(skt[-num_seq:])
                        Y.append(label_pt[-num_seq:])
                        vid_list.append(item)
                        start_list.append(skeleton.shape[0]-num_seq)
                    else:
                        print (pos1, pos2, skt.shape)
                        if pos1 - (num_seq - skt.shape[0]) > 0:
                            pos1 = pos1 - (num_seq - skt.shape[0])
                        else:
                            pos2 = pos2 + (num_seq - skt.shape[0])
                        X.append(skeleton[pos1:pos2])
                        Y.append(labels_pertime[pos1:pos2])
                        vid_list.append(item)
                        start_list.append(pos1)
            else:
                if skeleton.shape[0] > num_seq:
                    start = 0
                    while start + num_seq < skeleton.shape[0]:
                        X.append(skeleton[start:start+num_seq])
                        Y.append(labels_pertime[start:start+num_seq])
                        vid_list.append(item)
                        start_list.append(start)
                        start = start + ovr_num
                    X.append(skeleton[-num_seq:])
                    Y.append(labels_pertime[-num_seq:])
                    vid_list.append(item)
                    start_list.append(skeleton.shape[0]-num_seq)
                else:
                    skeleton = np.concatenate((np.zeros((num_seq-skeleton.shape[0], skeleton.shape[1], skeleton.shape[2])), skeleton), axis=0)
                    labels_pertime = np.concatenate((np.zeros((num_seq-labels_pertime.shape[0])), labels_pertime), axis=0)
                    X.append(skeleton)
                    Y.append(labels_pertime)
                    vid_list.append(item)
                    start_list.append(0) 
                    
    X = np.asarray(X).astype(np.float32)
    Y = np.asarray(Y)
    print (X.shape, Y.shape)
    return X, Y, vid_list, start_list


if __name__ =='__main__':
    spilit_path = '/mnt/netdisk/zhangjh/Code/skeleton_detection/detection/tools/pkuv1_cross_subject.txt'
    data_path = '/mnt/netdisk/Datasets/088-PKUMMD/PKUMMDv1/PKU_Skeleton_Renew/'
    label_path = '/mnt/netdisk/Datasets/088-PKUMMD/PKUMMDv1/Train_Label_PKU_final/'
    '''
    X, Y, vid_list, start_list = load_seq_lst(spilit_path=spilit_path,data_path=data_path,label_path=label_path,num_seq=200,train=True)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/train_data.npy',X)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/train_label.npy',Y)
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/train_video_name.txt', 'w') as  f:
        for i in vid_list:
            f.write(i + '\n')
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/train_start_list.txt', 'w') as  f:
        for i in start_list:
            f.write(str(i))
            f.write('\n')
    '''

    X, Y, vid_list, start_list = load_seq_lst(spilit_path=spilit_path,data_path=data_path,label_path=label_path,num_seq=200,train=False)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/val_data.npy',X)
    np.save('/mnt/netdisk/zhangjh/data/PKU1_Detection/val_label.npy',Y)
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/val_video_name.txt', 'w') as  f:
        for i in vid_list:
            f.write(i + '\n')
    with open('/mnt/netdisk/zhangjh/data/PKU1_Detection/val_start_list.txt', 'w') as  f:
        for i in start_list:
            f.write(str(i))
            f.write('\n')
    
    

