#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# torchlight
import torchlight.torchlight as torchlight
from torchlight.torchlight import str2bool
from torchlight.torchlight import DictAction
from torchlight.torchlight import import_class
from .evaluate_metrics import *
from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DT_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch']) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        if k==1:
            self.test_acc = int(accuracy*100000)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
    
    def train(self,epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        process = tqdm(loader)
        for data, label, start_pos, video_name in process:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)#N,T
            
            # forward
            output = self.model(data)
            N,T,_ = output.shape
            out4loss = output.reshape(N*T,-1)
            label_loss = label.reshape(N*T,)
            label_mask = label_loss!=255
            loss = self.loss(out4loss[label_mask], label_loss[label_mask])

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def merge_list(self, vid_list):
        group_list = []
        idx_per = []
        vid_list_uqk = []
        for idx, name in enumerate(vid_list):
            if idx == len(vid_list)-1:
                last_name = ''
            else:
                last_name = vid_list[idx+1]
            if name != last_name:
                idx_per.append(idx)
                group_list.append(np.asarray(idx_per).astype(int) )
                vid_list_uqk.append(name)
                idx_per = []
            else:
                idx_per.append(idx)
        return group_list, vid_list_uqk

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        start_frag = []
        video_name_frag = []
        process = tqdm(loader)
        sfm = nn.Softmax(dim=-1)
        for data, label,start_pos,video_name in process:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
                N,T,_ = output.shape
                #out4loss = output.reshape(N*T,-1)
                #loss = self.loss(out4loss, label.reshape(N*T,))
            output = sfm(output)
            result_frag.append(output)
            label_frag.append(label)
            start_frag = start_frag + start_pos.cpu().numpy().tolist()
            #print(video_name)
            video_name_frag = video_name_frag + list(video_name)
        
        start_list = np.array(start_frag).astype(int)
        prob_seq = torch.cat(result_frag,dim=0)
        
        gt_dict = {}
        res_dict = {}
        # print vid_name, prob_seq.shape
        for idx in range(len(video_name_frag)):
            prob_val = prob_seq[idx].cpu().numpy()
            pred_labels = get_interval_frm_frame_predict(prob_val) 
            vid_name = video_name_frag[idx]

            label_path = '/mnt/netdisk/Datasets/088-PKUMMD/PKUMMDv1/Train_Label_PKU_final/'
            labels = np.loadtxt(os.path.join(label_path, vid_name+'.txt'),delimiter=',').astype(int)#T,4 (label,start,end,confidence)
            # NOTE: for ground truth, action labels starts from 0, but for our detection, 0 indicates empty action
            labels[:,0] = labels[:,0] + 1

            gt_dict[vid_name] = labels
            res_dict[vid_name] = pred_labels


        metrics = eval_detect_mAP(gt_dict, res_dict, minoverlap=0.5)
        self.current_result = metrics['map']*100
        self.best_result = max(self.best_result,self.current_result)
        print (metrics['map'])


        
        
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
