import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchlight.torchlight import import_class
import random
from .gru import *


class GRU_Detect(nn.Module):
    def __init__(self,hidden_size=1024,num_class=60) -> None:
        super().__init__()
        #self.register_parameter('cl_prompt', nn.Parameter(torch.zeros((1,256,)),requires_grad=True) )
        #self.register_parameter('base_prompt', nn.Parameter(torch.zeros((1,1,150,)),requires_grad=True) )
        self.encoder_q = BIGRU(en_input_size=150, en_hidden_size=hidden_size, en_num_layers=3, num_class=num_class)
        self.fc = nn.Linear(hidden_size*2,num_class)

    def forward(self, x):
        N,T,V,C,M = x.shape
        x = x.reshape(N,T,-1)# N T VCM
        seq_feat = self.encoder_q(x)#, prompt=self.base_prompt,small_prompt=self.cl_prompt)#N,T,C
        seq_feat = seq_feat.reshape(N*T,-1)
        logits = self.fc(seq_feat)
        logits = logits.reshape(N,T,-1)

        return logits#N,T,num_class


        
