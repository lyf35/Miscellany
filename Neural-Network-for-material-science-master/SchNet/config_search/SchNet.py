#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
import torch_geometric
import torch
from torch_geometric.data import Data,DataLoader
import torch.nn as nn
import math
import datetime
from torch_geometric.nn import MessagePassing,global_mean_pool,global_add_pool
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
warnings.filterwarnings('ignore')


# The following is the definition of RBF layer. The result of RBF layer can be used by every filter generator layer, so it can be used only once in the architecture of Schnet to reduce calculation.

# In[2]:


class RBFLayer(nn.Module):
    def __init__(self,cutoff=6,gamma=0.1,rbfkernel_number=300):
        super(RBFLayer,self).__init__()
        self.cutoff=cutoff
        self.gamma=gamma
        self.rbfkernel_number=rbfkernel_number
        
    def forward(self,g):
        dist=torch.index_select(g.pos,0,g.edge_index[1])-torch.index_select(g.pos,0,g.edge_index[0])
        dist=torch.add(dist,g.edge_attr)
        dist=torch.norm(dist,p=2,dim=1,keepdim=True)
        centers=np.linspace(0,self.cutoff,self.rbfkernel_number,dtype=np.float32)
        rbf_kernel=torch.tensor(centers,device='cpu')
        rbf_tensor=dist-rbf_kernel
        rbf_tensor=torch.exp(-self.gamma*torch.mul(rbf_tensor,rbf_tensor))
        return rbf_tensor,dist


# The definition of Filter generator block. It is used to generate the weight of cfconv layer.

# In[3]:


class FilterGeneratorBlock(nn.Module):
    def __init__(self,rbfkernel_number=300,out_dim=64):
        super(FilterGeneratorBlock,self).__init__()
        self.linear1=nn.Linear(rbfkernel_number,out_dim)
        self.linear2=nn.Linear(out_dim,out_dim)
        
    def forward(self,rbf_tensor):
        weight=self.linear1(rbf_tensor)
        weight=torch.log(torch.exp(weight)+1.0)-torch.log(torch.tensor(2.0))
        weight=self.linear2(weight)
        weight=torch.log(torch.exp(weight)+1.0)-torch.log(torch.tensor(2.0))
        return weight


# The definition of cfconv layer, which is used for carring out the graph convolution operation of every atom.

# In[4]:


class cfconv(MessagePassing):
    def __init__(self,rbfkernel_number=300,feat_num=64,aggragate='add',node_flow='target_to_source'):
        super(cfconv,self).__init__(aggr=aggragate,flow=node_flow)
        self.filter_block=FilterGeneratorBlock(rbfkernel_number=rbfkernel_number,out_dim=feat_num)
     
    def message(self,x_j,weight):
        return torch.mul(x_j,weight)
    
    def update(self,aggr_out):
        return aggr_out
    
    def forward(self,x,edge_index,rbf_tensor,dist,cutoff):
        weight=self.filter_block(rbf_tensor)
        weight=weight*(1+torch.cos(3.14159265*dist/cutoff))
        return self.propagate(edge_index,x=x,weight=weight)


# The Interaction block is used to update node features.

# In[5]:


class InteractionBlock(nn.Module):
    def __init__(self,rbfkernel_number=300,feat_num=64):
        super(InteractionBlock,self).__init__()
        self.linear1=nn.Linear(feat_num,feat_num)
        self.cfconvlayer=cfconv(rbfkernel_number=rbfkernel_number,feat_num=feat_num)
        self.linear2=nn.Linear(feat_num,feat_num)
        self.linear3=nn.Linear(feat_num,feat_num)
    
    def forward(self,edge_index,node_feature,rbf_tensor,dist,cutoff):
        x=torch.tensor(node_feature)
        node_feature=self.linear1(node_feature)
        node_feature=self.cfconvlayer(node_feature,edge_index,rbf_tensor,dist,cutoff)
        node_feature=self.linear2(node_feature)
        node_feature=torch.log(torch.exp(node_feature)+1.0)-torch.log(torch.tensor(2.0))
        node_feature=self.linear3(node_feature)
        node_feature=torch.add(node_feature,x)
        return node_feature


# The architecture of Schnet.

# In[6]:


class Schnet(nn.Module):
    def __init__(self,cutoff=6,gamma=0.5,rbfkernel_number=300,hidden_layer_dimensions=64,num_conv=3):
        super(Schnet,self).__init__()
        self.num_conv=num_conv
        self.cutoff=cutoff
        
        self.rbf_layer=RBFLayer(cutoff=cutoff,gamma=gamma,rbfkernel_number=rbfkernel_number)
        self.embedding=nn.Embedding(3,hidden_layer_dimensions)
        self.interaction_blocks=nn.ModuleList([InteractionBlock(feat_num=hidden_layer_dimensions,
                                                                rbfkernel_number=rbfkernel_number) 
                                               for i in range(num_conv)])
#         self.interaction_blocks=InteractionBlock(feat_num=hidden_layer_dimensions,
#                                                 rbfkernel_number=rbfkernel_number)
        self.atomwise1=nn.Linear(hidden_layer_dimensions,int(hidden_layer_dimensions/2))
        self.atomwise2=nn.Linear(int(hidden_layer_dimensions/2),1)
        
    def forward(self,g):
        rbf_tensor,dist=self.rbf_layer(g)
        temp=self.embedding(g.x)
        for i in range(self.num_conv):
            temp=self.interaction_blocks[i](g.edge_index,temp,rbf_tensor,dist,self.cutoff)
#             temp=self.interaction_blocks(g.edge_index,temp,rbf_tensor,dist,self.cutoff)
        temp=self.atomwise1(temp)
        temp=torch.log(torch.exp(temp)+1.0)-torch.log(torch.tensor(2.0))
        temp=self.atomwise2(temp)
        temp=global_add_pool(temp,g.batch)
        return temp

if __name__=='__main__':
     print(' ')

