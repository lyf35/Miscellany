#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import random
from torch_geometric.data import Data
import torch


# In[2]:


CUTOFF_DISTANCE=6
BCC_lattice_positions=[
    [0.0,0.0,0.0],
    [0.5,0.5,0.5]
]
LATTICE_PARAMETER=3.0


# In[ ]:


def generate_reps(cutoff_dist,vec_1_len,vec_2_len,vec_3_len):
    vec_1_num=math.ceil(CUTOFF_DISTANCE/vec_1_len)
    vec_2_num=math.ceil(CUTOFF_DISTANCE/vec_2_len)
    vec_3_num=math.ceil(CUTOFF_DISTANCE/vec_3_len)
    vec_1_iter=[0]
    vec_2_iter=[0]
    vec_3_iter=[0]
    for i in range(vec_1_num):
        vec_1_iter.append(i+1)
        vec_1_iter.append(-i-1)
    for j in range(vec_2_num):
        vec_2_iter.append(j+1)
        vec_2_iter.append(-j-1)
    for k in range(vec_3_num):
        vec_3_iter.append(k+1)
        vec_3_iter.append(-k-1)
    return vec_1_iter,vec_2_iter,vec_3_iter


# In[ ]:


def pyg_graph_gen(atom_str,period_x,period_y,period_z,
                  cutoff=CUTOFF_DISTANCE,lattice_param=LATTICE_PARAMETER,lattice_pos=BCC_lattice_positions):
    #assert len(atom_str)==period_x*period_y*period_z*len(lattice_pos)
    assert len(atom_str)==period_x*period_y*period_z*len(lattice_pos)*4
    vec_1_x=period_x*lattice_param
    vec_1_y=0.0
    vec_1_z=0.0
    vec_2_x=0.0
    vec_2_y=period_y*lattice_param
    vec_2_z=0.0
    vec_3_x=0.0
    vec_3_y=0.0
    vec_3_z=period_z*lattice_param
    x_list=list()
    y_list=list()
    z_list=list()
    charge_list=list()
    atom_nums=period_x*period_y*period_z*len(lattice_pos)
    for j in range(period_x*period_y*period_z):
        cnt=j
        period_1=j%period_x
        j=j//period_x
        period_2=j%period_y
        period_3=j//period_y
        for k in range(len(lattice_pos)):
            charge_list.append(int(atom_str[j*len(lattice_pos)+k]))
            #x_list.append(lattice_pos[k][0]*lattice_param+period_1*lattice_param)
            #y_list.append(lattice_pos[k][1]*lattice_param+period_2*lattice_param)
            #z_list.append(lattice_pos[k][2]*lattice_param+period_3*lattice_param)
            temp=lattice_pos[k][0]*lattice_param+period_1*lattice_param+float(atom_str[atom_nums+(cnt*len(lattice_pos)+k)*3+0])
            if temp<0.0:
                temp+=vec_1_x
            if temp>vec_1_x:
                temp-=vec_1_x
            x_list.append(temp)
            temp=lattice_pos[k][1]*lattice_param+period_2*lattice_param+float(atom_str[atom_nums+(cnt*len(lattice_pos)+k)*3+1])
            if temp<0.0:
                temp+=vec_2_y
            if temp>vec_2_y:
                temp-=vec_2_y
            y_list.append(temp)
            temp=lattice_pos[k][2]*lattice_param+period_3*lattice_param+float(atom_str[atom_nums+(cnt*len(lattice_pos)+k)*3+2])
            if temp<0.0:
                temp+=vec_3_z
            if temp>vec_3_z:
                temp-=vec_3_z
            z_list.append(temp)
    
    vec_1_iter,vec_2_iter,vec_3_iter=generate_reps(cutoff,vec_1_x,vec_2_y,vec_3_z)
    len_vec1=len(vec_1_iter)
    len_vec2=len(vec_2_iter)
    len_vec3=len(vec_3_iter)
    period_vec_list=list()
    edge_list_u=list()
    edge_list_v=list()
    for m in range(atom_nums): #construct graph, considering the periodic boundaries at the same time
        for n in range(m):
            for period_num in range(len_vec1*len_vec2*len_vec3):
                temp=period_num
                vec_1_period=temp%len_vec1
                temp//=len_vec1
                vec_2_period=temp%len_vec2
                vec_3_period=temp//len_vec2
                x_dist=x_list[m]-x_list[n]-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x
                y_dist=y_list[m]-y_list[n]-vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y
                z_dist=z_list[m]-z_list[n]-vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z

                if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<CUTOFF_DISTANCE*CUTOFF_DISTANCE:

                    edge_list_u.append(n)
                    edge_list_v.append(m)
                    period_vec_list.append([-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x,
                                            -vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y,
                                            -vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z])
                    edge_list_u.append(m)
                    edge_list_v.append(n)
                    period_vec_list.append([vec_1_iter[vec_1_period]*vec_1_x+vec_2_iter[vec_2_period]*vec_2_x+vec_3_iter[vec_3_period]*vec_3_x,
                                            vec_1_iter[vec_1_period]*vec_1_y+vec_2_iter[vec_2_period]*vec_2_y+vec_3_iter[vec_3_period]*vec_3_y,
                                            vec_1_iter[vec_1_period]*vec_1_z+vec_2_iter[vec_2_period]*vec_2_z+vec_3_iter[vec_3_period]*vec_3_z])

        for period_num in range(len_vec1*len_vec2*len_vec3):
            if period_num==0:
                continue

            temp=period_num
            vec_1_period=temp%len_vec1
            temp//=len_vec1
            vec_2_period=temp%len_vec2
            vec_3_period=temp//len_vec2

            x_dist=-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x
            y_dist=-vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y
            z_dist=-vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z

            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<CUTOFF_DISTANCE*CUTOFF_DISTANCE:

                edge_list_u.append(m)
                edge_list_v.append(m)
                period_vec_list.append([-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x,
                                        -vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y,
                                        -vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z])

    data_temp=Data(x=torch.tensor(charge_list,dtype=torch.long),
                   pos=torch.tensor(np.array([x_list,y_list,z_list]).transpose(1,0),dtype=torch.float32,requires_grad=True),
                   edge_index=torch.tensor([edge_list_u,edge_list_v],dtype=torch.long),
                   edge_attr=torch.tensor(period_vec_list,dtype=torch.float32)
                  )
    return data_temp

if __name__=='__main__':
     print(' ')
