"""
Title: 
Authors:
- Lee Sael (sael@ajou.ac.kr) Ajou University
- Sang Suk Lee, Ajou University
- HeaWon Moon, Ajou University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import numpy as np
import tensorly as tl
#import os
import matplotlib.pyplot as plt
#from tensorly.base import unfold
from sklearn.utils.extmath import randomized_svd
#from scipy.linalg import block_diag
import copy
import torch
import time

# local imports
#from src.tensor import *
from src.tensor_edit import * 

#DEBUG=True
DEBUG=False


def slice_matrix_svd(tensor, rank=10):
    """make slice matrix and generate svd with default rank 10"""
    shape = np.asarray(np.shape(tensor))
    # mode-1 matrixization of input tensor 

    N = shape[1]
    M = np.prod(shape[2:])
    
    # new matrix
    u_lst = []
    s_lst = []
    vh_lst = []
    # make frontal slice 
    #unfold_1 = tl.base.unfold(tensor,0)
    unfold_1 = unfold_F(tensor,0)
    
    for myslice in [unfold_1[:, N * i:N * (i + 1)] for i in range(M)]:
    #TODO: currently only works for 3D tensor 
    #for myslice in [tensor[:,:,i] for i in range(M)]:
        #if(DEBUG): print (f'Slice\n', myslice)
        u, s, vh = randomized_svd(myslice,rank)
        u_lst.append(u)
        s_lst.append(s)
        vh_lst.append(vh)
    return u_lst, s_lst, vh_lst

def bidtucker_storage(tensor_stream, svd_rank=20):
                
    old_u = []
    old_s = []
    old_vt = []
    cusize = [0] 
    for b in range(len(tensor_stream)):
        '''Storage phase of stream'''
        block = tensor_stream[b]
        u,s,vt = slice_matrix_svd(block, rank=svd_rank)
        old_u.append(u)
        old_s.append(s)
        old_vt.append(vt)
        cusize.append(cusize[-1] + u[0].shape[0])
        
        if(DEBUG): #reconstruct svd to check accuracy 
            recon = [np.matmul(np.matmul(u[i],np.diag(s[i])),vt[i]) for i in range(len(s))]
            new_var = np.concatenate(recon, 1)
            new_var1 = np.concatenate([block[:,:,i] for i in range(len(s))],1)
            svdnorm = np.linalg.norm(new_var1 - new_var)
            print(f"Storage SVD diff norm {svdnorm}.")

    return old_u, old_s, old_vt, cusize




def _bidtucker_query_init(old_u ,old_s, old_vt, list_reduce, tensor_size,
        _orgten=[], svd_rank=20,  start_t=0, end_t=0):
    if(start_t==0 and end_t==0):
        new_u = old_u
        new_s = old_s
        new_vt =old_vt
    else:
        block_sizes = [len(old_u[u][0]) for u in range(len(old_u))]
        total_size = np.sum(block_sizes)
        num_slices = len(old_u[0])
        if( end_t == 0 ): end_t = total_size 
        S = max(-1,(start_t - block_sizes[0])) // block_sizes[1]  + 1       #include block S
        E = max(-1,(end_t  - 1  - block_sizes[0])) // block_sizes[1]  + 1    #include block E (not include end_t)
        new_u = old_u[S:E+1]
        new_s = old_s[S:E+1]
        new_vt =old_vt[S:E+1]

    stch_u = []
    stch_s = []
    stch_vt = []
    num_slice = len(new_u[0])
    rank_sizes = [a[0].shape[1] for a in new_u]
    rank_cumsum = np.cumsum(rank_sizes)
    block_sizes = [a[0].shape[0] for a in new_u]
    # for each slice 
    for k in range(num_slice):
        # for each block 
        SVs = [np.matmul(np.diag(new_s[b][k]),new_vt[b][k]) for b in range(len(new_vt))]
        cat_sv = np.concatenate((SVs), axis=0 )
        u, s, vt = randomized_svd(cat_sv, svd_rank) 
        stch_s.append(s)
        stch_vt.append(vt)
        # blockwise split Us to find new u 
        
        u_s = np.matmul(new_u[0][k], u[0:rank_sizes[0],:])
        for bb in range(1,len(new_u)): 
            new_var1 = new_u[bb][k]
            new_var2 = u[rank_cumsum[bb-1]:rank_cumsum[bb],:]
            new_var = np.matmul(new_var1,new_var2)
            u_s = np.concatenate((u_s, new_var))
        
        stch_u.append(u_s)
 
    # initialize factor matrices
    iter_factor = dt_query_init(stch_u , stch_s, stch_vt, list_reduce, tensor_size, _orgten=_orgten)
    
    return stch_u , stch_s, stch_vt, iter_factor
    
   
    
def _bidtucker_query_iter(temp_U,temp_S,temp_VT, iter_factor, list_reduce, tensor_size, iter_thresh=0.0001,  iteration = 5, _orgten=[]):
    Y_tensor, iter_factor = dt_query_iter(temp_U,temp_S,temp_VT, iter_factor, list_reduce, 
                                          tensor_size, iter_thresh=iter_thresh, iteration = iteration, _orgten=_orgten)
    return Y_tensor, iter_factor

def time2blocktime(start_t, end_t, block_sizes):
    st=start_t
    et=end_t
    if(start_t + end_t > 0):
        if( end_t == 0 ): end_t = total_size 
        S = max(-1,(start_t - block_sizes[0])) // block_sizes[1]  + 1       #include block S
        E = max(-1,(end_t  - 1  - block_sizes[0])) // block_sizes[1]  + 1    #include block E (not include end_t)
        if(S==0): st = 0
        if(E==0): et = block_sizes[0]-1
        if(S>0) : st = block_sizes[0] + (S-1)*block_sizes[1]    # included
        if(E>0) : et = block_sizes[0] + E*block_sizes[1]        # not included
    return st, et
        

def bidtucker_query(old_u,old_s,old_vt, list_reduce, tensor_size, start_t=0, end_t=0, svd_rank=20, 
              iteration=5, iter_thresh = 0.0001, _orgten=[]):

    # slice matrics svd
 #   temp_U = old_u
 #   temp_S = old_s
 #   temp_VT = old_vt
    order = len(tensor_size)
    block_sizes = [len(old_u[u][0]) for u in range(len(old_u))]
    total_size = np.sum(block_sizes)
    num_slices = len(old_u[0])
    st=start_t
    et=end_t
    if(start_t + end_t > 0):
        st,et = time2blocktime(start_t, end_t, block_sizes)
        tensor_size[0] = et - st
        if(len(_orgten)>0): 
            _orgten = _orgten[st:et,...]
        

    '''step 2. (Algorithm 5) init factor matrics create''' 
    temp_U, temp_S, temp_VT, init_factor = _bidtucker_query_init(old_u ,old_s, old_vt, list_reduce,
            tensor_size, _orgten =_orgten, svd_rank=svd_rank, start_t=st, end_t=et)


    '''step 3. (Algorithm 6) iteration and converge error'''
    Y_tensor, iter_factor = _bidtucker_query_iter(temp_U ,temp_S,temp_VT, init_factor, list_reduce, tensor_size, 
                                            iteration=iteration, iter_thresh=iter_thresh, _orgten=_orgten)
    
    '''core update '''
    for ord in range(2, order):
        Y_tensor = tl.tenalg.mode_dot(Y_tensor, iter_factor[ord].T, ord)

    return Y_tensor, iter_factor, block_sizes


 
def dt_query_init(temp_U ,temp_S,temp_VT, list_reduce, tensor_size, _orgten=[]):
    '''
    Step 2. (Algorithm 6) init factor matrics create 
    '''
    # slice matrics svd
    order = len(tensor_size) #test=3
    num_slice = len(temp_U) #test=3
    init_factor = [a for a in range(order)]

    # factor 1 init: line 1-2 of Algo 6
    factor1 = [np.matmul(temp_U[k],np.diag(temp_S[k])) for k in range(num_slice)]
    factor2 = np.concatenate(factor1 , 1)
    init_factor[0] = np.linalg.svd(factor2,full_matrices=False)[0][:,:list_reduce[0]] 

    # factor 2 init: line 3-6 of Algo 6
    
    Y2_inter = np.matmul(init_factor[0].T, np.concatenate(temp_U,1))

    blksv = [torch.DoubleTensor(np.matmul(np.diag(temp_S[k]),temp_VT[k])) for k in range(num_slice)] 
    Y_prev = np.matmul(Y2_inter, torch.block_diag(*blksv).numpy())   

    newsize = [list_reduce[0]]
    newsize.extend(tensor_size[1:]) 
    Y_tensor = np.reshape(Y_prev, newsize, order='F')
    # may need to make more efficient using numpy truncatedSVD
#    init_factor[1] = np.linalg.svd(tl.base.unfold(Y_tensor, 1),full_matrices=False)[0][:, :list_reduce[1]]
    init_factor[1] = np.linalg.svd(unfold_F(Y_tensor, 1),full_matrices=False)[0][:, :list_reduce[1]]

    # Others factor init: line 7-15 of Algo 6
    for i in range(2, order):
        if i == 2:
            blksv = [torch.DoubleTensor(np.matmul(np.diag(temp_S[a]),temp_VT[a])) for a in range(num_slice)]
            blksv = torch.block_diag(*blksv).numpy()
            Y_prev = np.reshape(np.matmul(Y2_inter, blksv), newsize, order='F') 
#            Y_tensor = tl.tenalg.mode_dot(Y_prev, init_factor[1].T, 1)
            Y_tensor = mode_dot_F(Y_prev, init_factor[1].T, 1)
        else:
#            Y_tensor = tl.tenalg.mode_dot(Y_tensor,  init_factor[i - 1].T, i-1)
            Y_tensor = mode_dot_F(Y_tensor,  init_factor[i - 1].T, i-1)

#        init_factor[i] = np.linalg.svd(tl.base.unfold(Y_tensor, i))[0][:, :list_reduce[i]]
        init_factor[i] = np.linalg.svd(unfold_F(Y_tensor, i))[0][:, :list_reduce[i]]

    if(DEBUG): 
        print("init factor matrics complete... testing Reconstruction")
        if _orgten.all()!=None: 
            print_recon_error(Y_tensor, init_factor, _orgten, ord_lst=[a for a in range(2, order)])
                     
    return init_factor 



def dt_query_iter(temp_U,temp_S,temp_VT, iter_factor, list_reduce, tensor_size, 
                  iteration = 5, iter_thresh= 0.0001, _orgten=[]):
    if(DEBUG): print("#################### iteration part ########################")

    order = len(tensor_size) #test=3
    num_slice = len(temp_U) #test=3
    iter_count = 0
    L = len(temp_S)
    order = len(tensor_size)
    cat_temp_V = np.concatenate([temp_VT[a].T for a in range(L)],1)
    SUT = [torch.DoubleTensor(np.matmul(np.diag(temp_S[k]),temp_U[k].T)) for k in range(L)] 
    SVT = [torch.DoubleTensor(np.matmul(np.diag(temp_S[k]),temp_VT[k])) for k in range(L)] 
    # TODO:need to update stoping criterion 
    prev_tensor = [] 
    error = 1000000
    
    while iter_count <= iteration and iter_thresh < error:
        if(DEBUG): print("iteration:", iter_count)
        Y_tensor = []
        # update first two factors 
        for ord in range(2):
            # factor1 update (line 3-5 in Algo 7) 
            if ord == 0:
                Y1_inter = np.matmul(iter_factor[1].T, cat_temp_V)
                Y1_inter = np.matmul(Y1_inter, torch.block_diag(*SUT).numpy())    
                newsize = [tensor_size[0], list_reduce[1]]
                newsize.extend(tensor_size[2:]) 
                Y_tensor = np.reshape(Y1_inter, newsize,order='F')

            # factor2 update (line 7-9 Algo 7)
            elif ord == 1:
                Y_prev = np.matmul(iter_factor[0].T, np.concatenate(temp_U,1))
                Y2_inter = np.matmul(Y_prev, torch.block_diag(*SVT).numpy())    
                newsize = [list_reduce[0]]
                newsize.extend(tensor_size[1:]) 
                Y_tensor = np.reshape(Y2_inter, newsize,order='F')
            
        # line 10 Algo 7
        for i in range (2, order): 
            Y_tensor = mode_dot_F(Y_tensor,  iter_factor[i].T, i)
        # line 11 Algo 7
        iter_factor[ord] = np.linalg.svd(unfold_F(Y_tensor, ord))[0][:, :list_reduce[ord]]

        # update other factors (lines 13-18)
        blksv = [torch.DoubleTensor(np.matmul(np.matmul(np.diag(temp_S[a]),temp_VT[a]), iter_factor[1]))
                     for a in range(L)]
        Y_prev =  np.matmul(Y_prev, torch.block_diag(*blksv).numpy())
        newsize = list_reduce[0:2]
        newsize.extend(tensor_size[2:])
        Y_tensor = np.reshape(Y_prev, newsize,order='F')
        for ord in range(2, order):
            Y_t = Y_tensor
            for i in range(2, order):
                if i != ord:  
                    Y_t = mode_dot_F(Y_t,  iter_factor[i].T, i)
            # line 17 Algo 7
            iter_factor[ord] = np.linalg.svd(unfold_F(Y_t, ord))[0][:, :list_reduce[ord]]
        
        error, prev_tensor = print_recon_error(Y_tensor, iter_factor, prev_tensor, ord_lst=[a for a in range(2, order)])
        if(DEBUG): print(f'iteration {iter_count}  difference {error}')
        iter_count += 1


        if(DEBUG): 
            print(f'iteration {iter_count} complete... testing Reconstruction')
            if _orgten.all()!=None: 
                print_recon_error(Y_tensor, iter_factor, _orgten, ord_lst=[a for a in range(2, order)])

    return Y_tensor, iter_factor

