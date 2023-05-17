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
from sklearn.utils.extmath import randomized_svd
#from scipy.linalg import block_diag
import torch
import time
# local imports
#from src.tensor import *
from src.tensor_edit import *
from src.bidtucker import *

#DEBUG=True
DEBUG=False

def d_tucker(tensor, list_reduce, tensor_size, svd_rank=20):

    order = len(tensor_size) 

    ###################################
    # step 1... slide matrics SVD
    ###################################
    temp_U, temp_S, temp_VT = slice_matrix_svd(tensor, rank=svd_rank)

    #######################################
    # step 2... init factor matrics create
    #######################################
    init_factor = dt_query_init(temp_U ,temp_S, temp_VT, list_reduce, tensor_size, _orgten=tensor)

    #########################################
    # step 3... iteration and converge error
    #########################################
    Y_tensor, iter_factor = dt_query_iter(temp_U, temp_S, temp_VT, init_factor, list_reduce, tensor_size, iteration = 5, _orgten=tensor)
    
    ############## core update ############
    for ord in range(2, order):
        Y_tensor = tl.tenalg.mode_dot(Y_tensor, iter_factor[ord].T, ord)

    return Y_tensor, iter_factor


# USED? 
