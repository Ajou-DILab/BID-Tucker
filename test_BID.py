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
from tensorly.base import unfold
from sklearn.utils.extmath import randomized_svd
#from scipy.linalg import block_diag
import copy
import torch
import time
# local imports
from src.bidtucker import *
from src.dtucker import *

AIRQSMALL='data/small-AirQ-org.npy'
#AIRQFILE='data/AirQ-org.npy'
AIRQFILE=AIRQSMALL
#DEBUG=True
DEBUG=False
#SVDRANK=20
#TRANK=[30,30,6]
SVDRANK=2
TRANK=[2,2,2]
MAXITER=10
BLOCK_SIZE = 4000
STREAM_SIZE = 24*7   # 24 hours 1 week
ITERTH = 0.00001


def main():
    bidtucker_test()
    bidtucker_store()
    
    e,t = bidtucker_ranmdomized_rank_test()
    print(f'randomized rank test error {e} time {t}') 

     # does not work for small tensors 
#    e,t = bidtucker_tucker_rank_test()
#    print(f'bidtucker tucker rank test error {e} time {t}')

    a,b,c,d = bidtucker_stream_test(['bidtucker','d_tucker','tucker'], trank=TRANK)


## d-tucker and bidtucker test
def bidtucker_test():
    org_tensor = tl.tensor(np.arange(23*4*6).reshape((23, 4, 6)))
    org_shape = list(org_tensor.shape)
    tensor_stream = create_tensor_stream(org_tensor, -5, svd_rank=2)
    old_u, old_s, old_vt, cusize = bidtucker_storage(tensor_stream, 2)
    start_t = 4
    end_t = 14

    start = time.time()
    core,factor,block_sizes = bidtucker_query(old_u, old_s, old_vt,[2,2,2], org_shape, start_t=start_t, end_t=end_t, 
                            svd_rank=2,iter_thresh=ITERTH, iteration=MAXITER, _orgten=org_tensor)
    end = time.time()

    new_tensor = multi_mode_dot_F(core, factor)

    st,et=time2blocktime(start_t, end_t, block_sizes)
    error = np.linalg.norm(org_tensor[st:et,...] - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
    print(f'BIDTuckerucker Err/Time:\t{error}\t{end-start}')


def bidtucker_store():
    org_tensor = np.load(AIRQFILE)
    tensor_stream = create_tensor_stream(org_tensor, -STREAM_SIZE, svd_rank=SVDRANK)
    org_shape = list(org_tensor.shape)           
    old_u, old_s, old_vt, cusize = bidtucker_storage(tensor_stream, SVDRANK)
    
    # QUERY ----------------------------------
    # for test let's stitch the whole range TODO: old an new should be same for block 1
    #new_u, new_s, new_vt = stitchSVD(old_u,old_s,old_vt,svd_rank=SVDRANK)

    start = time.time()
    core,factor,block_sizes = bidtucker_query(old_u, old_s, old_vt, TRANK, org_shape,
            svd_rank=SVDRANK,iter_thresh=ITERTH, iteration=MAXITER, _orgten=org_tensor)
    end = time.time()
#    new_tensor = tl.tenalg.multi_mode_dot(core, factor)
    new_tensor = multi_mode_dot_F(core, factor)
    error = np.linalg.norm(org_tensor - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
    print(f'BIDTuckerucker Err/Time:\t{error}\t{end-start}')
    np.save(f'out/bidtucker_{TRANK}_AirQ_core.npy', core)
    for ord in range(len(factor)):
        np.save(f'out/bidtucker_{TRANK}_AirQ_factor{ord}.npy', factor[ord])
    
# bidtucker randomized-rank test
def bidtucker_ranmdomized_rank_test():
    error_bidtucker = []
    time_bidtucker = []

    # 0:loc, 1:type, 2:time   
    svdranks = [a for a in range(5,55,5)]
    for svdrank in svdranks:
        org_tensor = np.load(AIRQFILE)
        tensor_stream = create_tensor_stream(org_tensor, -STREAM_SIZE, svd_rank=svdrank)
        org_shape = list(org_tensor.shape)           
        old_u, old_s, old_vt, cusize = bidtucker_storage(tensor_stream, svdrank)
        
        # QUERY ----------------------------------
        # for test let's stitch the whole range TODO: old an new should be same for block 1
        #new_u, new_s, new_vt = stitchSVD(old_u,old_s,old_vt,svd_rank=svdrank)

        start = time.time()
        core,factor,block_sizes = bidtucker_query(old_u, old_s, old_vt,TRANK, org_shape,
                svd_rank=svdrank, iter_thresh=ITERTH, iteration=MAXITER, _orgten=org_tensor)
        end = time.time()
        total_time = end-start
    #    new_tensor = tl.tenalg.multi_mode_dot(core, factor)
        new_tensor = multi_mode_dot_F(core, factor)
        error = np.linalg.norm(org_tensor - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
        #if(DEBUG):
        #print(f'org tensor {org_tensor}\nrecon tensor {new_tensor}')
        #print ("BIDTucker: orgnorm: ", np.linalg.norm(org_tensor), "org-new norm:", np.linalg.norm(org_tensor - new_tensor))
        print(f'BIDTuckerucker Err/Time:\t{error}\t{total_time}')

        error_bidtucker.append(error)
        time_bidtucker.append(end - start)

    np.savetxt('out/bidtucker_SVDRANK_out.txt', np.column_stack((svdranks, error_bidtucker, time_bidtucker)), delimiter='\t', fmt='%10.5f')

    
    fig, ax1 = plt.subplots()
    ax1.plot(svdranks, error_bidtucker,  'bo-', label='error')
    ax1.set_xlabel("SVD Trunkation Size")
    ax1.set_ylabel("Normalized Reconstruction Error")
    ax1.tick_params(axis='y')
    #ax1.ylim([0, 1.0])     # Y axis range: [ymin, ymax]
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Query Time")
    ax2.plot(svdranks, time_bidtucker,  'r^--', label='time')
    #ax2.ylim([0, 1.0])     # Y axis range: [ymin, ymax]
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.legend()
    plt.savefig("out/bidtucker_SVDranktest.pdf", format="pdf", bbox_inches="tight")

    #plt.show()

    return error_bidtucker,time_bidtucker


# bidtucker dimension test
def bidtucker_tucker_rank_test():
    error_bidtucker = []
    time_bidtucker = []

    ranks = [a for a in range(5,85,5)]
    org_tensor = np.load(AIRQFILE)

    for trank in ranks:
        trank = [trank, trank, 6]
        tensor_stream = create_tensor_stream(org_tensor, -STREAM_SIZE, svd_rank=SVDRANK)
        org_shape = list(org_tensor.shape)
        old_u, old_s, old_vt, cusize = bidtucker_storage(tensor_stream, SVDRANK)

        # QUERY ----------------------------------
        # for test let's stitch the whole range TODO: old an new should be same for block 1
        #new_u, new_s, new_vt = stitchSVD(old_u,old_s,old_vt,svd_rank=SVDRANK)

        start = time.time()
        core,factor,block_sizes = bidtucker_query(old_u, old_s, old_vt, trank, org_shape,
                svd_rank=SVDRANK, iter_thresh=ITERTH, iteration=MAXITER, _orgten=org_tensor)
        end = time.time()
        total_time = end-start

        new_tensor = multi_mode_dot_F(core, factor)
        error = np.linalg.norm(org_tensor - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
        print(f'BIDTucker {trank} Err/Time:\t{error}\t{total_time}')
        error_bidtucker.append(error)
        time_bidtucker.append(end-start)


    np.savetxt('out/bidtucker_tucker_rank_out.txt', np.column_stack((ranks, error_bidtucker, time_bidtucker)), delimiter='\t', fmt='%10.5f')

    
    fig, ax1 = plt.subplots()
    ax1.plot(ranks, error_bidtucker,  'bo-', label='error')
    ax1.set_xlabel("Tucker Rank [x,x,6]")
    ax1.set_ylabel("Normalized Reconstruction Error")
    ax1.tick_params(axis='y')
    #ax1.ylim([0, 1.0])     # Y axis range: [ymin, ymax]
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Query Time")
    ax2.plot(ranks, time_bidtucker,  'r^--', label='time')
    #ax2.ylim([0, 1.0])     # Y axis range: [ymin, ymax]
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    plt.savefig("out/bidtucker_tuckerranktest.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

#    return 0

    return error_bidtucker,time_bidtucker


def bidtucker_stream_test(method_list=['bidtucker', 'd_tucker', 'tucker'],trank=TRANK):
    error_tucker = []
    time_tucker = []
    error_d_tucker = []
    time_d_tucker = []
    error_bidtucker = []
    time_bidtucker = []

    org_tensor = np.load(AIRQFILE)
    org_shape = org_tensor.shape
    print(f'input tensor shape {org_shape}')
    org_tensor =[]
    data_block = [(k+1)*BLOCK_SIZE for k in range(org_shape[0]//BLOCK_SIZE)]
    data_block.append(org_shape[0])
    
    for cut in data_block:

        if(DEBUG): print("##############################cut({})####################".format(cut))
        for method in method_list:
            org_tensor = np.load(AIRQFILE)[:cut+1,:,:]
            if(DEBUG): print(org_tensor.shape)
            
            if method == 'tucker':
                start = time.time()
                core, factor = tl.decomposition.tucker(org_tensor, TRANK, n_iter_max=MAXITER)
                end = time.time()
                total_time = end-start

            #    new_tensor = tl.tenalg.multi_mode_dot(tcore, tfactor)
                new_tensor = multi_mode_dot_F(core, factor)
                error = np.linalg.norm(org_tensor - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
                print(f'Tucker{cut} Err/Time:\t{error}\t{total_time}')
                error_tucker.append(error)
                time_tucker.append(end-start)


            if method == 'd_tucker':
                start = time.time()
                core,factor = d_tucker(org_tensor,TRANK,list(org_tensor.shape), svd_rank=SVDRANK)
                end = time.time()
                total_time = end - start
                for k in range(len(factor)):
                    if k == 0:
                        new_tensor = mode_dot_F(core, factor[k], k)
                    else:
                        new_tensor = mode_dot_F(new_tensor, factor[k], k)
                error = np.linalg.norm(org_tensor - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
                print(f'DTucker{cut} Err/Time:\t{error}\t{total_time}')
                error_d_tucker.append(error)
                time_d_tucker.append(end-start)
            

            if method == 'bidtucker':
                org_shape = list(org_tensor.shape)
                tensor_stream = create_tensor_stream(org_tensor, -STREAM_SIZE, svd_rank=SVDRANK) 

                old_u, old_s, old_vt, cusize = bidtucker_storage(tensor_stream, SVDRANK)
    
                # QUERY ----------------------------------
                # for test let's stitch the whole range TODO: old an new should be same for block 1
                
                start = time.time()
                #new_u, new_s, new_vt = stitchSVD(old_u,old_s,old_vt,svd_rank=SVDRANK)

                
                core,factor,block_sizes = bidtucker_query(old_u, old_s, old_vt,TRANK, org_shape,
                        svd_rank=SVDRANK, iter_thresh=ITERTH, iteration=MAXITER, _orgten=org_tensor)
                end = time.time()
                total_time = end-start

                new_tensor = multi_mode_dot_F(core, factor)
                error = np.linalg.norm(org_tensor - new_tensor) ** 2 / np.linalg.norm(org_tensor) ** 2
                print(f'BIDTuckerucker{cut} Err/Time:\t{error}\t{total_time}')
                error_bidtucker.append(error)
                time_bidtucker.append(end-start)

                
    np.savetxt('out/bidtucker_stream_out.txt', np.column_stack((data_block, error_tucker,time_tucker, error_d_tucker,time_d_tucker, error_bidtucker, time_bidtucker)), delimiter='\t', fmt='%10.5f')


    plt.figure()
    plt.plot(data_block, error_tucker, 'r*--', label='Tucker')
    plt.plot(data_block, error_d_tucker, 'g^-.', label='DTucker')
    plt.plot(data_block, error_bidtucker,  'bo-', label='BIDTucker')
    plt.xlabel("Tensor Size (Time)")
    plt.ylabel("Normalized Reconstruction Error")
    plt.legend()
    #plt.grid(True)
    #plt.ylim([0, 0.05])     # Y axis range: [ymin, ymax]
    plt.savefig("out/Tensor_error_comp.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    plt.figure()
    plt.plot(data_block, time_tucker, 'r*--', label='Tucker')
    plt.plot(data_block, time_d_tucker, 'g^-.', label='DTucker')
    plt.plot(data_block, time_bidtucker,  'bo-', label='BIDTucker')
   
    plt.xlabel("Tensor Size (Time)")
    plt.ylabel("Query Time")
    plt.legend()
    #plt.grid(True)
    #plt.ylim([0, 1.0])     # Y axis range: [ymin, ymax]
    plt.savefig("out/Tensor_time_comp.pdf", format="pdf", bbox_inches="tight")
    plt.clf()



    #plt.show()
    return error_d_tucker,time_d_tucker,error_bidtucker,time_bidtucker




if __name__ == "__main__":
    main()


