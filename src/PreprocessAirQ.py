#!/usr/bin/env python
# coding: utf-8
#
# **전체 process**
# 
# 
# 1. raw data 취합
# 
# 
# 2. min-Max normalize
# 
# 
# 3. 2번 결과를 tensor 로 변환
#     3-1. 각 지역 코드를 key 로 갖는 dictionary 정의
#     3-2. key 값 별로 layer 나눈 뒤 stack
# 
# 
# 4. d-tucker 기반 proposed method로 텐서 분해
#  ( https://github.com/Ajou-DILab/Dynamic-Tensor-Decomposition/blob/master/src/dao_cp/methods/ddt.py )
# 
# 

import itertools
import os
import numpy as np
import pandas as pd
import tensorly as tl

from sklearn.preprocessing import MinMaxScaler

# global variables
#DEBUG=True
DEBUG=False

def main():
#    out_path = '../out/small-'
    out_path = '../out/'
    data_path = '../data/AirQ-Data/'
    years=['2018', '2019', '2020', '2021', '2022']
#    years=['small']
    # read data and get numpy tensor data 
    data = readAirQ(data_path, years, out_path)


def readAirQ(data_path, years, out_path=''):
    '''
    read Air Quality excel data with following cols:
    Index(['측정소코드', '측정일시', 'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25'] 
    '''
    # initialize DF
    data = pd.DataFrame()
    # for each year
    for year in years:
        # list files
        files= os.listdir(data_path+year+'/')
        files_xlsx = [f for f in files if f[-4:] == 'xlsx']
        for f in files_xlsx:
            # clear and read 
            df = pd.DataFrame()
            df = pd.read_excel(data_path+year+'/'+f, na_values=[''])
            # append to data     
            data = pd.concat([data, df])
            print('Finished Reading\t', f) 
            print('\tshape\t', df.shape)
            print('\taxes\t', df.axes)
            
    # store pickled 
    data.to_pickle(out_path+'AirQData_raw.zip', compression='zip')

    # ---------------normalize data----------- 
    data = normMinMaxAirQ(data)
    
    # store pickled 
    data.to_pickle(out_path+'AirQData_normMinMax.zip', compression='zip')

    # find index dic and diminsions of each order
    locinds  = sorted(data['측정소코드'].drop_duplicates())
    locdic = {locinds[i]:i for i in range(len(locinds))}
    timeinds = sorted(data['측정일시'].drop_duplicates())
    timedic = {timeinds[i]:i for i in range(len(timeinds))}
    typeinds = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
    typedic = {typeinds[i]:i for i in range(len(typeinds)) }
    # order rearanged to - (time, loc, type) 
    dims     = ( np.int64(len(timeinds)), np.int64(len(locinds)), np.int64(len(typeinds)) )
    print(f'dim(time, loc, type):\t \'{dims}\'.')

    # initialize tensor with nan  (time x loc x type)
    airqT = np.empty(dims) * np.nan
    # read data row by row and insert to tensor 
    for i, row in data.iterrows():
        # find index 
        for j in range(6):
            # time is in row 1 and loc in row 0
            indices = (np.int64(np.asarray(timedic[row[1]])) , np.int64(np.asarray(locdic[row[0]])), j)
            if not np.isnan(row[j+2]): airqT[indices] = np.double(row[j+2])
    if(DEBUG): 
        print("printing (normalized) airqT:\n")
        print(airqT)
    
    # find avg value for each location/type for filling missing values 
    #avg = np.empty(np.int64(np.asarray(dims[1])), np.int64(np.asarray(dims[2]))) 
    avg = np.zeros(shape=(dims[1], dims[2])) 
    print('avg\t', avg.shape)
    nalocs = []
    for keyl in locdic.keys():
        # check if loc has more than 10% nan if so remove 
        sli = airqT[:, locdic[keyl], :]
        if np.count_nonzero(np.isnan(sli))/(len(sli)*len(sli[0])) < 0.1 :  
            for keyt in typedic.keys():
                avg[(locdic[keyl],typedic[keyt])] = np.nanmean(airqT[:,
                    locdic[keyl], typedic[keyt]])
        else: # store loc the loc/type that contains all zeros to remove later
            nalocs.append(keyl)
            if(DEBUG): print('slice loc ', locdic[keyl],' contains more than 10% nans')

    # save the locations 
    ofname = out_path+'AirQ-removed_loc.lst'
    np.savetxt(ofname, nalocs, fmt='%s')
    print(f'Locations with missing values over all times saved as \'{ofname}\'!')

    # remove locs with nan fibers
    rinds = [locdic.get(key) for key in nalocs]
    airqT = np.delete(airqT, rinds, 1)
    avg = np.delete(avg, rinds, 0)
    # update loc index 
    locinds  = np.delete(locinds, rinds)
    locdic = {locinds[i]:i for i in range(len(locinds))}
    
    # store missing data index 
    ofname = out_path+'AirQ-nan-ind.npy'
    naninds = np.argwhere(np.isnan(airqT))
    np.save(f'{ofname}', arr=naninds)
    print(f'nan location in org tensor saved as \'{ofname}\'!')
    if(DEBUG): print ('nan index:', naninds)

    if(DEBUG): print('\tPrinting avg values for loc-type: \n', avg)
    # insert missing values with average values 
    for ind in naninds:
        airqT[tuple(ind)] = avg[ind[1], ind[2]]
        if(DEBUG): print('Set airqT ', tuple(ind), 'to ', avg[ind[1], ind[2]] )

    """ Save the original tensor with missing data interpolated with loc/type avg. (time, loc, type) """
    ofname = out_path+'AirQ-org.npy'
    np.save(f'{ofname}', airqT)
    print(f'Original tensor of shape {airqT.shape} saved as \'{ofname}\'!')

    ofname = out_path+'AirQ-label.txt'
    np.savetxt( f'{ofname}', [timeinds, locinds, typeinds], fmt='%s' ) 

    if(DEBUG): 
        print("printing final airqT:\n")
        print(airqT)

    return (airqT, timeinds, locinds, typeinds)   


def normMinMaxAirQ(data):
    """
    Normalize each column of a Pandas dataframe using min-max normalization.
    """
    # Create a DF to store the normalized data
    result = data.copy()
    # loop over each column
    # for col in data.columns:
    for col in ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']:
        # calculate min max of the column
        minv = data[col].min()
        maxv = data[col].max()
        # do minmax normalization 
        result[col] = (data[col] - minv)/(maxv-minv)
        
        if(DEBUG): print('Processing col ',col,': min ',minv,'max ',maxv)
    
    return result 


def normQ1Q9(data):
    """
    NOT CODED YET: Normalize each column of a Pandas dataframe using quantile normalization.
    """
    # Create a DF to store the normalized data
    result = data.copy()
    # loop over each column
    # calculate quantile table for q1 and q3 of each column
    q1 = data.quantile(0.1) 
    q9 = data.quantile(0.9)
    if(DEBUG): print('quatile:\n', quant)
        
    # for col in data.columns:
    for col in ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']:
        # do quantile normalization 
        result[col] = (data[col])/(q9[col]-q1[co])

    return result 


if __name__ == "__main__":
  main()


