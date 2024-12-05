import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
import csv
import torch
import pandas as pd
data = np.load('filepath')
# PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
num_vertice = data.shape[1]
#dtw matrix
data_mean = np.mean([data[:, :, 0][24 * 12 * i: 24 * 12 * (i + 1)]
                     for i in range(data.shape[0] // (24 * 12))],axis=0)
data_mean = data_mean.squeeze().T
dtw_distance = np.zeros((num_vertice, num_vertice))
for i in tqdm(range(num_vertice)):
    for j in range(i, num_vertice):
        dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
for i in range(num_vertice):
    for j in range(i):
        dtw_distance[i][j] = dtw_distance[j][i]
np.save('your filepath', dtw_distance)

#spatial matrix
dist_matrix = np.zeros((num_vertice, num_vertice)) + np.float('inf')
file = csv.reader('your filepath')
for line in file:
    break
for line in file:
    start = int(line[0])
    end = int(line[1])
    dist_matrix[start][end] = float(line[2])
    dist_matrix[end][start] = float(line[2])
np.save('your filepath', dist_matrix)

dist_matrix = np.load('your filepath')
# normalization
std = np.std(dist_matrix[dist_matrix != float('inf')])
mean = np.mean(dist_matrix[dist_matrix != float('inf')])
dist_matrix = (dist_matrix - mean) / std
sigma = args.sigma2
sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
sp_matrix[sp_matrix < args.thres2] = 0
np.save('your filepath', sp_matrix)

# corr matrix
data = np.load('filepath')['data'][:,:,0]
data_reshaped = data.reshape(-1,num_vertice)
std = np.std(data_reshaped)
mean = np.mean(data_reshaped)
dist_matrix = (data_reshaped - mean) / std
corr_matrix = np.corrcoef(dist_matrix, rowvar=False)
np.save('your filepath', corr_matrix)

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))