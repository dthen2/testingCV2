import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

l = [[0, 10, 20],\
     [30, 0, 40],\
     [0, 0, 0]]

# 下記二つは等価
matrix = csr_matrix(l)
matrix = csr_matrix(([10,20,30,40], ([0,0,1,1],[1,2,0,2]))).toarray()

mat2 = csr_matrix(([1,2,3,4],([0,0,1,1],[0,1,0,1])))
mat3 = csr_matrix(([1,2],([0,1],[0,1]))).toarray()
mat4 = csr_matrix(([mat3,mat3],([0,1],[0,1]))).toarray() # これはエラーが出る