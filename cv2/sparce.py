import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    s = tf.SparseTensor(\
                indices = [[0,0],[0,1],[0,2],\
                           [4,3],[5,0],[5,1]],\
                values = [1.2,1.2,1.2,1.2,1.2,1.2],\
                dense_shape = [6, 6] )

    p = tf.SparseTensor(\
                indices = [[0,0],[0,1],[0,2],\
                           [4,3],[5,0],[5,1]],\
                values = [1.3,1.3,1.3,1.3,1.3,1.3],\
                dense_shape = [6, 6] )
    sd = tf.sparse.to_dense(s)
    pd = tf.sparse.to_dense(p)
    print(sd)
    print("hoge")
    print(pd)
    print("hogehoge")
    x = tf.einsum('ij,jk->ik', s, pd)
    print(x)