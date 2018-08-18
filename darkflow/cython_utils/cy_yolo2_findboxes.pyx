import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from ..utils.box import BoundBox
from nms cimport NMS
import math

#expit
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float expit_c(float x):
    cdef float y= 1/(1+exp(-x))
    return y

#MAX
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float max_c(float a, float b):
    if(a>b):
        return a
    return b

"""
#SOFTMAX!
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _softmax_c(float* x, int classes):
    cdef:
        float sum = 0
        np.intp_t k
        float arr_max = 0
    for k in range(classes):
        arr_max = max(arr_max,x[k])

    for k in range(classes):
        x[k] = exp(x[k]-arr_max)
        sum += x[k]

    for k in range(classes):
        x[k] = x[k]/sum
"""



#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_constructor(meta,np.ndarray[float,ndim=3] net_out_in):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1,index,index2
        float  threshold = meta['thresh']
        float tempc,arr_max=0,sum=0
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()
        float maxz = meta['maxz']

    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']
    ANC = 5

    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:7]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :] DISTANCE       = net_out[:, :, :, 7]
        #float[:, :, :] ALPHA         = net_out[:, :, :, 8]
        float[:, :, :] VECX         = net_out[:, :, :, 8]
        float[:, :, :] VECY         = net_out[:, :, :, 9]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)

    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = exp(Bbox_pred[row, col, box_loop, 2]) * anchors[ANC * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[ANC * box_loop + 1] / H
                DISTANCE[row, col, box_loop]     = exp(DISTANCE[row, col, box_loop]) * maxz * anchors[ANC * box_loop + 2] / W
                #ALPHA [row, col, box_loop]       = exp(ALPHA[row, col, box_loop]) * anchors[ANC * box_loop + 3] / W
                VECX [row, col, box_loop]       = exp(VECX[row, col, box_loop]) * anchors[ANC * box_loop + 3] / W
                VECY [row, col, box_loop]       = exp(VECY[row, col, box_loop]) * anchors[ANC * box_loop + 4] / W
                #SOFTMAX BLOCK, no more pointer juggling
                for class_loop in range(C):
                    arr_max=max_c(arr_max,Classes[row,col,box_loop,class_loop])

                for class_loop in range(C):
                    Classes[row,col,box_loop,class_loop]=exp(Classes[row,col,box_loop,class_loop]-arr_max)
                    sum+=Classes[row,col,box_loop,class_loop]

                for class_loop in range(C):
                    tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc


    #NMS
    #return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5), np.ascontiguousarray(DISTANCE).reshape(H*W*B), np.ascontiguousarray(ALPHA).reshape(H*W*B))
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5), np.ascontiguousarray(DISTANCE).reshape(H*W*B))
    #return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5), np.ascontiguousarray(DISTANCE).reshape(H*W*B) , np.ascontiguousarray(VECX).reshape(H*W*B) , np.ascontiguousarray(VECY).reshape(H*W*B))
