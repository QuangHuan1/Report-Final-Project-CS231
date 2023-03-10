"""Poisson image editing.
"""
import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve

def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    return mat_A

def poisson_edit(source, target, mask, offset):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))
    maskcp = mask.copy()
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    
    mat_A = laplacian_matrix(y_range, x_range)
    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    mat_A = mat_A.tocsc()
    # source[mask == 0] = (0,200,0)
    # cv2.imshow("s", source)
    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        target[y_min:y_max, x_min:x_max, channel] = x

    kernel = np.ones((3, 3), np.uint8)
    for i in range(7):
        maskcp = cv2.erode(maskcp, kernel, iterations=1)
        a = 0.9
        target[maskcp != 0] = target[maskcp != 0]*(a) + source[maskcp != 0]  * (1-a)
        a -= 0.1
        cv2.imshow("mask", maskcp)
        cv2.imshow("target", target)
        cv2.waitKey(0)
    
    cv2.imshow("target", target)
    cv2.waitKey(0)
    return target
