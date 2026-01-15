# For your convenience:
# Paste the required functions from previous assignments here.
import numpy as np
import scipy as sp



def enforce_essential(E_approx):
    '''
    E_approx - Approximate Essential matrix (3x3)
    '''
    U, S, Vt = np.linalg.svd(E_approx)
    det = np.linalg.det(np.matmul(U,Vt))
    print(f'Determinant of Essential matrix (U @ Vt) is = {det}')
    if det < 0:
        #Vt[-1,:] = -Vt[-1,:]
        E = U @ np.diag(S) @ -Vt
    else:
        E = U @ np.diag(S) @ Vt
    return E

def enforce_fundamental(F_approx):
    '''
    F_approx - Approximate Fundamental matrix (3x3)
    '''
    det = np.linalg.det(F_approx)
    #print(f'Determinant of Fundamental matrix is {det}')
    try:
        assert det == 0, "Determinant of Fundamental matrix is not zero!"
    except AssertionError as e:
        print("AssertionError:", e)

def estimate_F_DLT(x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    '''
    nr_of_points = np.size(x1s[0])
    M = np.zeros((nr_of_points, 9))
    for i in range(len(x1s[0])):
        x1, y1, z1 = x2s[:,i]

        x2, y2, z2 = x1s[:,i]

        M[i,:] = [x1*x2,  x1*y2,  x1*z2,
                y1*x2,  y1*y2,  y1*z2,
                z1*x2,  z1*y2,  z1*z2
                ]

    U, S, Vt = np.linalg.svd(M)

    F = Vt[-1,:].reshape(3,3)
    smallest_eig_val = S[-1]
    Mv = np.linalg.norm(M @ Vt[-1,:])
    #print(f'In estimate F/E with DLT, |Mv| = {Mv}')
    #print(f'In estimate F/E with DLT, smallest singular value = {smallest_eig_val}')

    return F

def convert_E_to_F(E,K1,K2):
    '''
    A function that gives you a fundamental matrix from an essential matrix and the two calibration matrices
    E - Essential matrix (3x3)
    K1 - Calibration matrix for the first image (3x3)
    K2 - Calibration matrix for the second image (3x3)
    '''
    F_from_e = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)
    return F_from_e

def compute_epipolar_errors(F, x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    F - Fundamental matrix (3x3)
    returns: array contaning distance from point i to line corresponding to point i 
    '''
    D = np.zeros(len(x1s[1])) # zero vector for holding distances
    for points in range(len(x1s[1])):
        l2 = F @ x1s[:,points] # calculate l2, homogeneous
        x0, y0, _ = x2s[:,points] # extract corresponding point 
        temp = np.abs(l2[0]*x0 + l2[1]*y0 + l2[2])
        temp2 = np.sqrt(l2[0]**2 + l2[1]**2)
        d = temp/temp2
        D[points] = d

    return D

def extract_P_from_E(E):
     '''
    A function that extract the four P2 solutions given above
    E - Essential matrix (3x3)
    P - Array containing all four P2 solutions (4x3x4) (i.e. P[i,:,:] is the ith solution) 
    '''
     E = enforce_essential(E)
     U, S, Vt = np.linalg.svd(E)
     W = np.array([[0, -1, 0],
          [1, 0, 0],
          [0, 0, 1]])
     u3 = U[:,2]
     Pa = np.hstack((U @ W @ Vt, u3.reshape(3,1))) 
     Pb = np.hstack((U @ W @ Vt, -u3.reshape(3,1))) 
     Pc = np.hstack((U @ W.T @ Vt, u3.reshape(3,1))) 
     Pd = np.hstack((U @ W.T @ Vt, -u3.reshape(3,1))) 
     P = np.array([Pa, Pb, Pc, Pd])
     return P

def triangulate_3D_point_DLT(x1, x2, P1, P2):
    '''
    Takes as input two homogeneus points and two camera matrices 
    and outputs one 3d scene point in homogeneous coordinates
    '''
    u1, v1, _ = x1
    u2, v2, _ = x2

    #M = np.zeros((4, 4))
    #M[0] = u1 * P1[2] - P1[0]
    #M[1] = v1 * P1[2] - P1[1]
    #M[2] = u2 * P2[2] - P2[0]
    #M[3] = v2 * P2[2] - P2[1]
    M = np.vstack((u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0],  v2 * P2[2] - P2[1]))

    _, _, Vt = np.linalg.svd(M)
    X = Vt[-1]
    X = X / X[-1]
    return X.reshape(4,)