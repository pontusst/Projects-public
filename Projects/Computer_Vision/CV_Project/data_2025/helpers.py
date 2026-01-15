# For your convenience:
# Paste the required functions from previous assignments here.
import cv2
import numpy as np
import scipy as sp
import random
from cv2 import SIFT_create, cvtColor, COLOR_RGB2GRAY, FlannBasedMatcher, drawMatchesKnn
import matplotlib.pyplot as plt
from supplied import plot_camera


def enforce_essential(E_approx, verbose=False):
    '''
    E_approx - Approximate Essential matrix (3x3)
    '''
    U, S, Vt = np.linalg.svd(E_approx)
    S_tilde = np.diag([1, 1, 0])
    det = np.linalg.det(np.matmul(U,Vt))
    if verbose:
        print(f'Determinant of Essential matrix (U @ Vt) is = {det}')
    if det < 0:
        E = U @ S_tilde @ -Vt
    else:
        E = U @ S_tilde @ Vt
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

def estimate_F_DLT(x1s, x2s, verbose=False):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    '''
    nr_of_points = np.size(x1s[0])
    M = np.zeros((nr_of_points, 9))
    for i in range(nr_of_points):
        x1, y1, z1 = x2s[:,i]

        x2, y2, z2 = x1s[:,i]

        M[i,:] = [x1*x2,  x1*y2,  x1*z2,
                y1*x2,  y1*y2,  y1*z2,
                z1*x2,  z1*y2,  z1*z2]
                

    U, S, Vt = np.linalg.svd(M)

    F = Vt[-1,:].reshape(3,3)
    smallest_eig_val = S[-1]
    Mv = np.linalg.norm(M @ Vt[-1,:])
    if verbose:
        print(f'In estimate F/E with DLT, |Mv| = {Mv}')
        print(f'In estimate F/E with DLT, smallest singular value = {smallest_eig_val}')

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
    # zero vector for holding distances
    D = np.zeros(len(x1s[1])) 

    for points in range(len(x1s[1])):

        # calculate l2, homogeneous
        l2 = F @ x1s[:,points] 

        # extract corresponding point
        x0, y0, _ = x2s[:,points]  
        temp = np.abs(l2[0]*x0 + l2[1]*y0 + l2[2])
        temp2 = np.sqrt(l2[0]**2 + l2[1]**2)
        d = temp/temp2
        D[points] = d

    return D

def extract_P_from_E_old(E):
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

def extract_P_from_E(E):
    """
    Extract the four possible P2 matrices from an Essential matrix.
    """
    E = enforce_essential(E)

    U, _, Vt = np.linalg.svd(E)

    # Ensure U and Vt define a proper rotation
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:, 2]

    # Final determinant check (defensive)
    if np.linalg.det(R1) < 0:
        R1 = -R1
        t  = -t

    if np.linalg.det(R2) < 0:
        R2 = -R2
        t  = -t

    Pa = np.hstack((R1,  t.reshape(3,1)))
    Pb = np.hstack((R1, -t.reshape(3,1)))
    Pc = np.hstack((R2,  t.reshape(3,1)))
    Pd = np.hstack((R2, -t.reshape(3,1)))

    return np.array([Pa, Pb, Pc, Pd])


def triangulate_3D_point_DLT(x1, x2, P1, P2):
    '''
    Takes as input two homogeneus points and two camera matrices 
    and outputs one 3d scene point in homogeneous coordinates
    '''
    u1, v1, _ = x1
    u2, v2, _ = x2

    M = np.vstack((u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0],  v2 * P2[2] - P2[1]))
    
    _, _, Vt = np.linalg.svd(M)
    X = Vt[-1]
    X = X / X[-1]
    return X.reshape(4,)

def estimate_E_robust(x1, x2, eps, inlier_threshold, seed=None, verbose=False):
    """
    RANSAC estimate of essential matrix using normalized correspondences x1 and x2 and a normalized threshold.
    Note: Make sure to normalize things before using it in this function!
    -------------------------------------------
    x1: Normalized keypoints in image 1 - 3xN np.array or 2xN np.array, as you desire 
    x2: Normalized keypoints in image 2 - 3xN np.array or 2xN np.array, as you desire 
    eps: Normalized inlier threshold - float

    Returns:
    E: 3x3 essential matrix
    inliers: The inlier points
    errs: The epipolar errors
    iters: How many iterations it took
    """
    done = False
    iters = 0
    best_inliers = 0
    max_its = 200
    
    inlier_goal = inlier_threshold*x1.shape[1]
    
    while not done:
        count_inliers = 0
        random_sample = random.sample(range(len(x1[1])), 8)
        r_samples_x1 = np.vstack([x1[:,i] for i in random_sample]).T
        r_samples_x2 = np.vstack([x2[:,i] for i in random_sample]).T

        E_tild = enforce_essential(estimate_F_DLT(r_samples_x1, r_samples_x2)) # Giving 8 samples to DLT

        e1 = compute_epipolar_errors(E_tild, x1, x2)**2 
        e2 = compute_epipolar_errors(E_tild.T, x2, x1)**2

        assert len(e1) == len(x1[1])

        inliers_bool = (1/2)*(e1+e2) < eps**2
        count_inliers = np.sum(inliers_bool)

        if count_inliers > best_inliers:
            best_E = E_tild
            best_inliers = count_inliers
            best_inlier_bool = inliers_bool.copy()
            best_inliers_x1 = x1[:,inliers_bool]  
            best_inliers_x2 = x2[:,inliers_bool]
        if best_inliers >= inlier_goal or iters > max_its:
            done = True
            errs = (e1,e2)
            inliers_res = (best_inliers_x1, best_inliers_x2)
            E = best_E
        if verbose:
            print('inlier goal', inlier_goal)
            print('current inliers', count_inliers)
            print('current best inliers', best_inliers)
            print(f'iters = {iters}')
        
        iters += 1

    return E, inliers_res, errs, iters, best_inlier_bool


def homography_to_RT(H):
    
    def unitize(a,b):
        denom = 1.0 / (a**2+b**2)**(0.5)
        ra = a * denom
        rb = b * denom
        return ra, rb

    [U,S,Vt] = np.linalg.svd(H)
    s1 = S[0] / S[1]
    s3 = S[2] / S[1]
    a1 = (1 - s3**2)**(0.5)
    b1 = (s1**2 - 1)**(0.5)
    [a,b] = unitize(a1, b1)
    [c,d] = unitize(1+s1*s3, a1*b1 )
    [e,f] = unitize(-b/s1, -a/s3 )
    v1 = Vt.T[:,0]
    v3 = Vt.T[:,2]
    n1 = b * v1 - a * v3
    n2 = b * v1 + a * v3
    R1 = U @ np.array([[c,0,d], [0,1,0], [-d,0,c]]) @ Vt
    R2 = U @ np.array([[c,0,-d], [0,1,0], [d,0,c]]) @ Vt
    t1 = (e * v1 + f * v3).reshape(-1,1)
    t2 = (e * v1 - f * v3).reshape(-1,1)
  
    if n1[2] < 0:
        t1 = -t1
        n1 = -n1

    if n2[2] < 0:
        t2 = -t2
        n2 = -n2

    t1 = R1 @ t1
    t2 = R2 @ t2

    RT = np.zeros((2,3,4))
    RT[0] = np.hstack([R1, t1])
    RT[1] = np.hstack([R2, t2])
    return RT




def estimate_T_robust(xs, Xs, R, eps, inlier_threshold, verbose=False):
    """
    RANSAC estimate of Translation using normalized correspondences x1 and x2 and a normalized threshold.
    Note: Make sure to normalize things before using it in this function!
    -------------------------------------------
    xs: Normalized 2D keypoints in image i - 3xN np.array or 2xN np.array, as you desire 
    Xs: 3D points corresponding to 2D points - 3xN np.array or 2xN np.array, as you desire 
    eps: Normalized inlier threshold - float

    The xs and Xs must correspond to eachother at every index for this function to work

    Returns:
    t: 3x1 translation vector
    inliers: The inlier points
    errs: The epipolar errors
    iters: How many iterations it took
    """
    max_its = 40000
    done = False
    iters = 0
    best_count_inliers = 0
    inlier_goal = inlier_threshold*xs.shape[1]
    print(f'inlier goal = {inlier_goal}')
    while not done:
        count_inliers = 0

        # generating a random sample
        random_sample = random.sample(range(xs.shape[1]), 2)
        r_samples_xs = np.vstack([xs[:,i] for i in random_sample]).T
        r_samples_Xs = np.vstack([Xs[:,i] for i in random_sample]).T

        # Giving 2 samples to DLT
        T_tild = estimate_T_DLT(r_samples_xs, r_samples_Xs, R)

        # Generate P
        P_tilde = np.hstack((R, T_tild.reshape(3,1)))

        # Project 3D points with camera P_tilde
        x_tilde = P_tilde @ Xs
        x_tilde = x_tilde/x_tilde[-1]

        # computing errors
        e = np.linalg.norm(xs - x_tilde, axis=0)

        # checking inliers amongst errors - bool list
        inliers = e < eps

        count_inliers = np.sum(inliers)
        
        if count_inliers > best_count_inliers:
            # save best T
            best_T = np.copy(T_tild)

            # save best number of inliers 
            best_count_inliers = np.copy(count_inliers)

            # save inliers
            best_inliers = inliers
            
        if best_count_inliers >= inlier_goal or iters > max_its:
            done = True
            errs = e
            inliers = best_inliers
            T = best_T
        if verbose:
            print(f"In iteration {iters} there are {count_inliers} inliers")
            print(f"current best inliers = {best_count_inliers}")
        iters += 1

    return T, inliers, errs, iters

def estimate_camera_DLT(x, Xmodel, verbose=False):
    '''
    Docstring for estimate_camera_DLT
    
    Function taking in two points and estimating camera

    :param x: Projected point
    :param Xmodel: Scene point
    '''
    n = Xmodel.shape[1]
    M = np.zeros((2*n, 12))

    u = x[0][:]
    v = x[1][:]
    X = Xmodel[0][:]
    Y = Xmodel[1][:]
    Z = Xmodel[2][:]
    Xmodel_hom = np.column_stack([X,Y,Z, np.ones(n)])
    M[0::2, 0:4] = Xmodel_hom
    M[0::2, 8:12] = -u[:,None]*Xmodel_hom

    M[1::2, 4:8] = Xmodel_hom
    M[1::2, 8:12] = -v[:,None]*Xmodel_hom

    U, S, Vt = np.linalg.svd(M)

    P = Vt[-1,:].reshape(3,4)
    smallest_eig_val =S[-1]
    Mv = np.linalg.norm(M @ Vt[-1,:])
    if verbose:
        print(f'|Mv| = {Mv}')
        print(f'smallest singular value = {smallest_eig_val}')

    return P

def estimate_T_DLT(x, X, R):
    """
    x: 2x2 array of normalized image points [[u1,u2],[v1,v2]]
    X: 3x2 array of corresponding 3D points
    R: 3x3 known rotation
    """

    A = []
    b = []

    for i in range(2):
        u, v, _ = x[:, i]
        RX = R @ X[:3, i]

        A.append([1, 0, -u])
        A.append([0, 1, -v])

        b.append(u * RX[2] - RX[0])
        b.append(v * RX[2] - RX[1])

    A = np.array(A)
    b = np.array(b)

    T, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return T

def SIFT_feature_extraction(PROJECT_ROOT, img_names):
    im_list = []
    keypoint_list = []
    descriptor_list = []

    rgb2gray = lambda img: cvtColor(img, COLOR_RGB2GRAY)
    sift = SIFT_create(contrastThreshold=0.02, edgeThreshold=10, nOctaveLayers=3)

    # load images in greyscale
    for j, name in enumerate(img_names):
        print((f'./{name}'))
       
        im = (plt.imread(f'{PROJECT_ROOT}/{name}') * 255).astype('uint8')

        keypoints, descriptors = sift.detectAndCompute(rgb2gray(im),None)
    
        im_list.append(im)
        keypoint_list.append(keypoints)
        descriptor_list.append(descriptors)
    
    return im_list, keypoint_list, descriptor_list

def find_good_matches(desc1, desc2, keypoints1, keypoints2, ratio_threshold):
    '''
    Docstring for find_good_matches
    
    returns good matches made by SIFT with lowe ratio 0.75

    x1, x2 in pixel coordinates
    '''
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches12 = bf.knnMatch(desc1, desc2, k=2)
    matches21 = bf.knnMatch(desc2, desc1, k=2)

    good12 = [m for m,n in matches12 if m.distance < ratio_threshold*n.distance]
    good21 = [m for m,n in matches21 if m.distance < ratio_threshold*n.distance]

    # Mutual matches
    good_matches = []
    for m in good12:
        if any(m.trainIdx == mm.queryIdx and m.queryIdx == mm.trainIdx for mm in good21):
            good_matches.append(m)
    

    x1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).T
    x2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).T
    return x1, x2, good_matches, desc2

def save_in_mat(variables, file_name, variable_names):
    from scipy.io import savemat

    savemat(
        file_name,
        dict(zip(variable_names, variables)
        )
    )

def draw_keypoints(im1, im2, keypoints1, keypoints2):
    im1_kp = cv2.drawKeypoints(im1, keypoints1, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im2_kp = cv2.drawKeypoints(im2, keypoints2, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.imshow(im1_kp)
    ax1.set_title("Image 1 keypoints")

    ax2.imshow(im2_kp)
    ax2.set_title("Image 2 keypoints")

    plt.show()

def extract_good_matches(im1_idx, im2_idx, desc, kp, params, verbose=False):
    '''
    General match extractor
    input: image index
    output: x1 and x2 correspondances filtered with lowe ratio and the matches between im1 and im2

    '''
    # extracting correspondances from SIFT 
    x1, x2, good_matches, d2 = find_good_matches(desc[im1_idx], desc[im2_idx], kp[im1_idx], kp[im2_idx], params['lowe_ratio'])
    if verbose:
        print(f'shape of SIFT points from image {im1_idx} and {im2_idx}\n')
        print(f'shape x1 after matching = {x1.shape}')
        print(f'shape x2 after matching = {x2.shape}')

    return x1, x2, good_matches, d2

def extract_inliers(x1, x2, eps_K_norm, params, verbose=False):
    '''
    input: normalized SIFT points

    output: inliers, postions of inliers and estimated E 
    '''

    # Estimate E using RANSAC
    E, inliers, errs, iters, inlier_position = estimate_E_robust(x1, x2, eps_K_norm, params['inlier_threshold_E'], seed=None)
    
    

    # extracting inlier points from RANSAC
    inlier_x1, inlier_x2 = inliers
    if verbose:
        print(f'Shape of inlier x1 = {inlier_x1.shape}')
        print(f'Shape of inlier x2 = {inlier_x2.shape}')
        print(f'Shape of inlier position = {inlier_position.shape}')

    assert len(inlier_x1[1]) == sum(inlier_position)
    assert len(inlier_position) == len(x1[1])

    return E, inlier_x1, inlier_x2, inlier_position

def filter_init_descriptors(good_matches, inlier_position, descriptors_init):
    '''
    input: 
    good matches: matches between image "init_pair[1]" and descriptor of other images
    ie. for a image set of 9 images where 9 is "init_pair[1]", [1, 9], [2, 9], [3, 9], ... and so on.

    inlier_position: position of the inliers based of epipolar constrains between im1 and 9

    output:
    descriptors of im 9 that has passed lowe test and RANSAC
    '''
    # filtering out descriptors in good_matches where the inliers were
    # here we take all the good match indexes between image 1 and 9 and filter the descriptors of 
    # image 9 -> this gives us descriptors of im 9 that pass the lowe ratio test.
    # if inlier_position[0] is true then  that corresponds to that good_matches[0] is an inlier,
    # therefore we add that descriptor to the descriptors that describe all descriptors that pass
    # the lowe ratio test and the RANSAC test. 
    filtered_init_descriptors = [descriptors_init[m.trainIdx] for m, n in zip(good_matches, inlier_position) if n == 1]
    return filtered_init_descriptors

def triangulate(E, inlier_init_x1, inlier_init_x2, P1, verbose=False, plot=False):
    assert np.allclose(inlier_init_x1[2], 1)
    assert np.allclose(inlier_init_x2[2], 1)

    # Compute Cameras from E / relative rotation P1 P2
    P2_arr = extract_P_from_E(E)
        
    # stating how many inlier points are available
    N = inlier_init_x1.shape[1]

    # initilizing list to store 3D scene points in. 4xNx4.
    X_tot_arr = []

    # initilizing cherialty check list
    count = np.zeros(4)

    # triangulating X from RANSAC inliers
    for idxx, P2 in enumerate(P2_arr):
        X_arr = np.zeros((4, N))
        for p in range(N):
            X = triangulate_3D_point_DLT(inlier_init_x1[:,p], inlier_init_x2[:,p], P1, P2)
            

            X_arr[:,p] = X
            if X[2] < 0 or X[2] > 5000:
                  X_arr[:,p] = np.nan      
            
            # Cherialty check

            P1Xj = P1 @ X
            if P1Xj[2] <= 0:
                continue

            RXt = P2[:, :3] @ X[:3] + P2[:, 3]
            if RXt[2] <= 0:
                continue

            # if positive 
            count[idxx] += 1
            
        X_tot_arr.append(X_arr)    
    
    # cherialty check
    max_position = np.argmax(count)

    # saving correct P2 
    correct_P2 = P2_arr[max_position]
    correct_X = X_tot_arr[max_position]

    if verbose: 
        print('X3', count)
        print('max position', max_position)    

        for idxx, P2 in enumerate(P2_arr):
            Rrrr = P2[:,:-1]
            d = np.linalg.det(Rrrr)
            print('determinant of P2', d)

    if plot:
        R1 = np.diag([1, 1, 1])
        T1 = np.array([0,0,0]).T
        P_init = np.hstack((R1, T1.reshape(3,1)))
        fig22 = plt.figure()
        fig22.suptitle('position 1')
        ax22 = fig22.add_subplot(projection="3d")

        #ax1.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
        ax22.scatter(X_tot_arr[0][0], X_tot_arr[0][1], X_tot_arr[0][1], s=1)
        ax22.view_init(elev=-24, azim=-96, roll=179)
        plot_camera(P_init, 1, ax=ax22)
        plot_camera(P2_arr[0], 1, ax=ax22)

        fig222 = plt.figure()
        fig222.suptitle('position 2')
        ax222 = fig222.add_subplot(projection="3d")


        #ax1.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
        ax222.scatter(X_tot_arr[1][0], X_tot_arr[1][1], X_tot_arr[1][1], s=1)
        ax222.view_init(elev=-24, azim=-96, roll=179)
        plot_camera(P_init, 1, ax=ax222)
        plot_camera(P2_arr[1], 1, ax=ax222)

        fig2222 = plt.figure()
        fig2222.suptitle('position 3')
        ax2222 = fig2222.add_subplot(projection="3d")


        #ax1.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
        ax2222.scatter(X_tot_arr[2][0], X_tot_arr[2][1], X_tot_arr[2][1], s=1)
        ax2222.view_init(elev=-24, azim=-96, roll=179)
        plot_camera(P_init, 1, ax=ax2222)
        plot_camera(P2_arr[2], 1, ax=ax2222)

        fig22222 = plt.figure()
        fig22222.suptitle('position 4')
        ax22222 = fig22222.add_subplot(projection="3d")


        #ax1.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
        ax22222.scatter(X_tot_arr[3][0], X_tot_arr[3][1], X_tot_arr[3][1], s=1)
        ax22222.view_init(elev=-24, azim=-96, roll=179)
        plot_camera(P_init, 1, ax=ax22222)
        plot_camera(P2_arr[3], 1, ax=ax22222)

        plt.show()
    return correct_P2, correct_X
    # result from this is:
    # - correct X and correct P for estimated E

    
def calculate_one_T(filtered_descriptors, desc, R, keypoints, w_c_X, K_inv, eps_K_norm, verbose=False):

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # check matches between every descriptor and 3D descriptor 9 
    matches = bf.knnMatch(filtered_descriptors, desc, k=2)

    xs = np.zeros((2,len(matches)))
    Xs = np.zeros((3,len(matches)))

    for k, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            X_j = w_c_X[:, m.queryIdx]
            x_i = keypoints[m.trainIdx].pt
            xs[:, k] = x_i
            Xs[:, k] = X_j
            
    # homogenize and normalize xs
    o = np.ones(xs.shape[1])
    xsh = np.vstack((xs, o))
    xshn = K_inv @ xsh

    # homogenize Xs
    o = np.ones(Xs.shape[1])
    Xsh = np.vstack((Xs, o))
    
    if verbose:
        print(f'xshn = {xshn}')
        print(f'Xsh = {Xsh}')

    # cherialty check
    assert min(Xs[2]) >= 0

    # estimate T 
    T, inliers, errs, iters = estimate_T_robust(xshn, Xsh, R, 3*eps_K_norm, 0.8)
    return T, xshn