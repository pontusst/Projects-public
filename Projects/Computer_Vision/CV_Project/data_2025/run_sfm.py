import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Note: These functions are provided for your convenience, use them where needed
from supplied import pflat, plot_camera, rital, camera_center_and_axis
from helpers import estimate_F_DLT, enforce_essential, convert_E_to_F, enforce_fundamental, estimate_E_robust
from helpers import triangulate_3D_point_DLT, extract_P_from_E, compute_epipolar_errors, estimate_T_robust, estimate_T_DLT
from helpers import draw_keypoints, save_in_mat, find_good_matches, SIFT_feature_extraction, extract_good_matches
from helpers import extract_inliers, filter_init_descriptors, triangulate, calculate_one_T
from project_helpers import get_dataset_info
from pathlib import Path
from tqdm import tqdm

from cv2 import SIFT_create, cvtColor, COLOR_RGB2GRAY, FlannBasedMatcher, drawMatchesKnn

np.random.seed(1)

from pathlib import Path
# Absolute path to THIS file
THIS_FILE = Path(__file__).resolve()

# Project root 
PROJECT_ROOT = THIS_FILE.parents[0]



def init(dataset, verbose=False):
    # Choose params
    params = dict(
            dataset=dataset,
            inlier_threshold_E=0.6, 
            inlier_threshold_T=0.6,
            lowe_ratio=0.75
        )

    # initialize K and pixel threshold (given in assigement)
    K, img_names, init_pair, pixel_threshold = get_dataset_info(params['dataset'])
    K_inv = np.linalg.inv(K)

    # Pixel threshold for RANSAC (given in assignement)
    eps_K_norm = pixel_threshold / K[0,0]

    # Initialize P1 3x4 [I | 0]
    R1 = np.diag([1, 1, 1])
    T1 = np.array([0,0,0]).T
    P_init = np.hstack((R1, T1.reshape(3,1)))

    # extracting features from images in SIFT
    im, keypoints, descriptors = SIFT_feature_extraction(PROJECT_ROOT, img_names)

    if verbose:
        print(f'number of keypoint objects = {len(keypoints)}')
        print(f'number of descriptor objects = {len(descriptors)}')
        print(f'number of images = {len(im)}')
        print(f'Init pair = ', init_pair)

    return K, K_inv, eps_K_norm, im, keypoints, descriptors, R1, T1, P_init, params, init_pair

def run_sfm(dataset: int):
    '''
    Docstring for run_sfm
    
    running pipeline of dataset 1-7
    '''

#-------------------------------------------------
    # initialize
    print('INITIALIZING \n')
    K, K_inv, eps_K_norm, im, keypoints, descriptors, R1, T1, P_init, params, init_pair = init(dataset)


#-------------------------------------------------
    # extract keypoints for init pair
    print('EXTRACTING KEYPOINTS FROM INIT PAIR \n')
    first, second = init_pair

    init_x1, init_x2, good_matches_init, _ = extract_good_matches(first, second, descriptors, keypoints, params)
    draw_keypoints(im[first], im[second], keypoints[first], keypoints[second])

    # extract and save the descriptors for the points that gets triangulated
    descriptors_init_pair_2 = descriptors[second]

    

#-------------------------------------------------
    # restructuring and homogenizing SIFT points
    print('HOMOGENIZING SIFT POINTS \n')
    ones = np.ones((1, init_x1.shape[1])) 
    x1_h = np.vstack((init_x1, ones)) 
    x2_h = np.vstack((init_x2, ones))

    # normalizing SIFT points
    x1_hn = K_inv @ x1_h 
    x2_hn = K_inv @ x2_h


#-------------------------------------------------
    # get inlier pair from init pair
    print(f'GETTING INLIERS AND E FROM INIT PAIR {init_pair}\n')
    E, inlier_init_x1, inlier_init_x2, inlier_position = extract_inliers(x1_hn, x2_hn, eps_K_norm, params, verbose=False)


#-------------------------------------------------
    print(' VIZULIZING SIFT POINTS AND INLIER POINTS FOR INIT PAIR  \n')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.scatter(inlier_init_x1[0], inlier_init_x1[1], s=1) 
    ax2.scatter(inlier_init_x2[0], inlier_init_x2[1], s=1) 
    ax3.scatter(x1_hn[0], x1_hn[1], s=1) 
    ax4.scatter(x2_hn[0], x2_hn[1], s=1) 
    plt.show()


#-------------------------------------------------
    print('FILTERING 3D DESCRIPTORS \n')
    # extract descriptors of points in image "init_pair[1]" that gets triangulated
    filtered_descriptors = filter_init_descriptors(good_matches_init, inlier_position, descriptors_init_pair_2)

    # make filtered descriptors into a np array to match descriptor format
    filtered_descriptors = np.array(filtered_descriptors)

    assert len(filtered_descriptors) == len(inlier_init_x1[1])
    assert len(filtered_descriptors) == len(inlier_init_x2[1])


#-------------------------------------------------
    print(' TRIANGULATING POINTS FOR INIT PAIR \n')
    # put in an E and inlier points and get out P2 and X for those points
    correct_P2_init, correct_X_init = triangulate(E, inlier_init_x1, inlier_init_x2, P_init, verbose=True)


#-------------------------------------------------
    print(' VIZULIZING ESTIMATED CAMERAS AND 3D POINTS FOR INIT PAIR  \n')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection="3d")

    ax1.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
    ax1.view_init(elev=-24, azim=-96, roll=179)
    plot_camera(P_init, 1, ax=ax1)
    plot_camera(correct_P2_init, 1, ax=ax1)

    plt.show()


#-------------------------------------------------
    print(' SETTING UP RELATIVE ROTATIONS  \n')

    # initialize list for storing camera matrices for relative rotations
    P_relative_rotation = []
    x1_final = []
    x2_final = []
    X_inliers = []

    for idxx in tqdm(range(len(im)), desc='Computing relative cameras'):
        if idxx == len(im)-1:
            continue
        else:
            x1, x2, _, _ = extract_good_matches(idxx, idxx+1, descriptors, keypoints, params)

            # restructuring and homogenizing SIFT points
            ones = np.ones((1, x1.shape[1])) 
            rel_x1_h = np.vstack((x1, ones)) 
            rel_x2_h = np.vstack((x2, ones))

            # normalizing SIFT points
            rel_x1_hn = K_inv @ rel_x1_h 
            rel_x2_hn = K_inv @ rel_x2_h

            # get E and inlier pair from im idx, idx+1 pair
            E, rel_inlier_x1, rel_inlier_x2, _ = extract_inliers(rel_x1_hn, rel_x2_hn, eps_K_norm, params, verbose=False)

            correct_P2, X_inlier = triangulate(E, rel_inlier_x1, rel_inlier_x2, P_init, verbose=False)
            x1_final.append(rel_inlier_x1)
            x2_final.append(rel_inlier_x2)
            X_inliers.append(X_inlier)
            P_relative_rotation.append(correct_P2)



#-------------------------------------------------
    print(' VIZULIZING CAMERAS ESTIMATED FROM E AND 3D POINTS FROM INIT PAIR  \n')
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection="3d")

    ax3.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
    for p in P_relative_rotation:
        plot_camera(p, 1, ax=ax3)
    ax3.view_init(elev=-24, azim=-96, roll=179)

    plt.show()



#-------------------------------------------------
    print(' SETTING UP ABSOLUTE ROTATIONS  \n')
    # initialize list for storing roation matrices for absolute rotations
    R_absolute_rotation = []
    R_absolute_rotation.append(R1)
    # calculate absolute rotations using relative rotations
    for iter, relative_P in enumerate(P_relative_rotation):
        if iter == len(P_relative_rotation):
            continue
        Ri = R_absolute_rotation[-1]
        curr_P = P_relative_rotation[iter]
        R2 = curr_P[:,:-1]
        U, _, VT = np.linalg.svd(R2, full_matrices=False)
        R2 = U @ VT
        Rj = R2 @ Ri
        R_absolute_rotation.append(Rj)


#-------------------------------------------------
    print(' ROTATING X TO WORLD COORDINATES \n')
    # rotate X 
    if init_pair[0] != 0:
        world_coordinate_X = R_absolute_rotation[init_pair[0]].T @ correct_X_init[:3, :] 
    else: 
        world_coordinate_X = correct_X_init[:3, :] 

    assert len(world_coordinate_X[1]) == len(inlier_init_x1[1])



#-------------------------------------------------
    print(' VIZULIZING ROTATIONS AND 3D POINTS  \n')
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection="3d")

    ax4.scatter(world_coordinate_X[0], world_coordinate_X[1], world_coordinate_X[2], s=1)
    ax4.view_init(elev=-24, azim=-96, roll=179)

    for r in R_absolute_rotation:
        p = np.hstack((r, T1.reshape(3,1)))
        plot_camera(p, 1, ax=ax4)
        
    plt.show()



#-------------------------------------------------
    print(' COMPUTING T \n')
    # Initialize list for storing T
    T_absolute = []
    inliers_T = []

    assert world_coordinate_X.shape[1] == filtered_descriptors.shape[0]

    print(filtered_descriptors.shape[0])

    for idx, desc in enumerate(descriptors):
        if idx == len(im):
            continue
        else:
            current_camera_rotation = R_absolute_rotation[idx]
            T, xsh = calculate_one_T(filtered_descriptors, desc, current_camera_rotation, keypoints[idx], world_coordinate_X, K_inv, eps_K_norm)
            inliers_T.append(xsh) 
            T_absolute.append(T)



#-------------------------------------------------
    print(' PUTTING TOGETHER CAMERAS  \n')
    # initialize list for storing final P
    final_P = []

    for idx, (R_abs, T_abs) in enumerate(zip(R_absolute_rotation,T_absolute)):
        P = np.hstack((R_abs, T_abs.reshape(3,1)))
        final_P.append(P)



#-------------------------------------------------
    print(' VIZULIZING 3D POINTS FROM INIT PAIR + ALL ESTIMATED CAMERAS \n')
    print(final_P)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection="3d")

    R1_world_coordinate_X = world_coordinate_X

    ax3.scatter(R1_world_coordinate_X[0], R1_world_coordinate_X[1], R1_world_coordinate_X[2], s=1)

    ax3.view_init(elev=-24, azim=-96, roll=179)
    for i in final_P:
        plot_camera(i, 1, ax=ax3)

    plt.show()



#-------------------------------------------------
    print(' VIZULIZING ALL ESTIMATED CAMERAS AND 3D POINTS  \n')
    modified_P = []
    Xes_tot = []
    for idx, (x1, x2) in enumerate(zip(x1_final, x2_final)):
        N = len(x1[1])
        Xes = np.zeros((4, N))
        cam1 = final_P[idx]
        cam2 = final_P[idx+1]
        for p in range(N):
            X = triangulate_3D_point_DLT(x1[:, p], x2[:, p], cam1, cam2)
            Xes[:,p] = X 
        Xes_tot.append(Xes)

    fig15 = plt.figure()
    ax15 = fig15.add_subplot(projection="3d")
    for k, points in enumerate(final_P):
        cam = final_P[k]
        cam_R = cam[:, :-1]
        cam_T = cam[:, -1]
        cam_R = R_absolute_rotation[init_pair[0]].T @ cam_R
        mod_P = np.hstack((cam_R, cam_T.reshape(3,1)))
        modified_P.append(mod_P)
        plot_camera(mod_P, 1, ax=ax15)

    for k, points in enumerate(Xes_tot):
        rot_points = R_absolute_rotation[init_pair[0]] @ points[:3,:]
        
        ax15.scatter(rot_points[0], rot_points[1], rot_points[2], s=0.1, color='b')
    plt.show()

