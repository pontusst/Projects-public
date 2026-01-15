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

from cv2 import SIFT_create, cvtColor, COLOR_RGB2GRAY, FlannBasedMatcher, drawMatchesKnn

np.random.seed(1)


def init(dataset):
    # Choose params
    params = dict(
            dataset=dataset,
            inlier_threshold_E=0.6, 
            inlier_threshold_T=0.6,
            lowe_ratio=0.75
        )

    file_names = []

    # initialize K and pixel threshold (given in assigement)
    K, img_names, init_pair, pixel_threshold = get_dataset_info(params['dataset'])
    K_inv = np.linalg.inv(K)

    # Pixel threshold for RANSAC (given in assignement)
    eps_K_norm = pixel_threshold / K[0,0]
    print(f'eps_K_norm = {eps_K_norm}')

    # Initialize P1 3x4 [I | 0]
    R1 = np.diag([1, 1, 1])
    T1 = np.array([0,0,0]).T
    P_init = np.hstack((R1, T1.reshape(3,1)))
    print(P_init)

    # Initialize file path
    cwd = Path.cwd()
    print(cwd)

    # extracting features from images in SIFT
    im, keypoints, descriptors = SIFT_feature_extraction(cwd, img_names)


    print(f'number of keypoint objects = {len(keypoints)}')
    print(f'number of descriptor objects = {len(descriptors)}')
    print(f'number of images = {len(im)}')
    print(f'Init pair = ', init_pair)
    return K, K_inv, eps_K_norm, im, keypoints, descriptors, cwd, R1, T1, P_init, params, init_pair

def run_sfm(dataset):
    '''
    Pipeline: 
    - load images
    - Extract matching keypoints with SIFT
    - Estimate E via epipolar constraints and RANSAC 
    - Get P from E
    - triangulate points using DLT
    - Visualize 
    '''


    # initialize
    K, K_inv, eps_K_norm, im, keypoints, descriptors, cwd, R1, T1, P_init, params, init_pair = init(dataset)



    # extract keypoints for init pair
    first, second = init_pair

    print(init_pair)

    init_x1, init_x2, good_matches_init, descriptors_init = extract_good_matches(first, second, descriptors, keypoints, params)
    draw_keypoints(im[first], im[second], keypoints[first], keypoints[second])

    # extract and save the descriptors for the points that gets triangulated
    descriptors_init_pair_2 = descriptors[second]

    print(f'len descriptors init: {len(descriptors_init)}\n')
    print(f'len descriptors init_pair[0]: {len(descriptors[init_pair[0]])}\n')
    print(f'descriptors_init_pair_2: {len(descriptors[second])}\n')
    print(f'shape good_matches_init: {np.shape(good_matches_init)}')
    


    # restructuring and homogenizing SIFT points
    ones = np.ones((1, init_x1.shape[1])) 
    x1_h = np.vstack((init_x1, ones)) 
    x2_h = np.vstack((init_x2, ones))

    # normalizing SIFT points
    x1_hn = K_inv @ x1_h 
    x2_hn = K_inv @ x2_h



    # get inlier pair from init pair
    E, inlier_init_x1, inlier_init_x2, inlier_position = extract_inliers(x1_hn, x2_hn)



    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.scatter(inlier_init_x1[0], inlier_init_x1[1], s=1) 
    ax2.scatter(inlier_init_x2[0], inlier_init_x2[1], s=1) 
    ax3.scatter(x1_hn[0], x1_hn[1], s=1) 
    ax4.scatter(x2_hn[0], x2_hn[1], s=1) 
    plt.show()



    # extract descriptors of points in image "init_pair[1]" that gets triangulated
    filtered_descriptors = filter_init_descriptors(good_matches_init, inlier_position, descriptors_init_pair_2)

    # make filtered descriptors into a np array to match descriptor format
    filtered_descriptors = np.array(filtered_descriptors)

    print(len(init_x1[1]))
    print(len(inlier_init_x1[1]))
    assert len(filtered_descriptors) == len(inlier_init_x1[1])
    assert len(filtered_descriptors) == len(inlier_init_x2[1])



    # put in an E and inlier points and get out P2 and X for those points
    correct_P2_init, correct_X_init = triangulate(E, inlier_init_x1, inlier_init_x2, P_init, verbose=True)



    print(correct_P2_init)
    print(correct_X_init)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection="3d")

    ax1.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
    ax1.view_init(elev=-24, azim=-96, roll=179)
    plot_camera(P_init, 1, ax=ax1)
    plot_camera(correct_P2_init, 1, ax=ax1)

    plt.show()



    print(len(im))

    # initialize list for storing camera matrices for relative rotations
    P_relative_rotation = []
    x1_final = []
    x2_final = []
    X_inliers = []
    #P_relative_rotation.append(P_init)

    for idxx, img in enumerate(im):
        if idxx == len(im)-1:
            continue
        else:
            x1, x2, _, _ = extract_good_matches(idxx, idxx+1)

            # restructuring and homogenizing SIFT points
            ones = np.ones((1, x1.shape[1])) 
            rel_x1_h = np.vstack((x1, ones)) 
            rel_x2_h = np.vstack((x2, ones))

            # normalizing SIFT points
            rel_x1_hn = K_inv @ rel_x1_h 
            rel_x2_hn = K_inv @ rel_x2_h

            # get E and inlier pair from im idx, idx+1 pair
            E, rel_inlier_x1, rel_inlier_x2, _ = extract_inliers(rel_x1_hn, rel_x2_hn, eps_K_norm, params, verbose=False)

            correct_P2, X_inlier = triangulate(E, rel_inlier_x1, rel_inlier_x2, P_init, verbose=True)
            x1_final.append(rel_inlier_x1)
            x2_final.append(rel_inlier_x2)
            X_inliers.append(X_inlier)
            print(idxx)
            P_relative_rotation.append(correct_P2)




    print(P_relative_rotation)
    #print(correct_P2_init)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection="3d")

    ax3.scatter(correct_X_init[0], correct_X_init[1], correct_X_init[2], s=1)
    for p in P_relative_rotation:
        plot_camera(p, 1, ax=ax3)
    ax3.view_init(elev=-24, azim=-96, roll=179)

    plt.show()




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

        print(f'should be 1 and is = {np.linalg.det(R2)}')          # should be very close to +1
        print(f'should be very small and is = {np.linalg.norm(R2.T @ R2 - np.eye(3))}')  # should be very small

    print(f'R_absolute_rotation = {R_absolute_rotation}')

    for RR in R_absolute_rotation:
        print(f'should be 1 and is = {np.linalg.det(RR)}') 
        print(f'should be very small and is = {np.linalg.norm(RR.T @ RR - np.eye(3))}')  




    # rotate X 
    if init_pair[0] != 0:
        world_coordinate_X = R_absolute_rotation[init_pair[0]].T @ correct_X_init[:3, :] 
    else: 
        world_coordinate_X = correct_X_init[:3, :] 

    assert len(world_coordinate_X[1]) == len(inlier_init_x1[1])




    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection="3d")

    ax4.scatter(world_coordinate_X[0], world_coordinate_X[1], world_coordinate_X[2], s=1)
    ax4.view_init(elev=-24, azim=-96, roll=179)
    t = T1
    for r in R_absolute_rotation:
        p = np.hstack((r, T1.reshape(3,1)))
        plot_camera(p, 3, ax=ax4)
        
    plt.show()




    # Initialize list for storing T
    T_absolute = []
    inliers_T = []
    #T_absolute.append(T1)

    assert world_coordinate_X.shape[1] == filtered_descriptors.shape[0]

    print(filtered_descriptors.shape[0])

    for idx, desc in enumerate(descriptors):
        if idx == len(im):
            continue
        else:
            current_camera_rotation = R_absolute_rotation[idx]
            T, xsh = calculate_one_T(filtered_descriptors, desc, current_camera_rotation, keypoints[idx], world_coordinate_X, K_inv, eps_K_norm, verbose=True)
            inliers_T.append(xsh) 
            T_absolute.append(T)

    number_of_Ts = len(T_absolute)
    print(f'number of T:s = {number_of_Ts}')
    for nr in range(number_of_Ts):
        print(f'T = {T_absolute[nr].reshape(3,1)}')




    # initialize list for storing final P
    final_P = []
    #final_P.append(P1)

    for idx, (R_abs, T_abs) in enumerate(zip(R_absolute_rotation,T_absolute)):
        P = np.hstack((R_abs, T_abs.reshape(3,1)))
        final_P.append(P)

    print(f'final_P[1] = \n {final_P[1]}')
    print(f'final_R[1] = \n {R_absolute_rotation[1]}')
    print(f'final_T[1] = \n {T_absolute[1].reshape(3,1)}')
    print(f'final_P = \n {final_P}')
