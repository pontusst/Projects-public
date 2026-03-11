# Project aim

This project was aimed at creating a structure from motion pipeline in order to reconstruct 3D structure from 2D point correspondances

## Pipeline 
### Images
Multiple images such as this one were selected.  
Below are som example images 
*** 

 <p align="center">
  <img src="data_2025/data/4/DSC_0480.JPG" alt="Example image" width="400">
</p> 

***


***
<p align="center">
  <img src="data_2025/data/4/DSC_0490.JPG" alt="Example image" width="400">
</p> 

***

### Keypoint extraction
Two images suitable as inital pair were run through a SIFT algorithm to extract keypoints.  
With the keypoint and the camera calibraition matrix an essential matrix can be calculated and with that 3D points were extracted with the RANSAC algorithm. The result can be viewed in the image below. 
 As might be visable in the figure the left most camera is set to the identity matrix and the rightmost is the relative rotation from that camera.  
***

<p align="center">
  <img src="data_2025/water_fountain4.png" alt="Example image" width="400">
</p> 

***

### Estimating camera rotations
Since the restulting camera positions only are defined up to scale when extracting cameras from the esseential matrix only the camera rotations could be extracted and used in the first step.  

With the initial point corresponadances relative rotation from a set camera could be extracted. The rotations were taken as realtive to an baseline camera with a rotation that was defined as the identity matrix and a translation defined as a zero vector. 
Extracting the relative rotations gave the following result:  

***
<p align="center">
  <img src="data_2025/water_fountain3.png" alt="Example image" width="400">
</p>  

***
After this, one cameras rotation matrix was kept as the identity matrix and the other ones were chained together to create the next image:

***
<p align="center">
  <img src="data_2025/water_fountain2.png" alt="Example image" width="400">
</p> 

***
The rotations follow a smooth pattern which is also how the camera was moving when the pictures were taken.  
### Estimation camera translations
The translations between the cameras were estimated through a RANSAC algorithm with the now known rotations matrises. 
***

<p align="center">
  <img src="data_2025/water_fountain1.png" alt="Example image" width="400">
</p>

***
### Final result
All points that were estimated were then plotted together with the estimated camera poses to create the final reconstruction of the cameras and the scene.
***

<p align="center">
  <img src="data_2025/water_fountain.png" alt="Example image" width="400">
</p>

***

### Improvements
This concludes the basic Sfm pipeline but more improvements can be made to increase accuracy of the pipeline.  
* Homography estimation can be used to detect planar degeneracies and filter matches before estimating the essential matrix.
* The 5-point algorithm can be used within RANSAC to robustly estimate the essential matrix from minimal correspondences.
* If sparse features from SIFT are insufficient, dense matchers like RoMa can provide more correspondences and improve reconstruction density. 
* Perform bundle adjustment with the Levenberg–Marquardt algorithm. 





