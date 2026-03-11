# Project aim

This project was aimed at creating a structure from motion pipeline in order to reconstruct 3D structure from 2D point correspondances

## Pipeline 
### Images
Multiple images such as this one were selected  
***
![Example image](data_2025/data/4/DSC_0480.JPG) 
***
and this one 
***
![Example image](data_2025/data/4/DSC_0490.JPG) 
***
was selected.  

### Keypoint extraction
Two images suitable as inital pair were run through a SIFT algorithm to extract keypoints.  
WIth the keypoint one can estimate a essential matrix and with that one can estimate 3D points and realtive camera positions. The result can be viewed in the image below. 
 As might be visable in the figure the left most camera is set to the identity matrix and the rightmost is the relative rotation from that camera.  
***
![Example image](data_2025/water_fountain4.png) 
***
### Estimating cameras  

To get a better estimation of the points more correspondances could be used. 
With the initial point corresponadances a fundamental matrix could be extracted and from that the realtive rotations. The rotations were taken as realtive to an baseline rotation that was defined as the identity matrix. 
Extracting the realtive rotations and setting their translation to 0 gave the following result:  
***
![Example image](data_2025/water_fountain3.png)  
***
After this, one rotation was set to be the identity matrix and the other ones were chained together to create the next image:
***
![Example image](data_2025/water_fountain2.png) 
***
We can see that the rotations follow a smooth pattern which is also how the camera was moving when taking the pictures.   
The translations between the cameras were estimated through a RANSAC algorithm with the now known rotations matrises. 
***
![Example image](data_2025/water_fountain1.png) 
***
### Final result
All points that were estimated were then plotted together with the estimated camera poses to create the final reconstruction. 
***
![Example image](data_2025/water_fountain.png) 





