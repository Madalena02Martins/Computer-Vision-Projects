# Lab 6 - Feature detection, matching and geometric transformations

## Outline
* SIFT (Scale-Invariant Feature Transform) for keypoint detection
* Descriptor matching with BFMatcher and FLANN
* Filtering matches and evaluating quality
* Geometric transformations (Affine transform & Homography)
*   Estimating scale, rotation, and translation
*   Applying the transformation to images
*   Evaluating differences between transformed images

## Scripts
`aula_06_ex_01.py` → Reads an image, prints its characteristics, applies an affine transformation (rotation + translation) using a predefined matrix, and displays/saves the transformed image.
`aula_06_ex_02.py` → Allows manual selection of 3 corresponding points in two images, estimates the affine transformation, applies it, computes transformation parameters (scale, rotation), and shows differences between the warped image and the reference.
`aula_06_ex_03.py` → Detects keypoints using SIFT and visualizes them.
`aula_06_ex_04.py` → Matches descriptors between two images using BFMatcher and displays the correspondences.
`aula_06_ex_05.py` → Estimates and applies an affine transformation, computes parameters (scale, rotation), shows the transformed image and differences. Also includes matching with FLANN.
`aula_06_ex_06.py` → Allows manual selection of 4 points in two images, estimates a homography transformation with `cv2.findHomography`, warps the image accordingly, and prints transformation parameters (translation and angle).

## References
[OpenCV Feature Detection (SIFT)](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)  
[OpenCV Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)  
[OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)  
[OpenCV Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)  

