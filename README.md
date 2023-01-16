This is the code for moving targets detection by a moving camera.
1. The image features are selected by good_feature_to_track 
2. The features are then tracked by LK optical flow through consecutive 3 frames
3. The 2nd and 3rd images are transformed on the 1st image by registering the features
4. The new 2nd image are substracted by the new 3rd image, the residual image is then postprocessed by thresholding, and morphological operations.
