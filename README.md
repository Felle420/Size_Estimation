# Size_Estimation
Python script for estimating your shirt size with the power of AI. 
This script is built to help the user estimate their shirt size without the hassle of whipping out a tape measurer.

# Requirements
You must have and be able to run at least python 3.7 results are not guaranteed with earlier versions.

# How to use
start by pasting the following lines into the VS Code terminal;

pip install cv2

pip install cvzone

pip install numpy

pip install mediapipe

Then start the "storlek.py" script and make sure you have a camera in your device. 
make sure your device is placed in such a way that the camera can see your entire body and that it sees you from straight on. 
Avoid making the camera see you from an angle if you desire the most accurate results.
Proceed by placing yourself in the silouette displayed in the cameraview, you should end up about 1.5m from the camera.
Notice the EU size displayed on the screen, this is your estimated shirt size.

# Reasons for errors
This estimation model is not flexible to different FOVs, therefore if your camera has a higher or lower FOV than 1.67m from left to right at 1.5m distance from the screen, your results may not be correct.
The model is not as accurate for plus size people because it doesn't account for waist measurements and can't understand profile view.
