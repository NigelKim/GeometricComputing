# Geometric Computing
Final Project on Drosophila bifurca Sperm Cell Length Measurement\
Language: Python\
Course: CSE 554 Geometric Computing for Biomedicine\
Collaborator: Dohoon Kim\
Instructor: Tao Ju Ph.D.\
Institution: Washington University In St. Louis\

# How to use the Sperm Cell Length Measurement GUI

1. Run “python -m tkinter” from the command line to check if the package is installed on your system. If it is not installed on your system, download the tcl and tk package.

2. If OpenCV is not installed on your system, download OpenCV by running a command “brew install opencv” from the command line, or manually download the library.

3. Open GUI.py in IDLE and run.

4. When the GUI starts up, it will ask you for a input image to start with. Select your input image from local system.

5. If your target sperm cell is not very visible, use the basic modification tools to adjust brightness and contrast, zoom in or out, and check image dimensions and pixel intensities.

6. If your target sperm cell is visible, use the Crop button to select a tight enclosing region around your sperm cell, and Crop Image.

7. Use the Eraser and Draw buttons to get rid of noises and completely connect the sperm cell. 

8. Now you can start the measurement pipeline by clicking through the buttons labeled 1 through 7. (Make necessary modifications using the Eraser and the Brush in between the pipeline to get a better result by eliminating false information and connect the disconnected regions. Check the top right of the GUI during the pipeline process to monitor the completeness of each pipeline.)

9. When the measurement pipeline buttons are executed in correct order(ascending from 1 to 7), the sperm cell length will be measured and displayed at the top of the screen.

10. Press Start Over button to reset the program and select the input image again to make additional measurements.

