# btrf
Backtracking regression forest

This is a modified implementation of paper

@inproceedings{meng2017backtracking,
	title={Backtracking Regression Forests for Accurate Camera Relocalization},
	author={Meng, Lili and Chen, Jianhui and Tung, Frederick and Little J., James and Valentin, Julien and Silva, Clarence},
	booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2017)},  
	year={2011}  
}

Dependences:
1. OpenCV 3.1 or later. 
2. Eigen 3.2.6 or later.
3. flann 1.8.4 or later.

The code is tested on Xcode 6.4 on a Mac 10.10.5 system. But the code has minimum dependence on compile and system, so it should work well on linux and windows as well.

File structure:
src/btrf_*.hpp and src/btrf_*.cpp: the main algorithm for backtracking regression forst.

src/cmd: three files for training, testing of world coordinates prediction from RGB-D images, and camera pose estitation

src/dt_common: common function for decition tree, for example, objective functions

src/opencv_util: wrap of opencv function for the project

src/pose_estimation: camera pose estimation using Kabsch and preemptive RANSAC

src/Walsh_Hadamard: Walsh hardamard transformation. Code modifed from : http://www.faculty.idc.ac.il/toky/Software/wh/code.htm

src/yael_io.*: code for binary matrix read/write. Code modifed from: https://gforge.inria.fr/projects/yael

parameters/4scenes_param.txt: dataset parameter
parameters/forest_param.txt: forest parameter example. 
parameters/apt1_kitchen/  : training/testing file sequence examples 

Todo: cmake file


