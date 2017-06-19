//
//  cvxCalib3d.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-12.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxCalib3d_cpp
#define cvxCalib3d_cpp

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <Eigen/Dense>

using std::vector;

class CvxCalib3D
{
public:    
    static void rigidTransform(const vector<cv::Point3d> & src, const cv::Mat & affine, vector<cv::Point3d> & dst);
    
        
    static void KabschTransform(const vector<cv::Point3d> & src, const vector<cv::Point3d> & dst, cv::Mat & affine);
    
    static Eigen::Affine3d KabschTransform(const vector<Eigen::Vector3d> & src, const vector<Eigen::Vector3d> & dst);
    
    
    // camera transform from both point-point pairs, and line segment end point pairs
    static void KabschTransform(const vector<cv::Point3d> & src, const vector<cv::Point3d> & dst,
                                const vector<cv::Point3d> & line_end_src, const vector<cv::Point3d> & line_end_dst,
                                cv::Mat & affine);
    
       
    
        
    
    // camera_depth_img: CV_64FC1
    // camera_to_world_pose: 4x4 CV_64FC1
    // calibration_matrix: 3x3 CV_64FC1
    // depth_factor: e.g. 1000.0 for MS 7 scenes
    // camera_xyz: output camera coordinate location, CV_64FC3
    // mask: output CV_8UC1 0 -- > invalid, 1 --> valid
    // return: CV_64FC3 , x y z in world coordinate
    static cv::Mat cameraDepthToWorldCoordinate(const cv::Mat & camera_depth_img,
                                                const cv::Mat & camera_to_world_pose,
                                                const cv::Mat & calibration_matrix,
                                                const double depth_factor,
                                                const double min_depth,
                                                const double max_depth,
                                                cv::Mat & camera_coordinate,
                                                cv::Mat & mask);
    
};

#endif /* cvxCalib3d_cpp */
