//
//  cvx_pose_estimation.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-31.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvx_pose_estimation_cpp
#define cvx_pose_estimation_cpp

#include <stdio.h>
#include "cvx_image_310.hpp"
#include <vector>

using std::vector;

struct PreemptiveRANSACParameter
{
    double reproj_threshold; // re-projection error threshold, unit pixel
    
public:
    PreemptiveRANSACParameter()
    {
        reproj_threshold = 5.0;
    }
};

struct PreemptiveRANSAC3DParameter
{
    double dis_threshold_;    // distance threshod, unit meter
    int sample_number_;
    int refine_camera_num_;   // refine camera using all inliers
public:
    PreemptiveRANSAC3DParameter()
    {
        dis_threshold_ = 0.1;
        refine_camera_num_ = -1;
        sample_number_ = 1024;
    }
    
};

class CvxPoseEstimation
{
public:    
    
    
    
    
       
    // camera_pts: camera coordinate locations
    // candidate_wld_pts: corresonding world coordinate locations, estimated points, had outliers, multiple choices
    static bool preemptiveRANSAC3DOneToMany(const vector<cv::Point3d> & camera_pts,
                                            const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                            const PreemptiveRANSAC3DParameter & param,
                                            cv::Mat & camera_pose);
    
    
    
    
    
    // angle_distance: degree
    static void poseDistance(const cv::Mat & src_pose,
                      const cv::Mat & dst_pose,
                      double & angle_distance,
                      double & euclidean_disance);
    
   
    
private:
    
    
    // 3x3 rotation matrix to eular angle
    static Mat rotationToEularAngle(const cv::Mat & rot);
    
    // return CV_64FC1 4x1
    static Mat rotationToQuaternion(const cv::Mat & rot);
    
    // return CV_64FC1 3x3
    static Mat quaternionToRotation(const cv::Mat & q);
    
    
    
    static inline float SIGN(float x) {return (x >= 0.0f) ? +1.0f : -1.0f;}
    static inline float NORM(float a, float b, float c, float d) {return sqrt(a * a + b * b + c * c + d * d);}
};

#endif /* cvxPoseEstimation_cpp */
