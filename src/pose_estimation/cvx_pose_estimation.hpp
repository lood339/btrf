//  Created by jimmy on 2016-03-31.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef __cvx_pose_estimation__
#define __cvx_pose_estimation__

// preemptive RANSAC camera pose estimation
#include <stdio.h>
#include <vector>
#include "cvx_image_310.hpp"

using std::vector;

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
    // param: RANSAC parameter
    // camera_pose: output, camera coordinates to world coordinates
    static bool preemptiveRANSAC3DOneToMany(const vector<cv::Point3d> & camera_pts,
                                            const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                            const PreemptiveRANSAC3DParameter & param,
                                            cv::Mat & camera_pose);
    
    // pose distance
    // angle_distance: degree
    // euclidean_disance: meter
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

#endif /* __cvx_pose_estimation__ */
