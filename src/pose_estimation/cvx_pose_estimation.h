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
    // SCRF_testing_result: predictions from random forests
    // pts_camera_view: correspondent camera view 3D coordinate
    // camera_pose: camera to world, 4 * 4 matrix
    // outlier_threshold: unit of pixel
  
    // 3D - 2D
    static bool estimateCameraPose(const cv::Mat & camera_matrix,
                                   const cv::Mat & dist_coeff,
                                   const vector<cv::Point2d> & im_pts,
                                   const vector<cv::Point3d> & wld_pts,
                                   cv::Mat & camera_pose,
                                   const double outlier_threshold = 8.0);
    
    // wld_pts: estimated points, has outliers
    static bool estimateCameraPose(const vector<cv::Point3d> & camera_pts,
                                   const vector<cv::Point3d> & wld_pts,
                                   cv::Mat & camera_pose);
    
    // wld_pts: estimated points, has outliers
    static bool preemptiveRANSAC(const vector<cv::Point2d> & img_pts,
                                 const vector<cv::Point3d> & wld_pts,
                                 const cv::Mat & camera_matrix,
                                 const cv::Mat & dist_coeff,
                                 const PreemptiveRANSACParameter & param,
                                 cv::Mat & camera_pose);
    
    //corresonding world coordinate locations, estimated points, had outliers, multiple choices
    static bool preemptiveRANSAC2DOneToMany(const vector<cv::Point2d> & img_pts,
                                            const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                            const cv::Mat & camera_matrix,
                                            const cv::Mat & dist_coeff,
                                            const PreemptiveRANSACParameter & param,
                                            cv::Mat & camera_pose,
                                            cv::Mat & rvec,
                                            cv::Mat & tvec);
    
    // wld_pts: estimated points, had outliers
    static bool preemptiveRANSAC3D(const vector<cv::Point3d> & camera_pts,
                                   const vector<cv::Point3d> & wld_pts,
                                   const PreemptiveRANSAC3DParameter & param,
                                   cv::Mat & camera_pose);
    
    // camera_pts: camera coordinate locations
    // candidate_wld_pts: corresonding world coordinate locations, estimated points, had outliers, multiple choices
    static bool preemptiveRANSAC3DOneToMany(const vector<cv::Point3d> & camera_pts,
                                            const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                            const PreemptiveRANSAC3DParameter & param,
                                            cv::Mat & camera_pose);
    
    // camera_pts: camera coordinate locations
    // wld_pts: corresonding world coordinate locations, estimated points, had outliers, multiple choices
    // using all inliers
    static bool preemptiveRANSAC3DAllInliers(const vector<cv::Point3d> & camera_pts,
                                             const vector<vector<cv::Point3d>> & candidate_wld_pts,
                                             const PreemptiveRANSAC3DParameter & param,
                                             cv::Mat & camera_pose);
    
    // wld_pts: estimated points, had outliers
    // inliers: inliers for the finale camera pose
    static bool preemptiveRANSAC3D(const vector<cv::Point3d> & camera_pts,
                                   const vector<cv::Point3d> & wld_pts,
                                   const PreemptiveRANSAC3DParameter & param,
                                   cv::Mat & camera_pose,
                                   vector<bool> & inliers);
    
    
    // angle_distance: degree
    static void poseDistance(const cv::Mat & src_pose,
                      const cv::Mat & dst_pose,
                      double & angle_distance,
                      double & euclidean_disance);
    
    //
    // database_camera_poses: camera pose 4x4 64FC1
    // query_pose: query camera pose
    // angular_threshold: angular threshold, default value 10 degrees
    // return: smallest camera distance when the camera angular is smaller than the angular threshold, in meter
    static double minCameraDistanceUnderAngularThreshold(const vector<cv::Mat> & database_camera_poses,
                                                         const cv::Mat & query_pose,
                                                         const double angular_threshold);
    
    //
    // database_camera_poses: camera pose 4x4 64FC1
    // query_pose: query camera pose
    // translation_threshold: angular threshold, default value 10 degrees
    // return: smallest camera angular distance when the camera distance is smaller than the threshold, in degree
    static double minCameraAngleUnderTranslationalThreshold(const vector<cv::Mat> & database_camera_poses,
                                                         const cv::Mat & query_pose,
                                                         const double translation_threshold);
    
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
