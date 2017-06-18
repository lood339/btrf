//  Created by jimmy on 2016-06-06.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef ms7ScenesUtil_cpp
#define ms7ScenesUtil_cpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <string>
#include <vector>
#include <Eigen/Dense>

using std::string;
using std::vector;

class Ms7ScenesUtil
{
public:
    // read camera pose file
    static cv::Mat read_pose_7_scenes(const char *file_name);
    
    //
    static bool read_pose_7_scenes(const char *file_name, Eigen::Affine3d& affine);
    
    // invalid depth is 0.0
    static cv::Mat camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose);
        
    // camera_depth_img 16 bit
    // return CV_64_FC3 for x, y, z, unit in meter
    static cv::Mat camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose);
    
    // mask: CV_8UC1 0 --> invalid sample
    static cv::Mat camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img,
                                                    const cv::Mat & camera_to_world_pose,
                                                    cv::Mat & mask);
    // mask: CV_8UC1 0 --> invalid sample
    static cv::Mat camera_depth_to_camera_coordinate(const cv::Mat & camera_depth_img,                                                    
                                                     cv::Mat & mask);
    // mask: CV_8UC1 0 --> invalid sample
    // return CV_64_FC3 for x, y, z, unit in meter
    static void camera_depth_to_camera_and_world_coordinate(const cv::Mat & camera_depth,
                                                            const cv::Mat & camera_to_world_pose,
                                                            cv::Mat & camera_coord,
                                                            cv::Mat & world_coord,
                                                            cv::Mat & mask);
    
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
   
    static cv::Mat camera_matrix();
    
    
    static inline int invalid_camera_depth(){return 65535;}
    
    static bool load_prediction_result(const char *file_name, string & rgb_img_file, string & depth_img_file, string & camera_pose_file,
                                       vector<cv::Point2d> & img_pts,
                                       vector<cv::Point3d> & wld_pts_pred,
                                       vector<cv::Point3d> & wld_pts_gt);
    
    // load prediction result from decision trees with color information
    static bool load_prediction_result_with_color(const char *file_name,
                                                  string & rgb_img_file,
                                                  string & depth_img_file,
                                                  string & camera_pose_file,
                                                  vector<cv::Point2d> & img_pts,
                                                  vector<cv::Point3d> & wld_pts_pred,
                                                  vector<cv::Point3d> & wld_pts_gt,
                                                  vector<cv::Vec3d> & color_pred,
                                                  vector<cv::Vec3d> & color_sample);
    
    // load prediction result from all decision trees with color information
    static bool load_prediction_result_with_color(const char *file_name,
                                                  string & rgb_img_file,
                                                  string & depth_img_file,
                                                  string & camera_pose_file,
                                                  vector<cv::Point2d> & img_pts,
                                                  vector<cv::Point3d> & wld_pts_gt,
                                                  vector<vector<cv::Point3d> > & candidate_wld_pts_pred,
                                                  vector<cv::Vec3d> & color_sample,
                                                  vector<vector<cv::Vec3d> > & candidate_color_pred);
    
    // load prediction result form all decision trees with feature distance information
    static bool load_prediction_result_with_distance(const char *file_name,
                                                     string & rgb_img_file,
                                                     string & depth_img_file,
                                                     string & camera_pose_file,
                                                     vector<cv::Point2d> & img_pts,
                                                     vector<cv::Point3d> & wld_pts_gt,
                                                     vector<vector<cv::Point3d> > & candidate_wld_pts_pred,
                                                     vector<vector<double> > & candidate_feature_dists);
    
    // load prediction result from all decision trees with feature distance and uncertainty
    static bool load_prediction_result_with_uncertainty(const char *file_name,
                                                        string & rgb_img_file,
                                                        string & depth_img_file,
                                                        string & camera_pose_file,
                                                        vector<Eigen::Vector2d> & img_pts,
                                                        vector<Eigen::Vector3d> & wld_pts_gt,
                                                        vector<vector<Eigen::Vector3d> > & candidate_wld_pts_pred,
                                                        vector<vector<Eigen::Matrix3d> > & candidate_wld_pts_pred_covariance,
                                                        vector<vector<double> > & candidate_feature_dists);
                                                  
    
    // load camera estimation result
    static bool load_estimated_camera_pose(const char *file_name,
                                           string & rgb_img_file,
                                           string & depth_img_file,
                                           string & camera_pose_file,
                                           cv::Mat & estimated_pose);
    
    // read file names in a file
    static vector<string> read_file_names(const char *file_name);   
    
};

#endif /* ms_7scenes_util_cpp */
