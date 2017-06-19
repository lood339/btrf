//  Created by jimmy on 2016-06-06.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef __ms7scenes_util__
#define __ms7scenes_util__

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

// read/load ground truth or prediction data
class Ms7ScenesUtil
{
public:
    // read camera pose from a .txt file
    static cv::Mat read_pose_7_scenes(const char *file_name);
    
    // load prediction result form all decision trees with feature distance information
    // file_name: name of a .txt file
    // rgb_img_file, depth_img_file: RGB and depth image file name
    // camera_pose_file: ground truth camera pose file name
    // img_pts: sampled location in image space
    // gt_wld_pts: world coordinate ground truth location
    // pred_wld_pts: predicted world coordinate from random forest
    // feature_dists: feature distance of predicted world coordinate, auxiliary data
    static bool load_prediction_result_with_distance(const char *file_name,
                                                     string & rgb_img_file,
                                                     string & depth_img_file,
                                                     string & camera_pose_file,
                                                     vector<cv::Point2d> & img_pts,
                                                     vector<cv::Point3d> & gt_wld_pts,
                                                     vector<vector<cv::Point3d> > & pred_wld_pts,
                                                     vector<vector<double> > & feature_dists);
    
};

#endif /* __ms7scenes_util__ */
