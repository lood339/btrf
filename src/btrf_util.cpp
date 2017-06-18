//
//  bt_rnd_util.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btrf_util.h"
#include "cvx_io.hpp"
#include "ms7scenes_util.hpp"
#include "cvxWalshHadamard.h"
#include "cvx_util.hpp"


void
BTRNDUtil::randomSampleFromRgbdImages(const char * rgb_img_file,
                                     const char * depth_img_file,
                                     const char * camera_pose_file,
                                     const int num_sample,
                                     const int image_index,
                                     const DatasetParameter & dataset_param,
                                     const bool use_depth,
                                     const bool verbose,
                                     vector<SCRFRandomSample> & features,
                                     vector<Eigen::VectorXf> & labels)
{
    
    BTRNDUtil::randomSampleFromRgbdImagesImpl(rgb_img_file, depth_img_file, camera_pose_file,
                                          num_sample, image_index, dataset_param.depth_factor_,
                                          dataset_param.camera_matrix(), dataset_param.min_depth_,
                                          dataset_param.max_depth_, use_depth, verbose, features, labels);
}


void
BTRNDUtil::randomSampleFromRgbdImagesImpl(const char * rgb_img_file,
                                      const char * depth_img_file,
                                      const char * camera_pose_file,
                                      const int num_sample,
                                      const int image_index,
                                      const double depth_factor,
                                      const cv::Mat calibration_matrix,
                                      const double min_depth,
                                      const double max_depth,
                                      const bool use_depth,
                                      const bool verbose,
                                      vector<SCRFRandomSample> & features,
                                      vector<Eigen::VectorXf> & labels)
{
    assert(rgb_img_file);
    assert(depth_img_file);
    assert(camera_pose_file);
    
    assert(features.size() == 0);
    assert(labels.size() == 0);
    
    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    bool is_read = CvxIO::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
    assert(is_read);
    CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
    
    
    cv::Mat pose = Ms7ScenesUtil::read_pose_7_scenes(camera_pose_file);
    
    const int width = rgb_img.cols;
    const int height = rgb_img.rows;
    
    cv::Mat mask;
    cv::Mat camera_coordinate;
    cv::Mat world_coordinate =  Ms7ScenesUtil::cameraDepthToWorldCoordinate(camera_depth_img, pose, calibration_matrix, depth_factor,
                                                                            min_depth, max_depth, camera_coordinate, mask);
    
    for (int i = 0; i<num_sample; i++) {
        int x = rand()%width;
        int y = rand()%height;
        
        // ignore bad depth point
        if (mask.at<unsigned char>(y, x) == 0) {
            continue;
        }
        double depth = 1.0;
        if (use_depth) {
            depth = camera_depth_img.at<double>(y, x)/depth_factor;
        }
        SCRFRandomSample sp;
        sp.p2d_ = Eigen::Vector2f(x, y);
        sp.inv_depth_ = 1.0/depth;
        sp.image_index_ = image_index;
        features.push_back(sp);
        
        cv::Vec3d wld_pt = world_coordinate.at<cv::Vec3d>(y, x);
        Eigen::VectorXf label = Eigen::Vector3f(wld_pt[0], wld_pt[1], wld_pt[2]);
        labels.push_back(label);
    }
    assert(features.size() == labels.size());
    
    if (verbose) {
        printf("rgb image is %s\n", rgb_img_file);
        printf("depth image is %s\n", depth_img_file);
        printf("camera pose file is %s\n", camera_pose_file);
        printf("sampled %lu samples\n", features.size());
    }
    return ;
}

void
BTRNDUtil::extractWHFeatureFromRgbImages(const char * rgb_img_file,
                                        vector<SCRFRandomSample> & features,
                                        const int single_channel_dim,
                                        const bool verbose)
{
    vector<cv::Point2d> locations;
    for (int i =0; i<features.size(); i++) {
        double x = features[i].p2d_[0];
        double y = features[i].p2d_[1];
        locations.push_back(cv::Point2d(x, y));
    }
    
    cv::Mat rgb_img;
    CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
    
    vector<Eigen::VectorXf> local_features;
    CvxWalshHadamard::generateWHFeatureWithoutFirstPattern(rgb_img, locations, 64, single_channel_dim, local_features);
    
    assert(local_features.size() == locations.size());
    
    for (int i = 0; i<features.size(); i++) {
        features[i].x_descriptor_ = local_features[i]/local_features[i].norm();
    }
}



