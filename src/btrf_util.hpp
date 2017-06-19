//  Created by jimmy on 2017-01-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __btrf_util__
#define __btrf_util__

#include "btrf_param.hpp"

class BTRFUtil
{
public:
    
    // sample SCRF features in an RGB-D image using ground truth camera pose
    // num_sample: around 5000
    // image_index: image Id
    // dataset_param: dataset parameter such as focal length
    // use_depth: if use depth image, Set it false, if use RGB features
    // features: output, SCRF feature location
    // labels: output, 3D location
    static void
    randomSampleFromRgbdImages(const char * rgb_img_file,
                               const char * depth_img_file,
                               const char * camera_pose_file,
                               const int num_sample,
                               const int image_index,
                               const DatasetParameter & dataset_param,
                               const bool use_depth,
                               const bool verbose,                               
                               vector<SCRFRandomSample> & features,
                               vector<Eigen::VectorXf> & labels);
    
    
    
    // Walsh Hadamard feature wihtou first pattern
    static void
    extractWHFeatureFromRgbImages(const char * rgb_img_file,
                                  vector<SCRFRandomSample> & features,  // output
                                  const int single_channel_dim,
                                  const bool verbose);   
    
    
private:
    // implementation of randomSampleFromRgbdImages
    static void
    randomSampleFromRgbdImagesImpl(const char * rgb_img_file,
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
                                   vector<Eigen::VectorXf> & labels);

    
    
    
};


#endif /* defined(__btrf_util__) */
