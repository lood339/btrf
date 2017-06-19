//  Created by jimmy on 2016-08-28.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __cvx_wht__
#define __cvx_wht__

// Walsh Hadamard transform (WHT) feature

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <Eigen/Dense>

using std::vector;
using cv::Mat;


class CvxWalshHadamardTransform
{
public:
    
    // WHT feature without the first pattern which has all positive component
    // points:    locations in image
    // patch_size: unit pixel, 2^{2,3,4,5}
    // kernel_num: feature dimension in single channel
    // features: output, feature dimension == (kernel_num - 1) * 3
    static
    bool generateWHFeatureWithoutFirstPattern(const cv::Mat & rgb_image,
                                              const vector<cv::Point2d> & points,
                                              const int patch_size,
                                              const int kernel_num,
                                              vector<Eigen::VectorXf> & features);
private:
    // rgb_image: input image
    // points:    locations in image
    // patch_size: unit pixel, 2^{2,3,4,5}
    // kernel_num: feature dimension in single channel
    // features: output, feature dimension == kernel_num * 3
    static
    bool generateWHFeature(const cv::Mat & rgb_image,
                           const vector<cv::Point2d> & points,
                           const int patch_size,
                           const int kernel_num,
                           vector<Eigen::VectorXf> & features);
    
    
    
    
    
};

#endif /* defined(__cvx_wht__) */
