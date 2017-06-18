//
//  cvxWalshHadamard.h
//  RGBD_RF
//
//  Created by jimmy on 2016-08-28.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__cvxWalshHadamard__
#define __RGBD_RF__cvxWalshHadamard__

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <vector>
#include <Eigen/Dense>

using std::vector;
using cv::Mat;


class CvxWalshHadamard
{
public:
    // feature dimension == kernelNum * 3
    // patchSize: pixel
    // kernelNum: feature dimension in single dimension
    static
    bool generateWHFeature(const cv::Mat & rgb_image,
                           const vector<cv::Point2d> & pts,
                           const int patchSize,
                           const int kernelNum,
                           vector<Eigen::VectorXf> & features);
    
    
    // WH feature without first pattern (all positive)
    static
    bool generateWHFeatureWithoutFirstPattern(const cv::Mat & rgb_image,
                           const vector<cv::Point2d> & pts,
                           const int patchSize,
                           const int kernelNum,
                           vector<Eigen::VectorXf> & features);
    
    
};

#endif /* defined(__RGBD_RF__cvxWalshHadamardProjection__) */
