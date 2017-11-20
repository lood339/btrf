//  Created by jimmy on 2017-06-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __btrf_param__
#define __btrf_param__

#include <stdio.h>

#include <Eigen/Dense>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include "dataset_param.hpp"

using Eigen::VectorXf;
using std::vector;

// training sample using Scene Coordinate Regression Forests (SCRF) method
// Feature location and local descriptor,
// The actual random feature is learned in RandomSplitParameter
class SCRFRandomSample
{
public:
    Eigen::Vector2f p2d_;    // 2d location (x, y)
    double inv_depth_;       // inverted depth, optional
    int image_index_;        // image index, used to query image during training
    
    Eigen::VectorXf x_descriptor_; // (SIFT, WH) descriptor, default value is 1/64 or 1/128.0
    
public:
    SCRFRandomSample();
    // depth adaptive offset
    Eigen::Vector2f addOffset(const  Eigen::Vector2f & offset) const;
};

// random feature split parameter
class RandomSplitParameter
{
public:
    int split_channles_[2];      // rgb image channel, c1 and c2
    Eigen::Vector2f offset_;     // image location offset, [x, y]
    double threshold_;           // threshold of splitting. store result
    
    RandomSplitParameter();
    RandomSplitParameter(const RandomSplitParameter & other);
};


class BTRFTreeParameter
{
public:
    bool is_use_depth_;           // true --> use depth, false depth is constant 1.0
    int max_frame_num_;           // sampled frames for a tree
    int sampler_num_per_frame_;   // sampler numbers in one frame
    
    int tree_num_;                // number of trees
    int max_depth_;               // maximum tree depth
    int max_balanced_depth_;     // 0 - max_balanced_tree_depth, encourage balanced tree instead of smaller entropy
    
    int min_leaf_node_;           // minimum leaf node size
    double min_split_node_std_dev_;           // 0.05 meter
    
    int max_pixel_offset_;            // in pixel
    int pixel_offset_candidate_num_;  // large number less randomness
    int split_candidate_num_;        // number of split in [v_min, v_max]
    bool verbose_;                   // output training
    bool verbose_leaf_;              // output leaf information
    
    int wh_kernel_size_;               // Walsh Hadamard kernel size in singe channel
    
    
    
    BTRFTreeParameter();
    
    bool readFromFile(const char *fileName);
    
    bool readFromFile(FILE *pf);
    
    bool writeToFile(FILE *pf) const;
    
    void printSelf() const;
    
};



#endif /* defined(__btrf_param__) */
