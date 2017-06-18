//  Created by jimmy on 2017-01-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_util__
#define __RGBD_RF__bt_rnd_util__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include "dataset_param.h"

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
    SCRFRandomSample()
    {
        image_index_ = 0;
        inv_depth_ = 1.0;
    }
    Eigen::Vector2f addOffset(const  Eigen::Vector2f & offset) const
    {
        return p2d_ + offset * inv_depth_;
    }
};

// random feature split parameter
class RandomSplitParameter
{
public:
    int split_channles_[2];      // rgb image channel, c1 and c2
    Eigen::Vector2f offset_;     // image location offset, [x, y]
    double threshold_;           // threshold of splitting. store result
    
    RandomSplitParameter()
    {
        split_channles_[0] = 0;
        split_channles_[1] = 1;
        offset_[0] = 0.0f;
        offset_[1] = 0.0f;
        threshold_ = 0.0;
    }
    
    RandomSplitParameter(const RandomSplitParameter & other)
    {
        split_channles_[0] = other.split_channles_[0];
        split_channles_[1] = other.split_channles_[1];
        offset_ = other.offset_;
        threshold_ = other.threshold_;        
    }
    
};


class BTRNDTreeParameter
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
    
    
    BTRNDTreeParameter()
    {
        // sampler parameters 3
        is_use_depth_ = true;
        max_frame_num_ = 500;
        sampler_num_per_frame_ = 5000;
        
        // tree structure parameter 5
        tree_num_ = 5;
        max_depth_ = 15;
        max_balanced_depth_ = 8;
        min_leaf_node_ = 50;
        min_split_node_std_dev_ = 0.05;
        
        // random sample parameter 5
        max_pixel_offset_ = 131;
        pixel_offset_candidate_num_ = 10;
        split_candidate_num_ = 20;
        verbose_ = true;
        verbose_leaf_ = false;
        
        // feature type parameter 1
        wh_kernel_size_ = 20;
    }
    
    bool readFromFile(const char *fileName)
    {
        FILE *pf = fopen(fileName, "r");
        if (!pf) {
            printf("can not read from %s \n", fileName);
            return false;
        }
        
        this->readFromFile(pf);
        fclose(pf);
        return true;
    }

    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        const int param_num = 14;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                printf("read tree parameter Error: %s %f\n", s, val);
                assert(ret == 2);
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 14);
        
        is_use_depth_ = (imap[string("is_use_depth")] == 1);
        max_frame_num_ = imap[string("max_frame_num")];
        sampler_num_per_frame_ = imap[string("sampler_num_per_frame")];
        
        tree_num_ = imap[string("tree_num")];
        max_depth_ = imap[string("max_depth")];
        max_balanced_depth_ = imap[string("max_balanced_depth")];
        min_leaf_node_ = imap[string("min_leaf_node")];
        min_split_node_std_dev_ = imap[string("min_split_node_std_dev")];
        
        max_pixel_offset_ = imap[string("max_pixel_offset")];
        pixel_offset_candidate_num_ = imap[string("pixel_offset_candidate_num")];
        split_candidate_num_ = imap[string("split_candidate_num")];
        wh_kernel_size_ = imap[string("wh_kernel_size")];
        
        verbose_ = imap[string("verbose")];
        verbose_leaf_ = imap[string("verbose_leaf")];
        return true;
    }

    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "is_use_depth %d\n", is_use_depth_);
        fprintf(pf, "max_frame_num %d\n", max_frame_num_);
        fprintf(pf, "sampler_num_per_frame %d\n\n", sampler_num_per_frame_);
        
        fprintf(pf, "tree_num %d\n", tree_num_);
        fprintf(pf, "max_depth %d\n", max_depth_);
        fprintf(pf, "max_balanced_depth %d\n", max_balanced_depth_);
        fprintf(pf, "min_leaf_node %d\n", min_leaf_node_);
        fprintf(pf, "min_split_node_std_dev %f\n\n", min_split_node_std_dev_);
        
        fprintf(pf, "max_pixel_offset %d\n", max_pixel_offset_);
        fprintf(pf, "pixel_offset_candidate_num %d\n", pixel_offset_candidate_num_);
        fprintf(pf, "split_candidate_num %d\n", split_candidate_num_);
        fprintf(pf, "wh_kernel_size %d\n\n", wh_kernel_size_);
        
        fprintf(pf, "verbose %d\n", (int)verbose_);
        fprintf(pf, "verbose_leaf %d\n\n", (int)verbose_leaf_);
        return true;
    }
    
    void printSelf() const
    {
        writeToFile(stdout);
    }
};


class BTRNDUtil
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


#endif /* defined(__RGBD_RF__bt_rnd_util__) */
