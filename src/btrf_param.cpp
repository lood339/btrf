//
//  btrf_param.cpp
//  BTRF
//
//  Created by jimmy on 2017-06-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btrf_param.hpp"

SCRFRandomSample::SCRFRandomSample()
{
    image_index_ = 0;
    inv_depth_ = 1.0;
}
Eigen::Vector2f SCRFRandomSample::addOffset(const  Eigen::Vector2f & offset) const
{
    return p2d_ + offset * inv_depth_;
}


RandomSplitParameter::RandomSplitParameter()
{
    split_channles_[0] = 0;
    split_channles_[1] = 1;
    offset_[0] = 0.0f;
    offset_[1] = 0.0f;
    threshold_ = 0.0;
}

RandomSplitParameter::RandomSplitParameter(const RandomSplitParameter & other)
{
    split_channles_[0] = other.split_channles_[0];
    split_channles_[1] = other.split_channles_[1];
    offset_ = other.offset_;
    threshold_ = other.threshold_;
}


BTRFTreeParameter::BTRFTreeParameter()
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

bool BTRFTreeParameter::readFromFile(const char *fileName)
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

bool BTRFTreeParameter::readFromFile(FILE *pf)
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

bool BTRFTreeParameter::writeToFile(FILE *pf)const
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

void BTRFTreeParameter::printSelf() const
{
    writeToFile(stdout);
}




