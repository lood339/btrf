//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_tree__
#define __RGBD_RF__bt_rnd_tree__

// backtracking decision tree
// a forest has multiple independent trees
#include <stdio.h>
#include <algorithm>
// flann
#include <flann/util/heap.h>
#include <flann/util/result_set.h>
#include <flann/flann.hpp>
// Eigen
#include <Eigen/Dense>

#include "btrf_util.h"
#include "dt_random.h"


using std::vector;
using Eigen::VectorXf;
using flann::BranchStruct;

class BTRNDTreeNode;

class BTRNDTree
{
private:
    friend class BTRFForest;
    
    typedef flann::L2<float> Distance;
    typedef Distance::ResultType DistanceType;
    typedef Distance::ElementType ElementType;
    
    typedef BTRNDTreeNode Node;
    typedef BTRNDTreeNode* NodePtr;
    typedef BranchStruct<NodePtr, DistanceType > BranchSt;
    typedef BranchSt* Branch;
    
    typedef SCRFRandomSample FeatureType;
    
    NodePtr root_;    // tree root
    BTRNDTreeParameter tree_param_; // tree parameter
    
    Distance distance_;   // the distance functor
    int leaf_node_num_;   // total leaf node number
    vector<NodePtr> leaf_nodes_;   // leaf node for back tracking
    
    DTRandom rnd_generator_;   // random number generator
    
public:
    BTRNDTree();
    ~BTRNDTree();
    
    // build a decision tree using training examples
    // features: sampled image pixel locations
    // labels: 3D location
    // indices: index of samples
    // rgb_images: training RGB images, 8bit
    // param: tree parameter
    bool buildTree(const vector<FeatureType> & features,
                   const vector<VectorXf> & labels,
                   const vector<unsigned int> & indices,
                   const vector<cv::Mat> & rgb_images,
                   const BTRNDTreeParameter & param);
    
    // predict 3D locaiton of a pixel in an image
    // feature: input, testing feature location
    // rgb_image: input, an RGB image
    // max_check: input, back tracking parameter, >= 1
    // pred: output predicted 3D location
    // dist: output local patch descriptor distance
    bool predict(const FeatureType & feature,
                 const cv::Mat & rgb_image,
                 const int max_check,
                 VectorXf & pred,
                 float & dist) const;
    
    
    
    // for model save/load, each row is a descriptor, row index is leaf node index
    void getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    void setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    
    // tree parameters
    const BTRNDTreeParameter & getTreeParameter(void) const {return tree_param_;}
    void setTreeParameter(const BTRNDTreeParameter & param) {tree_param_ = param;}
    
private:
    // build tree implementation
    bool buildTreeImpl(const vector<FeatureType> & features,
                       const vector<VectorXf> & labels,
                       const vector<cv::Mat> & rgb_images,
                       const vector<unsigned int> & indices,                       
                       NodePtr node);
    
    // optimize split parameter
    // left_indices: example index in leaf child node, output
    // right_indices: example index in right child node, output
    // split_param: split parameter, output
    double optimizeRandomFeature(const vector<FeatureType> & features,
                                 const vector<VectorXf> & labels,
                                 const vector<cv::Mat> & rgbImages,
                                 const vector<unsigned int> & indices,                                
                                 const int depth,
                                 vector<unsigned int> & left_indices,
                                 vector<unsigned int> & right_indices,
                                 RandomSplitParameter & split_param) const;
    
    // optimize threshold of the random feature
    double optimizeThreshold(const vector<FeatureType> & features,
                             const vector<VectorXf> & labels,
                             const vector<cv::Mat> & rgbImages,
                             const vector<unsigned int> & indices,
                             const int depth,
                             RandomSplitParameter & split_param,
                             vector<unsigned int> & left_indices,
                             vector<unsigned int> & right_indices) const;
    // generate leaf node parameters
    bool setLeafNode(const vector<FeatureType> & features,
                     const vector<VectorXf> & labels,
                     const vector<unsigned int> & indices,
                     NodePtr node);
    
    // record leaf node point in an array for O(1) access
    void hashLeafNode();
    
    void recordLeafNodes(NodePtr node,
                         vector<NodePtr> & leafNodes,
                         int & leafnode_index);
    
    // priority search tree, this function is modifed from flann
    // result_set: search result
    // vec: local descriptor
    // node: current tree node
    // check_count: number of checked leaf node
    // max_check  : threshold of backtracking
    // heap: priority heap
    // sample: testing sample
    // rgb_image: testing image
    void searchLevel(flann::ResultSet<DistanceType>  & result_set,
                     const ElementType* vec,
                     const NodePtr node,
                     int & check_count,
                     const int max_check,
                     flann::Heap<BranchSt>* heap,
                     const FeatureType & sample,
                     const cv::Mat & rgb_image) const;
    
    // from CVPR2015 paper
    // "Exploiting uncertainty in regression forests for accurate camera relocalization"
    // depth adapted RGB pixel comparison feature
    // image: an RGB image
    // feat: feature location
    // split: offset and color channel
    // return: pixel comparision feature
    static double computeRandomFeature(const cv::Mat & rgb_image,
                                       const FeatureType * feat,
                                       const RandomSplitParameter & split);
    
    
};


#endif /* defined(__RGBD_RF__bt_rnd_tree__) */
