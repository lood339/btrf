//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_tree_node__
#define __RGBD_RF__bt_rnd_tree_node__

// backtracking random tree node
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "btrf_util.h"


using std::vector;
using Eigen::VectorXf;


class BTRNDTreeNode
{
    friend class BTRNDTree;
private:
    
    typedef BTRNDTreeNode* NodePtr;
    typedef RandomSplitParameter SplitParameter;

    BTRNDTreeNode *left_child_;  // left child node
    BTRNDTreeNode *right_child_; // right child node
    int depth_;                  // tree depth from 0
    bool is_leaf_;               // indicator of leaf node
   
    // non-leaf node parameter
    // aka, weaker learner model
    SplitParameter split_param_;
    
    // leaf node parameter
    VectorXf label_mean_;      // mean of labels, e.g., 3D location
    VectorXf label_stddev_;    // standard deviation of labels,
                               // if it is too large, needs a deeper tree
    VectorXf feat_mean_;       // mean value of local descriptors, e.g., WHT features
    int index_;                // leaf node index from 0, for save/store leaf node
    
public:
    // constructor
    BTRNDTreeNode(int depth);
    // de-constructor, release memory
    ~BTRNDTreeNode();
    
    // write a node to a .txt file
    static bool writeTree(const char *file_name, const NodePtr root,
                          const int n_leaf_node, const int label_dim);
    // read a tree from a .txt file
    static bool readTree(const char *file_name, NodePtr & root,
                         int &n_leaf_node);
    
private:
    // write a single node
    static void writeNode(FILE *pf, const NodePtr node);
    // read a single node
    static void readNode(FILE *pf, NodePtr & node, const int label_dim);
};




#endif /* defined(__RGBD_RF__bt_rnd_tree_node__) */
