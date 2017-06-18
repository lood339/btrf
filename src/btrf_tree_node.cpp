//
//  bt_rnd_tree_node.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btrf_tree_node.h"

BTRNDTreeNode::BTRNDTreeNode(int depth)
{
    left_child_ = NULL;
    right_child_ = NULL;
    depth_   = depth;
    is_leaf_ = false;
    index_ = -1;
}

BTRNDTreeNode::~BTRNDTreeNode()
{
    if (left_child_) {
        delete left_child_;
        left_child_ = NULL;
    }
    if (right_child_) {
        delete right_child_;
        right_child_ = NULL;
    }
}

void BTRNDTreeNode::writeNode(FILE *pf, const NodePtr node)
{
    if (!node) {
        // empty node, end of a tree branch,
        fprintf(pf, "#\n");
        return;
    }
    
    // write current node split parameter
    BTRNDTreeNode::SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %2d\t %2d\t %lf\t %lf\t %lf\n",
            node->depth_, (int)node->is_leaf_, param.split_channles_[0], param.split_channles_[1],
            param.offset_[0], param.offset_[1],
            param.threshold_);
    
    if (node->is_leaf_) {
        // leaf node index and mean label dimension
        fprintf(pf, "%d\n", node->index_);
        // store mean value and standard deviation
        for (int i = 0; i<node->label_mean_.size(); i++) {
            fprintf(pf, "%lf ", node->label_mean_[i]);
        }
        fprintf(pf, "\n");
        for (int i = 0; i<node->label_stddev_.size(); i++) {
            fprintf(pf, "%lf ", node->label_stddev_[i]);
        }
        fprintf(pf, "\n");
    }
    
    // recursively write left and wright child node
    BTRNDTreeNode::writeNode(pf, node->left_child_);
    BTRNDTreeNode::writeNode(pf, node->right_child_);
}

bool BTRNDTreeNode::writeTree(const char *fileName, const NodePtr root,
                              const int n_leaf_node, const int label_dim)
{
    
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d\t %d\n", n_leaf_node, label_dim);
    fprintf(pf, "depth\t isLeaf\t c1\t c2\t offset_x\t offset_y\t threshold\t mean\t stddev\n");
    BTRNDTreeNode::writeNode(pf, root);
    fclose(pf);
    return true;
}

bool BTRNDTreeNode::readTree(const char *fileName, NodePtr & root,
                             int & leafNodeNum)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    
    // read leaf node number
    // and remove ending '\n'
    int label_dim = 0;
    int ret = fscanf(pf, "%d %d \n", &leafNodeNum, &label_dim);
    assert(ret == 1);
    
    //read marking line,
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    
    BTRNDTreeNode::readNode(pf, root, label_dim);
    fclose(pf);
    return true;
}

void BTRNDTreeNode::readNode(FILE *pf, NodePtr & node, const int label_dim)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    // empty node, end of a tree branch
    if (lineBuf[0] == '#') {
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new BTRNDTreeNode(0);
    assert(node);
    int depth = 0;
    int is_leaf = 0;
    int c1 = 0, c2 =0;
    double offset_x = 0.0, offset_y = 0.0;
    double threshold = 0.0;
    
    // internal node parameter
    int ret_num = sscanf(lineBuf, "%d %d %d %d %lf %lf %lf",
                         &depth, &is_leaf, &c1, &c2, &offset_x, &offset_y, &threshold);
    assert(ret_num == 7);
    
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    
    BTRNDTreeNode::SplitParameter& param = node->split_param_;
    param.split_channles_[0] = c1;
    param.split_channles_[1] = c2;
    param.offset_ = Eigen::Vector2f(offset_x, offset_y);
    param.threshold_ = threshold;    
    
    if (is_leaf) {
        int index = 0;
        ret_num = fscanf(pf, "%d", &index);
        assert(ret_num == 1);
        Eigen::VectorXf mean = Eigen::VectorXf::Zero(label_dim);
        Eigen::VectorXf stddev = Eigen::VectorXf::Zero(label_dim);
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            mean[i] = val;
        }
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            stddev[i] = val;
        }
        // remove '\n' at the end of the line
        char dummy_line_buf[1024] = {NULL};
        fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
        node->label_mean_ = mean;
        node->label_stddev_ = stddev;
        node->index_ = index;
    }
    
    BTRNDTreeNode::readNode(pf, node->left_child_, label_dim);
    BTRNDTreeNode::readNode(pf, node->right_child_, label_dim);
}


