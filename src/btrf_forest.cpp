//
//  bt_rnd_regressor.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-21.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btrf_forest.h"
#include "btrf_tree.h"
#include "yael_io.h"
#include "btrf_tree_node.h"
#include "cvx_util.hpp"


BTRFForest::BTRFForest()
{
    label_dim_ = 3;
}

BTRFForest::~BTRFForest()
{
    // @todo release memory in each tree
}

bool BTRFForest::predict(const FeatureType & feature,
                             const cv::Mat & rgb_image,
                             const int max_check,
                             vector<Eigen::VectorXf> & predictions,
                             vector<float> & dists) const
{
    assert(trees_.size() > 0);
    assert(predictions.size() == 0);
    assert(dists.size() == 0);
    
    // Step 1: predict from each tree
    vector<Eigen::VectorXf> unordered_predictions;
    vector<float> unordered_dists;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        float dist;
        bool is_pred = trees_[i]->predict(feature, rgb_image, max_check, cur_pred, dist);
        if (is_pred) {
            unordered_predictions.push_back(cur_pred);
            unordered_dists.push_back(dist);
        }
    }
    
    // Step 2: ordered by local patch feature distance
    vector<size_t> sortIndexes = CvxUtil::sortIndices<float>(unordered_dists);
    for (int i = 0; i<sortIndexes.size(); i++) {
        predictions.push_back(unordered_predictions[sortIndexes[i]]);
        dists.push_back(unordered_dists[sortIndexes[i]]);
    }
    
    return predictions.size() > 0;
    assert(predictions.size() == dists.size());    
    return predictions.size() == trees_.size();
}

const BTRNDTreeParameter & BTRFForest::getTreeParameter(void) const
{
    return tree_param_;
}

const DatasetParameter & BTRFForest::getDatasetParameter(void) const
{
    return dataset_param_;
}
const BTRNDTree * BTRFForest::getTree(int index) const
{
    assert(index >=0 && index < trees_.size());
    return trees_[index];
}

bool BTRFForest::saveModel(const char *file_name) const
{
    assert(trees_.size() > 0);
    assert(strlen(file_name) < 1000);
    
    // write forest and tree files
    FILE *pf = fopen(file_name, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", file_name);
        return false;
    }
    fprintf(pf, "%d\n", label_dim_);
    assert(label_dim_ > 0);
    
    // Step 1: dataset parameter e.g., 4 Scenes dataset
    dataset_param_.writeToFile(pf);
    // Step 2: tree parameter
    tree_param_.writeToFile(pf);
    
    // Step 3: store trees
    // Each tree has two parts:
    // 1. tree structure, store in tree_file
    // 2. local patch descriptor in leaf node, store in leaf_node_file
    string base_name = string(file_name);  // model name without postfix, e.g., '.txt'
    base_name = base_name.substr(0, base_name.size()-4);
    
    // tree structure file
    for (int i = 0; i<trees_.size(); i++) {
        char tree_file_name[1024] = {NULL};
        char leaf_file_name[1024] = {NULL};
        sprintf(tree_file_name, "%s_%08d.txt", base_name.c_str(), i);
        sprintf(leaf_file_name, "%s_%08d.fvec", base_name.c_str(), i);
        fprintf(pf, "%s\n", tree_file_name);
        fprintf(pf, "%s\n", leaf_file_name);
        
        if (trees_[i]) {
            // 1. write tree structure
            BTRNDTreeNode::writeTree(tree_file_name,
                                     trees_[i]->root_,
                                     trees_[i]->leaf_node_num_,
                                     label_dim_);
            
            // 2. write leaf descriptor
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> leaf_descriptor; // temporal data
            // get descriptors from leaf node
            trees_[i]->getLeafNodeDescriptor(leaf_descriptor);
            // store as binary file
            YaelIO::write_fvecs_file(leaf_file_name, leaf_descriptor);
        }
    }
    
    fclose(pf);
    printf("save to %s\n", file_name);
    return true;
}

bool BTRFForest::loadModel(const char *model_file_name)
{
    FILE *pf = fopen(model_file_name, "r");
    if (!pf) {
        printf("Error: can not read file %s\n", model_file_name);
        return false;
    }
    
    int ret_num = fscanf(pf, "%d", &label_dim_);
    assert(ret_num == 1);
    assert(label_dim_ > 0);
    
    bool is_read = dataset_param_.readFromFile(pf);
    assert(is_read);
    dataset_param_.printSelf();
    
    is_read = tree_param_.readFromFile(pf);
    assert(is_read);
    tree_param_.printSelf();
    
    // read tree file and leaf node descriptor file
    vector<string> treeFiles;
    vector<string> leaf_node_files;
    for (int i = 0; i<tree_param_.tree_num_; i++) {
        {
            char buf[1024] = {NULL};
            fscanf(pf, "%s", buf);
            treeFiles.push_back(string(buf));
        }
        {
            char buf[1024] = {NULL};
            fscanf(pf, "%s", buf);
            leaf_node_files.push_back(string(buf));
        }
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            delete trees_[i];
            trees_[i] = NULL;
        }
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<treeFiles.size(); i++) {
        BTRNDTreeNode * root = NULL;
        int leaf_node_num = 0;
        
        // read tree structure
        is_read = BTRNDTreeNode::readTree(treeFiles[i].c_str(), root, leaf_node_num);
        assert(is_read);
        assert(root);
        
        BTRNDTree *tree = new BTRNDTree();
        assert(tree);
        tree->root_ = root;
        tree->setTreeParameter(tree_param_);
        tree->leaf_node_num_ = leaf_node_num;
        
        // read leaf node descriptor and set it in the tree
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> leaf_node_feature;
        is_read = YaelIO::read_fvecs_file(leaf_node_files[i].c_str(), leaf_node_feature);
        assert(is_read);
        tree->setLeafNodeDescriptor(leaf_node_feature);
        
        trees_.push_back(tree);
    }
    printf("read from: %s, tree nume: %lu\n", model_file_name, trees_.size());
    return true;
}
