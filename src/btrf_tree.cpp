//
//  bt_rnd_tree.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btrf_tree.hpp"
#include "btrf_tree_node.hpp"
#include "dt_random.hpp"
#include "cvx_util.hpp"
#include "dt_util.hpp"
#include <iostream>


using std::cout;
using std::endl;

BTRFTree::BTRFTree()
{
    root_ = NULL;
    leaf_node_num_ = 0;    
}

BTRFTree::~BTRFTree()
{
    if (root_) {
        delete root_;
        root_ = NULL;
    }
}

bool BTRFTree::buildTree(const vector<FeatureType> & features,
                          const vector<VectorXf> & labels,
                          const vector<unsigned int> & indices,
                          const vector<cv::Mat> & rgb_images,
                          const BTRNDTreeParameter & param)
{
    
    assert(indices.size() <= features.size());
    assert(labels.size() == features.size());
    
    // Step 1: build a tree
    root_ = new Node(0);
    tree_param_ = param;
    leaf_node_num_ = 0;
    this->buildTreeImpl(features, labels, rgb_images, indices, root_);
    assert(leaf_node_num_ > 0);
    
    // Step 2: record leaf node
    this->hashLeafNode();
    return true;
}

bool BTRFTree::buildTreeImpl(const vector<FeatureType> & features,
                              const vector<VectorXf> & labels,
                              const vector<cv::Mat> & rgb_images,
                              const vector<unsigned int> & indices,                              
                              NodePtr node)
{
    assert(indices.size() <= features.size());
    int depth = node->depth_;
    
    // Step 1: check if reaches a leaf node
    // Stop if tree depth is larger than a threshold
    // or number of samples is smaller than a threshold
    if (depth >= tree_param_.max_depth_ ||
        indices.size() <= tree_param_.min_leaf_node_) {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    // Early stop by checking standard deviation of labels,
    if (depth > tree_param_.max_depth_/2) {
        double variance = DTUtil::spatialVariance(labels, indices);
        double std_dev = sqrt(variance/indices.size());
        if (std_dev < tree_param_.min_split_node_std_dev_) {
            return this->setLeafNode(features, labels, indices, node);
        }
    }
    
    // Step 2: learn split parameter, computaional bottleneck
    // split samples into left and right node using random feature
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    RandomSplitParameter rnd_split_param;
    double min_loss = this->optimizeRandomFeature(features, labels, rgb_images, indices,
                                                  depth,
                                                  left_indices, right_indices, rnd_split_param);
    
    bool is_split = min_loss < std::numeric_limits<double>::max();
    if (is_split) {
        // Split training data by random feature
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_ && depth < tree_param_.max_balanced_depth_) {
            printf("left, right node number is %lu %lu, percentage: %f \n", left_indices.size(),
                   right_indices.size(),
                   100.0*left_indices.size()/indices.size());
        }
        // store split parameters
        node->split_param_ = rnd_split_param;
        
        // recursively split node
        if (left_indices.size() != 0) {
            NodePtr left_node = new Node(depth + 1);
            this->buildTreeImpl(features, labels, rgb_images, left_indices, left_node);
            node->left_child_ = left_node;
        }
        if (right_indices.size() != 0) {
            Node *right_node = new Node(depth + 1);   // increase depth
            this->buildTreeImpl(features, labels, rgb_images, right_indices,  right_node);
            node->right_child_ = right_node;
        }
    }
    else
    {
        // Early stop
        return this->setLeafNode(features, labels, indices, node);
    }
    return true;
}

double BTRFTree::optimizeRandomFeature(const vector<FeatureType> & features,
                                        const vector<VectorXf> & labels,
                                        const vector<cv::Mat> & rgb_images,
                                        const vector<unsigned int> & indices,
                                        const int depth,
                                        vector<unsigned int> & left_indices,   //output
                                        vector<unsigned int> & right_indices,
                                        RandomSplitParameter & split_param) const
{
    // split samples into left and right node
    const int max_pixel_offset = tree_param_.max_pixel_offset_;
    const int max_channel = 3;
    const int max_random_num   = tree_param_.pixel_offset_candidate_num_;
    
    double min_loss = std::numeric_limits<double>::max();
    for (int i = 0; i<max_random_num; i++) {
        // Step 1: generate random features: pixel offset and color channels
        double x_offset = rnd_generator_.getRandomNumber(-max_pixel_offset, max_pixel_offset);
        double y_offset = rnd_generator_.getRandomNumber(-max_pixel_offset, max_pixel_offset);
        
        RandomSplitParameter cur_split_param;
        cur_split_param.offset_[0] = x_offset;
        cur_split_param.offset_[1] = y_offset;
        cur_split_param.split_channles_[0] = rand()%max_channel;
        cur_split_param.split_channles_[1] = rand()%max_channel;
        
        
        // Step 2: optimize the threshold
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_loss = this->optimizeThreshold(features, labels, rgb_images, indices,
                                                  depth,
                                                  cur_split_param, cur_left_indices, cur_right_indices);
        
        if (cur_loss < min_loss) {
            min_loss = cur_loss;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
            split_param = cur_split_param;
        }
    }
    return min_loss;
}

double BTRFTree::computeRandomFeature(const cv::Mat & rgb_image, const FeatureType * feat, const RandomSplitParameter & split)
{
    Eigen::Vector2f p1 = feat->p2d_;
    Eigen::Vector2f p2 = feat->addOffset(split.offset_);
    
    // feature channels
    const int c1 = split.split_channles_[0];
    const int c2 = split.split_channles_[1];
    
    // feature location
    int p1x = p1[0];
    int p1y = p1[1];
    int p2x = p2[0];
    int p2y = p2[1];
    
    bool is_inside_image2 = CvxUtil::isInside(rgb_image.cols, rgb_image.rows, p2x, p2y);
    double pixel_1_c = 0.0;   // out of image as black pixels, another option is random pixel values in [0, 255]
    double pixel_2_c = 0.0;
    
    // pixel value at (y, x) [c], y is vertical, x is horizontal
    pixel_1_c = (rgb_image.at<cv::Vec3b>(p1y, p1x))[c1]; // (row, col)
    
    if (is_inside_image2) {
        pixel_2_c = (rgb_image.at<cv::Vec3b>(p2y, p2x))[c2];
    }
    return pixel_1_c - pixel_2_c;
}

double
BTRFTree::optimizeThreshold(const vector<FeatureType> & features,
                                    const vector<VectorXf> & labels,
                                    const vector<cv::Mat> & rgb_images,
                                    const vector<unsigned int> & indices,
                                    const int depth,
                                    RandomSplitParameter & split_param,
                                    vector<unsigned int> & left_indices,
                                    vector<unsigned int> & right_indices) const
{
    double min_loss = std::numeric_limits<double>::max();
    const int min_node_size = tree_param_.min_leaf_node_;
    const int split_candidate_num = tree_param_.split_candidate_num_;
    const int max_balance_depth = tree_param_.max_balanced_depth_;
    
    // Step 1: compute pixel difference
    vector<double> feature_values(indices.size(), 0.0); // 0.0 for invalid pixels
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < features.size());
        
        const FeatureType* smp = &(features[index]);  // avoid copy, use pointer
        feature_values[i] = BTRFTree::computeRandomFeature(rgb_images[smp->image_index_], smp, split_param);
    }
    
    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());
    if (!(min_v < max_v)) {
        return min_loss;
    }
    
    // Step 2: generate random thresholds as split values
    vector<double> split_values = rnd_generator_.getRandomNumbers(min_v, max_v, split_candidate_num);
   
    
    // Step 3: choose one threshold that has smallest loss
    bool is_split = false;
    for (int i = 0; i<split_values.size(); i++) {
        double split_v = split_values[i];
        vector<unsigned int> cur_left_index;
        vector<unsigned int> cur_right_index;
        double cur_loss = 0;
        for (int j = 0; j<feature_values.size(); j++) {
            int index = indices[j];
            if (feature_values[j] < split_v) {
                cur_left_index.push_back(index);
            }
            else
            {
                cur_right_index.push_back(index);
            }
        }
        assert(cur_left_index.size() + cur_right_index.size() == indices.size());       
        
        // avoid too small internal node
        if (cur_left_index.size() < min_node_size/2 || cur_right_index.size() < min_node_size/2) {
            continue;
        }
        
        if (depth <= max_balance_depth) {
            // Object one: sample-balanced objective
            cur_loss = DTUtil::balanceLoss((int)cur_left_index.size(), (int)cur_right_index.size());
        }
        else {
            // Object two: spatial variance objective
            cur_loss = DTUtil::spatialVariance(labels, cur_left_index);
            if (cur_loss > min_loss) {
                continue;
            }
            cur_loss += DTUtil::spatialVariance(labels, cur_right_index);
        }
        
        if (cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices  = cur_left_index;
            right_indices = cur_right_index;
            split_param.threshold_ = split_v;
        }
    }
    if (!is_split) {
        return min_loss;
    }
    assert(left_indices.size() + right_indices.size() == indices.size());
    
    return min_loss;
}


bool BTRFTree::setLeafNode(const vector<FeatureType> & features,
                            const vector<VectorXf> & labels,
                            const vector<unsigned int> & indices,
                            NodePtr node)
{
    // local patch descriptor, such as WHT feature
    vector<Eigen::VectorXf> local_features(indices.size());
    for (int i = 0; i<indices.size(); i++) {
        int idx = indices[i];
        local_features[i] = features[idx].x_descriptor_;
    }
    node->feat_mean_ = DTUtil::mean<Eigen::VectorXf>(local_features);
    
    // 3D location mean and standard deviation
    node->is_leaf_ = true;
    DTUtil::meanStddev<Eigen::VectorXf>(labels, indices,
                                           node->label_mean_,
                                           node->label_stddev_);
    leaf_node_num_++;
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth: %d example number: %lu\n", node->depth_, indices.size());
        cout<<"mean              : \n"<<node->label_mean_.transpose()<<endl;
        cout<<"standard deviation: \n"<<node->label_stddev_.transpose()<<endl;
    }
    return true;
}


void BTRFTree::hashLeafNode()
{
    assert(leaf_node_num_ > 0);
    leaf_nodes_.resize(leaf_node_num_);
    
    //  traversal tree by pre-order
    int index = 0;  // leaf node index
    this->recordLeafNodes(root_, leaf_nodes_, index);
}

void BTRFTree::recordLeafNodes(NodePtr node, vector<NodePtr> & leafNodes, int & index)
{
    assert(node);
    if (node->is_leaf_) {
        // for tree read from a file, index is pre-computed and stored
        if (node->index_ != -1) {
            assert(node->index_ == index);
        }
        node->index_ = index;
        leafNodes[index] = node;
        index++;
        return;
    }
    if (node->left_child_) {
        this->recordLeafNodes(node->left_child_, leafNodes, index);
    }
    if (node->right_child_) {
        this->recordLeafNodes(node->right_child_, leafNodes, index);
    }
}

bool BTRFTree::predict(const FeatureType & feature,
                        const cv::Mat & rgb_image,
                        const int max_check,
                        VectorXf & pred,
                        float & dist) const
{
    assert(root_);
    
    int check_count = 0;
    const int knn = 1;
    
    BranchSt branch;
    flann::Heap<BranchSt> * heap = new flann::Heap<BranchSt>(leaf_node_num_*2);  // why use so large heap ?
    
    flann::KNNResultSet2<DistanceType> result(knn); // only keep the nearest one
    const ElementType *vec = feature.x_descriptor_.data();
    
    // Step 1: search tree down to leaf
    this->searchLevel(result, vec, root_, check_count, max_check, heap, feature, rgb_image);
    
    // Step 2: back tracking
    while (heap->popMin(branch) &&
           (check_count < max_check || !result.full())) {
        assert(branch.node);
        this->searchLevel(result, vec, branch.node, check_count, max_check, heap, feature, rgb_image);
    }
    delete heap;
    assert(result.size() == knn);
    
    // Step 3: find leaf node index
    size_t index = 0;
    DistanceType distance;
    result.copy(&index, &distance, 1, false);
    
    pred = leaf_nodes_[index]->label_mean_;
    dist = (float)distance;
    return true;
}

void BTRFTree::searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, const NodePtr node,
                            int & check_count, const int max_check,
                            flann::Heap<BranchSt>* heap,
                            const FeatureType & feature,     // new added parameter
                            const cv::Mat & rgb_image) const
{
    // check leaf node
    if (node->is_leaf_) {
        int index = node->index_;  // store leaf node index
        if (check_count >= max_check &&
            result_set.full()) {
            return;
        }        
        check_count++;
        // squared distance, use local patch descriptor
        DistanceType dist = distance_(node->feat_mean_.data(), vec, node->feat_mean_.size());
        result_set.addPoint(dist, index);
        return;
    }
    
    // Step 1: binary test
    // create a branch record for the branch not taken, use random feature
    double rnd_feat = BTRFTree::computeRandomFeature(rgb_image, &feature, node->split_param_);
    DistanceType diff = rnd_feat - node->split_param_.threshold_;
    NodePtr bestChild  = (diff < 0 ) ? node->left_child_: node->right_child_;
    NodePtr otherChild = (diff < 0 ) ? node->right_child_: node->left_child_;
    
    // Step 2: insert all possible branches,
    // because 1. distance measurement in random feature and local patch feature are different
    //         2. binary test uses only one dimension of random feature
    DistanceType dist = (DistanceType)fabs(diff);
    if (!result_set.full()) {
        heap->insert(BranchSt(otherChild, dist));
    }
    
    // Step 3: call recursively to search next level
    this->searchLevel(result_set, vec, bestChild, check_count, max_check, heap, feature, rgb_image);
}

void BTRFTree::getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == leaf_nodes_.size());
    
    const int rows = leaf_node_num_;
    const int cols = (int)leaf_nodes_[0]->feat_mean_.size();
    
    data = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(rows, cols);
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        data.row(i) = leaf_nodes_[i]->feat_mean_;
    }
}

void BTRFTree::setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == data.rows());
    
    this->hashLeafNode();
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        leaf_nodes_[i]->feat_mean_ = data.row(i);
    }
}



