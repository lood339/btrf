//  Created by jimmy on 2017-01-21.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __btrf_forest__
#define __btrf_forest__

#include <stdio.h>
#include <vector>
#include "btrf_tree.hpp"

using std::vector;

/*
 This the random forest
 The main function is "predict" that predict 3D location using
 random feature and local descriptors, such as WHT feature
 The forest is trained from "BTRFForestBuilder"
 */

class BTRFForest
{
    friend class BTRFForestBuilder;   
    
    typedef BTRFTree *  TreePtr;
    typedef SCRFRandomSample FeatureType;
    typedef BTRFTreeParameter  TreeParameter;
    
    vector<TreePtr> trees_;
    TreeParameter tree_param_;
    DatasetParameter dataset_param_;    
   
    int label_dim_;   // label dimesion, for 3D location, it is 3
public:
    BTRFForest();
    ~BTRFForest();
    
    // feature: testing sample
    // rgb_image: testing image
    // max_check: back tracking parameter
    // predictions: predictions from each tree, output
    //              ordered by distance
    // dists: local patch descriptor distance, output    
    // return: true if every tree gives a prediction
    bool predict(const FeatureType & feature,
                 const cv::Mat & rgb_image,
                 const int max_check,
                 vector<Eigen::VectorXf> & predictions,
                 vector<float> & dists) const;
    
    const TreeParameter & getTreeParameter(void) const;
    const DatasetParameter & getDatasetParameter(void) const;
    
    
    
    // save model to a .txt file
    bool saveModel(const char *file_name) const;
    
    // load trained model
    bool loadModel(const char *file_name);
    
    
};


#endif /* defined(__btrf_forest__) */
