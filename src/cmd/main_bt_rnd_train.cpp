//  Created by jimmy on 2017-01-22.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <string>
#include "cvx_image_310.hpp"
#include "cvx_io.hpp"
#include "ms7scenes_util.hpp"
#include "btrf_forest_builder.hpp"
#include "btrf_forest.hpp"
#include "dataset_param.hpp"

using std::string;

#if 0

static void help()
{
    printf("program       datasetParam      RGBImageList  depthImageList cameraPoseList dtParameterFile  maxCheck saveFile\n");
    printf("BT_RND_train  dataset_4scenes   rgbs.txt      depth.txt      poses.txt      RF_param.txt     32       bt_rgbd_RF.txt\n");
    printf("parameter fits to corresponding dataset to datasetParam \n");
    printf("dtParameterFile: decision tree parameter.\n");
}

int main(int argc, const char * argv[])
{
    if (argc != 8) {
        printf("argc is %d, should be 8 \n", argc);
        help();
        return -1;
    }
    
    const char * dataset_param_filename = argv[1];
    const char * rgb_image_file = argv[2];
    const char * depth_image_file = argv[3];
    const char * camera_to_wld_pose_file = argv[4];
    const char * tree_param_file = argv[5];
    const int max_check = strtod(argv[6], NULL);
    const char * save_model_file = argv[7];
    
    DatasetParameter dataset_param;
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    
    vector<string> rgb_files   = Ms7ScenesUtil::read_file_names(rgb_image_file);
    vector<string> depth_files = Ms7ScenesUtil::read_file_names(depth_image_file);
    vector<string> pose_files  = Ms7ScenesUtil::read_file_names(camera_to_wld_pose_file);
    assert(rgb_files.size() == depth_files.size());
    assert(rgb_files.size() == pose_files.size());

    
    // read from tree parameter
    BTRNDTreeParameter tree_param;
    bool is_read = tree_param.readFromFile(tree_param_file);
    assert(is_read);
    if (tree_param.is_use_depth_) {
        printf("Note: depth is used ...................................................\n");
    }
    else {
        printf("Node: depth is Not used ............................................\n");
    }
    
    
    BTRFForestBuilder builder;
    BTRFForest model;
    builder.setDatasetParameter(dataset_param);
    builder.setTreeParameter(tree_param);
    builder.buildModel(model, rgb_files, depth_files, pose_files, max_check, save_model_file);
    
    model.saveModel(save_model_file);
    printf("save model to %s\n", save_model_file);
    dataset_param.printSelf();
    tree_param.printSelf();
    
    return 0;
}

#endif

