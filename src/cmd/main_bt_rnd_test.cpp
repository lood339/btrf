//
//  main.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include <iostream>
#include "cvx_image_310.hpp"
#include <string>
#include "cvx_io.hpp"
#include <unordered_map>
#include "ms7scenes_util.hpp"
#include "btrf_forest_builder.hpp"
#include "btrf_forest.hpp"

using std::string;

#if 1

static void help()
{
    printf("program     modelFile  RGBImageList depthImageList cameraPoseList numSamplePerImage maxCheck saveFilePrefix\n");
    printf("BT_RND_test bt_RF.txt  rgbs.txt     depth.txt      poses.txt      5000              32       to3d \n");
    printf("parameter fits to MS 7 Scenes, TUM dataset, 4 Scenes\n");
    printf("multiple 3D prediction ordered by feature distances, save files to result folder\n");
}

int main(int argc, const char * argv[])
{
    if (argc != 8) {
        printf("argc is %d, should be 8\n", argc);
        help();
        return -1;
    }
    
    const char * model_file = argv[1];
    const char * rgb_image_file = argv[2];
    const char * depth_image_file = argv[3];
    const char * camera_to_wld_pose_file = argv[4];
    const int num_random_sample = (int)strtod(argv[5], NULL);
    const int max_check = (int)strtod(argv[6], NULL);
    const char * prefix = argv[7];
    
    assert(num_random_sample > 100);
     
    /*
    const char * model_file = "/Users/jimmy/Desktop/IROS_app/model/kitchen_model.txt";
    const char * rgb_image_file = "/Users/jimmy/Desktop/IROS_app/apt1/kitchen/test_files/rgb_image_list.txt";
    const char * depth_image_file = "/Users/jimmy/Desktop/IROS_app/apt1/kitchen/test_files/depth_image_list.txt";
    const char * camera_to_wld_pose_file = "/Users/jimmy/Desktop/IROS_app/apt1/kitchen/test_files/camera_pose_list.txt";
    const int num_random_sample = 5000;
    const int max_check = 16;
    const char * prefix = "temp";
     */
    
    
    vector<string> rgb_files   = CvxIO::read_file_names(rgb_image_file);
    vector<string> depth_files = CvxIO::read_file_names(depth_image_file);
    vector<string> pose_files  = CvxIO::read_file_names(camera_to_wld_pose_file);
    
    assert(rgb_files.size() == depth_files.size());
    assert(rgb_files.size() == pose_files.size());
    
    // read model
    BTRFForest model;
    bool is_read = model.loadModel(model_file);
    if (!is_read) {
        printf("Error: can not read from file %s\n", model_file);
        return -1;
    }
    
    const BTRNDTreeParameter & tree_param = model.getTreeParameter();
    const DatasetParameter  & dataset_param = model.getDatasetParameter();
    const bool use_depth = tree_param.is_use_depth_;
    if (use_depth) {
        printf("use depth in the feature.\n");
    }
    else {
        printf("not use depth in the feature.\n");
    }
    
    dataset_param.printSelf();
    tree_param.printSelf();
    
    cv::Mat camera_matrix = dataset_param.camera_matrix();
    const int wh_kernel_size = tree_param.wh_kernel_size_;
    const bool is_use_depth = tree_param.is_use_depth_;
    
    using FeatureType = SCRFRandomSample;
    // read images, and predict one by one
    for (int k = 0; k<rgb_files.size(); k++) {
        const char *cur_rgb_img_file     = rgb_files[k].c_str();
        const char *cur_depth_img_file   = depth_files[k].c_str();
        const char *cur_pose_file        = pose_files[k].c_str();
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(cur_rgb_img_file, rgb_img);
        vector<FeatureType>     features;
        vector<Eigen::VectorXf> labels;
        BTRNDUtil::randomSampleFromRgbdImages(cur_rgb_img_file, cur_depth_img_file, cur_pose_file,
                                              num_random_sample, k, dataset_param,
                                              is_use_depth, false,
                                              features, labels);
        BTRNDUtil::extractWHFeatureFromRgbImages(cur_rgb_img_file, features, wh_kernel_size, false);
        assert(features.size() == labels.size());
        
        // predict from the model
        vector<vector<Eigen::VectorXf> > all_predictions;
        vector<vector<float> > all_distances;
        vector<Eigen::VectorXf> all_labels;    // labels
        vector<Eigen::Vector2f> all_locations; // 2d location
        
        for (int j = 0; j<features.size(); j++) {
            vector<Eigen::VectorXf> preds;
            vector<float> dists;
            bool is_predict = model.predict(features[j], rgb_img, max_check, preds, dists);
            if (is_predict) {
                all_predictions.push_back(preds);
                all_distances.push_back(dists);
                all_labels.push_back(labels[j]);
                all_locations.push_back(features[j].p2d_);
            }
        }
        // check prediction quality, only use the smallest one
        const double threshold = 0.1;
        int inlier_num = 0;
        for (int i = 0; i<all_predictions.size(); i++) {
            float dis = (all_labels[i] - all_predictions[i][0]).norm();
            if (dis < threshold) {
                inlier_num++;
            }
        }
        printf("optimal: inlier percentage %f, threshold %f distance\n", 1.0 * inlier_num/all_predictions.size(), threshold);
        
        {
            char save_file[1024] = {NULL};
            sprintf(save_file, "%s_%06d.txt", prefix, k);
            FILE *pf = fopen(save_file, "w");
            assert(pf);
            fprintf(pf, "%s\n", cur_rgb_img_file);
            fprintf(pf, "%s\n", cur_depth_img_file);
            fprintf(pf, "%s\n", cur_pose_file);
            fprintf(pf, "imageLocation\t  groundTruth3d \t num_pred \t, pred_3d, feature_distance \n");
            for (int i = 0; i<all_predictions.size(); i++) {
                Eigen::Vector2f p2d = all_locations[i];
                Eigen::VectorXf p3d = all_labels[i];
                assert(p3d.size() == 3);
                const int num = (int)all_predictions[i].size();
                fprintf(pf, "%d %d\t %lf %lf %lf\n", (int)p2d[0], (int)p2d[1], p3d[0], p3d[1], p3d[2]);
                fprintf(pf, "%d \n", num);
                for (int j = 0; j<all_predictions[i].size(); j++) {
                    Eigen::VectorXf pred   = all_predictions[i][j];
                    assert(pred.size() == 3);
                    float   dist = all_distances[i][j];
                    fprintf(pf, "\t %lf %lf %lf\t %lf\n", pred[0], pred[1], pred[2], dist);
                }
            }
            fclose(pf);
            printf("save to %s\n", save_file);
        }
    }
     
    
    return 0;
}

#endif
