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

#if 0

static void help()
{
    printf("program   modelFile  RGBImageList depthImageList cameraPoseList numSample maxCheck saveFilePrefix\n");
    printf("BTRF_test model.txt  rgbs.txt     depth.txt      poses.txt      5000      16       to3d \n");
    printf("This the random forest prediction (testing) program.\nInput a trained model and testing images, output predicted 3D locations.\n");
    printf("This is the intermedia output. The camera pose is estimated from this output.\n");
    printf("model.txt: trained random forest model.\n");
    printf("RGBImageList  : a list of RGB images\n");
    printf("depthImageList: a list of depth images\n");
    printf("cameraPoseList: a list of camera pose (ground truth) file. The data in this file is used as a reference, not use in prediction\n");
    printf("numSample: number of sampled pixels in an image\n");
    printf("maxCheck: back tracking number. Larger number gives more accurate prediction with slower speed.\n");
    printf("saveFilePrefix: save file prefix. For example, predictions/to3d ");
    printf("The program gives multiple 3D predictions which are ordered by feature distances\n");
}

int main(int argc, const char * argv[])
{
    if (argc != 8) {
        printf("argc is %d, should be %d \n", argc-1, 8-1);
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
    assert(num_random_sample > 100);  // magic number
     
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
    
    const BTRFTreeParameter & tree_param = model.getTreeParameter();
    const DatasetParameter  & dataset_param = model.getDatasetParameter();
    
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
        
        // read image and extract WHT feature
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(cur_rgb_img_file, rgb_img);
        vector<FeatureType>     features;
        vector<Eigen::VectorXf> labels;
        BTRFUtil::randomSampleFromRgbdImages(cur_rgb_img_file, cur_depth_img_file, cur_pose_file,
                                             num_random_sample, k, dataset_param,
                                             is_use_depth, false,
                                             features, labels);
        BTRFUtil::extractWHFeatureFromRgbImages(cur_rgb_img_file, features, wh_kernel_size, false);
        assert(features.size() == labels.size());
        
        // predict from the model
        vector<vector<Eigen::VectorXf> > all_predictions;  // predicted world location
        vector<vector<float> >  all_distances; // feature distance
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
        // measure prediction quality
        const double threshold = 0.1;  // meter
        int inlier_num = 0;
        for (int i = 0; i<all_predictions.size(); i++) {
            float dis = (all_labels[i] - all_predictions[i][0]).norm();
            if (dis < threshold) {
                inlier_num++;
            }
        }
        printf("optimal: inlier percentage %f, threshold %f distance\n", 1.0 * inlier_num/all_predictions.size(), threshold);
        
        {
            // save prediciton to a .txt file
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
