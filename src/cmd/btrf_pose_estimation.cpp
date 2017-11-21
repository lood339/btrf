//  Created by jimmy on 2016-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include "cvx_image_310.hpp"
#include "cvx_io.hpp"
#include "cvx_pose_estimation.hpp"
#include "ms7scenes_util.hpp"
#include "dataset_param.hpp"
#include "cvx_calib3d.hpp"

using namespace::std;

#if 0

static void help()
{
    printf("program    datasetFile   predictions  sampleNumber inlierThreshold rotation translation saveFilePrefix           \n");
    printf("BTRF_Pose  dataset.txt   result/*.txt 500          0.1             5        0.05        estimaged_poses/camera \n");
    printf("Estimate camera pose using preemptive RANSAC and Kabsch. \n");
    printf("Output: percenrage of current camera poses, and median rotation, translation errors.\n");
    printf("datasetFile: dataset parameter, for example focal length. \n");
    printf("predictions: predicted 3D location files\n");
    printf("sampleNumber: sample numbers in each RANSAC.\n");
    printf("inlierThreshold: 3D location reprojection error threshold, unit meter. A parameter in RANSAC.\n");
    printf("rotation: rotation error metric, unit degree   \n");
    printf("translation: translation error metric, unit meter \n");
    printf("saveFilePrefix: camera pose error. (degree, meter). \n");
    
}

int main(int argc, const char * argv[])
{
    if (argc != 8) {
        help();
        printf("parameter number is %d, should be 8.\n", argc);
        return -1;
    }
    
    const char * dataset_param_filename = argv[1];
    const char *prediction_folder        = argv[2];
    const int sample_num = (int)strtod(argv[3], NULL);
    const double inlier_threshold = strtod(argv[4], NULL);
    const double angle_threshold    = strtod(argv[5], NULL);
    const double distance_threshold = strtod(argv[6], NULL);
    const char *prefix = argv[7];
    
    // read dataset parameter
    DatasetParameter dataset_param;
    dataset_param.readFromFileDataParameter(dataset_param_filename);
 
    // read prediction files
    vector<string> files = CvxIO::read_files(prediction_folder);
    assert(files.size() > 0);
    
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    const double depth_factor = dataset_param.depth_factor_;
    const double min_depth = dataset_param.min_depth_;
    const double max_depth = dataset_param.max_depth_;
    
    vector<double> angle_errors;
    vector<double> translation_errors;
    
    // save to files
    vector<cv::Mat> estimated_poses;
    vector<string> rgb_img_files;
    vector<string> depth_img_files;
    vector<string> camera_pose_files;
    for(int k = 0; k < files.size(); k++) {
        string cur_file = files[k];
        string rgb_img_file, depth_img_file, camera_pose_file;
        
        // Step 1: load predictions
        vector<cv::Point2d> img_pts;
        vector<cv::Point3d> wld_pts_gt;
        vector< vector<cv::Point3d> > wld_pts_pred_candidate;
        vector< vector<double > > candidate_feature_dists;
        bool is_read = Ms7ScenesUtil::load_prediction_result_with_distance(cur_file.c_str(),
                                                                           rgb_img_file,
                                                                           depth_img_file,
                                                                           camera_pose_file,
                                                                           img_pts, wld_pts_gt,
                                                                           wld_pts_pred_candidate,
                                                                           candidate_feature_dists);
        
        assert(is_read);
        rgb_img_files.push_back(rgb_img_file);
        depth_img_files.push_back(depth_img_file);
        camera_pose_files.push_back(camera_pose_file);
        
       
        // Step 2: load depth image and ground truth camera pose (for comparison only)
        cv::Mat depth_img;
        CvxIO::imread_depth_16bit_to_64f(depth_img_file.c_str(), depth_img);
        cv::Mat camera_to_world_pose = Ms7ScenesUtil::read_pose_7_scenes(camera_pose_file.c_str());
        
        cv::Mat mask;
        cv::Mat camera_coordinate_position;
        CvxCalib3D::cameraDepthToWorldCoordinate(depth_img,
                                                            camera_to_world_pose,
                                                            calibration_matrix,
                                                            depth_factor,
                                                            min_depth,
                                                            max_depth,
                                                            camera_coordinate_position,
                                                            mask);
        
        // 2D pixel location to 3D camera coordiante location
        vector<vector<cv::Point3d> > valid_wld_pts_candidate;
        vector<cv::Point3d> valid_camera_pts;
        for(int i = 0; i<img_pts.size(); i++) {
            int x = img_pts[i].x;
            int y = img_pts[i].y;
            if(mask.at<unsigned char>(y, x) != 0) {
                cv::Point3d p = cv::Point3d(camera_coordinate_position.at<cv::Vec3d>(y, x));
                valid_camera_pts.push_back(p);
                valid_wld_pts_candidate.push_back(wld_pts_pred_candidate[i]);
            }
        }
        
        cv::Mat estimated_camera_pose = cv::Mat::eye(4, 4, CV_64F);
        // Too few predictions
        if (valid_camera_pts.size() < 20) {
            angle_errors.push_back(180.0);
            translation_errors.push_back(10.0);
            estimated_poses.push_back(estimated_camera_pose);
            continue;
        }
        
        // Step 3: estimate camera pose using Kabsch
        PreemptiveRANSAC3DParameter param;
        param.dis_threshold_ = inlier_threshold;
        param.sample_number_ = sample_num;
        bool isEstimated = CvxPoseEstimation::preemptiveRANSAC3DOneToMany(valid_camera_pts,
                                                                          valid_wld_pts_candidate,
                                                                          param,
                                                                          estimated_camera_pose);
        if (isEstimated) {
            // measure rotation and translation errors
            double rotation_error = 0.0;
            double translation_error = 0.0;
            cv::Mat gt_pose = Ms7ScenesUtil::read_pose_7_scenes(camera_pose_file.c_str());
            CvxPoseEstimation::poseDistance(gt_pose, estimated_camera_pose, rotation_error, translation_error);
            angle_errors.push_back(rotation_error);
            translation_errors.push_back(translation_error);
            printf("angle distance, location distance are %lf %lf\n", rotation_error, translation_error);
        }
        else
        {
            // arbitrary large errors
            angle_errors.push_back(180.0);
            translation_errors.push_back(10.0);
        }
        estimated_poses.push_back(estimated_camera_pose);
        
        if (k % 10 == 0) {
            // number of cameras inside threshold
            int num_small_error_cameras = 0;
            for (int i = 0; i<angle_errors.size(); i++) {
                if (angle_errors[i] < angle_threshold && translation_errors[i] < distance_threshold) {
                    num_small_error_cameras++;
                }
            }
            printf("--------------------------camera number %lu, correct pose percentage is %lf, threshold(%lf %lf)-------------------\n",
                   angle_errors.size(),
                   1.0 * num_small_error_cameras/angle_errors.size(),
                   angle_threshold,
                   distance_threshold);
        }
    }
    assert(angle_errors.size() == translation_errors.size());
    assert(angle_errors.size() == files.size());
    
    // statistic of prediction error
    // number of correct cameras
    int num_correct_camera = 0;
    for (int i = 0; i<angle_errors.size(); i++) {
        if (angle_errors[i] < angle_threshold && translation_errors[i] < distance_threshold) {
            num_correct_camera++;
        }
    }
    printf("correct pose estimation percentage is %lf, threshold(%lf %lf)\n",
           1.0 * num_correct_camera/angle_errors.size(),
           angle_threshold, distance_threshold);
    
    std::sort(angle_errors.begin(), angle_errors.end());
    std::sort(translation_errors.begin(), translation_errors.end());
    printf("median angle error: %lf, translation error: %lf\n",
           angle_errors[angle_errors.size()/2],
           translation_errors[translation_errors.size()/2]);
    
    assert(estimated_poses.size() == files.size());
    assert(estimated_poses.size() == rgb_img_files.size());
    assert(estimated_poses.size() == depth_img_files.size());
    assert(estimated_poses.size() == camera_pose_files.size());
    
    // save predicted cameras
    for (int k = 0; k<estimated_poses.size(); k++) {
        char save_file[1024] = {NULL};
        sprintf(save_file, "%s_%06d.txt", prefix, k);
        FILE *pf = fopen(save_file, "w");
        assert(pf);
        fprintf(pf, "%s\n", rgb_img_files[k].c_str());
        fprintf(pf, "%s\n", depth_img_files[k].c_str());
        fprintf(pf, "%s\n", camera_pose_files[k].c_str());
        Mat pose = estimated_poses[k];
        for (int r = 0; r<4; r++) {
            for (int c = 0; c<4; c++) {
                fprintf(pf, "%lf\t", pose.at<double>(r, c));
            }
            fprintf(pf, "\n");
        }
        fclose(pf);
    }
    printf("save to %s\n", prefix);
    
    return 0;
}

#endif
