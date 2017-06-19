#include "ms7scenes_util.hpp"
#include <iostream>
#include <opencv2/core/eigen.hpp>

using cv::Mat;
using std::cout;
using std::endl;

Mat Ms7ScenesUtil::read_pose_7_scenes(const char *file_name)
{
    Mat P = Mat::zeros(4, 4, CV_64F);
    FILE *pf = fopen(file_name, "r");
    if(pf == NULL)
    {
        cout<<file_name<<endl;
    }
    assert(pf);
    for (int row = 0; row<4; row++) {
        for (int col = 0; col<4; col++) {
            double v = 0;
            fscanf(pf, "%lf", &v);
            P.at<double>(row, col) = v;
        }
    }
    fclose(pf);   
    return P;
}


bool Ms7ScenesUtil::load_prediction_result_with_distance(const char *file_name,
                                                         string & rgb_img_file,
                                                         string & depth_img_file,
                                                         string & camera_pose_file,
                                                         vector<cv::Point2d> & img_pts,
                                                         vector<cv::Point3d> & wld_pts_gt,
                                                         vector<vector<cv::Point3d> > & candidate_wld_pts_pred,
                                                         vector<vector<double> > & candidate_feature_dists)
                                         
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        rgb_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        depth_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s\n", buf);   // remove the last \n
        camera_pose_file = string(buf);
    }
    
    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        printf("%s\n", dummy_buf);
    }
    
    while (1) {
        double val[5] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4]);
        if (ret != 5) {
            break;
        }
        img_pts.push_back(cv::Point2d(val[0], val[1]));
        wld_pts_gt.push_back(cv::Point3d(val[2], val[3], val[4]));
       
        
        int num = 0;
        ret = fscanf(pf, "%d", &num);
        assert(ret == 1);
        
        vector<cv::Point3d> wld_pts_pred;
        vector<double> feat_dists;
        for (int i = 0; i<num; i++) {
            ret = fscanf(pf, "%lf %lf %lf %lf", &val[0], &val[1], &val[2], &val[3]);
            assert(ret == 4);
            wld_pts_pred.push_back(cv::Point3d(val[0], val[1], val[2]));
            feat_dists.push_back(val[3]);
        }
        candidate_wld_pts_pred.push_back(wld_pts_pred);
        candidate_feature_dists.push_back(feat_dists);
    }
    fclose(pf);
    
    assert(img_pts.size() == wld_pts_gt.size());
    assert(img_pts.size() == candidate_wld_pts_pred.size());
    assert(img_pts.size() == candidate_feature_dists.size());
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    return true;
}



