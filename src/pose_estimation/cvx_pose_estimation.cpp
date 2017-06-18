//
//  cvx_pose_estimation.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-31.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_pose_estimation.hpp"
#include <iostream>
#include <Eigen/Geometry>
#include "cvx_calib3d.hpp"

using std::cout;
using std::endl;
using cv::Mat;


bool CvxPoseEstimation::estimateCameraPose(const cv::Mat & camera_matrix,
                                           const cv::Mat & dist_coeff,
                                           const vector<cv::Point2d> & im_pts,
                                           const vector<cv::Point3d> & wld_pts,
                                           cv::Mat & camera_pose,
                                           const double outlier_threshold)
{
    assert(im_pts.size() == wld_pts.size());
    
    Mat rvec;
    Mat tvec;
    bool is_solved = cv::solvePnPRansac(Mat(wld_pts), Mat(im_pts), camera_matrix, Mat(), rvec, tvec, false, 1000, outlier_threshold);
//    bool is_solved = cv::solvePnP(Mat(wld_pts), Mat(im_pts), camera_matrix, dist_coeff, rvec, tvec, false, CV_EPNP);
    if (!is_solved) {
        printf("warning: solver PnP failed.\n");
        return false;
    }
    
    if (0)
    {
        // test re-projection error
        vector<cv::Point2d> projected_pts;
        int num = 0;
        cv::projectPoints(Mat(wld_pts), rvec, tvec, camera_matrix, Mat(), projected_pts);
        assert(im_pts.size() == projected_pts.size());
        for (int i = 0; i<projected_pts.size(); i++) {
            double error_reproj = cv::norm(im_pts[i] - projected_pts[i]);
            
            if (error_reproj > 10) {
            //    printf("reprojection error are %lf\n", error_reproj);
           //     cout<<"correct position   "<<im_pts[i]<<endl;
           //     cout<<"projected position "<<projected_pts[i]<<endl<<endl;
                num++;
            }
        }
        printf("bad projection (reprojection error > 10) number is %d, percentage %lf.\n", num, 1.0*num/projected_pts.size());
    }
    
    Mat rot;
    cv::Rodrigues(rvec, rot);
    assert(rot.type()  == CV_64F);
    assert(tvec.type() == CV_64F);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    for (int j = 0; j<3; j++) {
        for (int i = 0; i<3; i++) {
            camera_pose.at<double>(i, j) = rot.at<double>(i, j);
        }
    }
    camera_pose.at<double>(0, 3) = tvec.at<double>(0, 0);
    camera_pose.at<double>(1, 3) = tvec.at<double>(1, 0);
    camera_pose.at<double>(2, 3) = tvec.at<double>(2, 0);
    
    // camere to world coordinate
    camera_pose = camera_pose.inv();
    //cout<<"camera to world coordinate is "<<camera_pose<<endl;
    return true;
}

bool CvxPoseEstimation::estimateCameraPose(const vector<cv::Point3d> & camera_pts,
                                           const vector<cv::Point3d> & wld_pts,
                                           cv::Mat & camera_pose)
{
    assert(camera_pts.size() == wld_pts.size());
    
    Mat affine;
    Mat inlier;
    bool is_solved = cv::estimateAffine3D(Mat(camera_pts), Mat(wld_pts), affine, inlier, 0.9);
    if (is_solved) {
        camera_pose = cv::Mat::eye(4, 4, CV_64F);
        affine.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    }
    return is_solved;
}


struct HypotheseLoss
{
    double loss_;
    Mat rvec_;       // rotation     for 3D --> 2D projection
    Mat tvec_;       // translation  for 3D --> 2D projection
    Mat affine_;     //              for 3D --> 3D camera to world transformation
    vector<unsigned int> inlier_indices_;         // camera coordinate index
    vector<unsigned int> inlier_candidate_world_pts_indices_; // candidate world point index
    
    // store all inliers from preemptive ransac
    vector<cv::Point3d> camera_pts_;
    vector<cv::Point3d> wld_pts_;
    
    HypotheseLoss()
    {
        loss_ = INT_MAX;
    }
    HypotheseLoss(const double loss)
    {
        loss_  = loss;
    }
    
    HypotheseLoss(const HypotheseLoss & other)
    {
        loss_ = other.loss_;
        rvec_ = other.rvec_;
        tvec_ = other.tvec_;
        affine_ = other.affine_;
        inlier_indices_.clear();
        inlier_indices_.resize(other.inlier_indices_.size());
        inlier_candidate_world_pts_indices_.clear();
        inlier_candidate_world_pts_indices_.resize(other.inlier_candidate_world_pts_indices_.size());
        for(int i = 0; i<other.inlier_indices_.size(); i++) {
            inlier_indices_[i] = other.inlier_indices_[i];
        }
        for(int i = 0; i<other.inlier_candidate_world_pts_indices_.size(); i++){
            inlier_candidate_world_pts_indices_[i] = other.inlier_candidate_world_pts_indices_[i];
        }
        if(inlier_candidate_world_pts_indices_.size() != 0){
            assert(inlier_indices_.size() == inlier_candidate_world_pts_indices_.size());
        }
        
        // copy camera points and world coordinate points
        if(other.camera_pts_.size() > 0)
        {
            camera_pts_.resize(other.camera_pts_.size());
            wld_pts_.resize(other.wld_pts_.size());
            
            assert(camera_pts_.size() == wld_pts_.size());
            for(int i = 0; i<camera_pts_.size(); i++)
            {
                camera_pts_[i] = other.camera_pts_[i];
                wld_pts_[i] = other.wld_pts_[i];
            }
        }
    }    
    
    bool operator < (const HypotheseLoss & other) const
    {
        return loss_ < other.loss_;
    }
    
    HypotheseLoss & operator = (const HypotheseLoss & other)
    {
        if (&other == this) {
            return *this;
        }
        loss_ = other.loss_;
        rvec_ = other.rvec_;
        tvec_ = other.tvec_;
        affine_ = other.affine_;
        inlier_indices_.clear();
        inlier_indices_.resize(other.inlier_indices_.size());
        inlier_candidate_world_pts_indices_.clear();
        inlier_candidate_world_pts_indices_.resize(other.inlier_candidate_world_pts_indices_.size());
        for(int i = 0; i<other.inlier_indices_.size(); i++) {
            inlier_indices_[i] = other.inlier_indices_[i];
        }
        for(int i = 0; i<other.inlier_candidate_world_pts_indices_.size(); i++){
            inlier_candidate_world_pts_indices_[i] = other.inlier_candidate_world_pts_indices_[i];
        }
        if(inlier_candidate_world_pts_indices_.size() != 0){
            assert(inlier_indices_.size() == inlier_candidate_world_pts_indices_.size());
        }
        
        // copy camera points and world coordinate points
        if(other.camera_pts_.size() > 0)
        {
            camera_pts_.resize(other.camera_pts_.size());
            wld_pts_.resize(other.wld_pts_.size());
            
            assert(camera_pts_.size() == wld_pts_.size());
            for(int i = 0; i<camera_pts_.size(); i++)
            {
                camera_pts_[i] = other.camera_pts_[i];
                wld_pts_[i] = other.wld_pts_[i];
            }
        }
        
        return *this;
    }
};

bool CvxPoseEstimation::preemptiveRANSAC(const vector<cv::Point2d> & img_pts,
                                         const vector<cv::Point3d> & wld_pts,
                                         const cv::Mat & camera_matrix,
                                         const cv::Mat & dist_coeff,
                                         const PreemptiveRANSACParameter & param,
                                         cv::Mat & camera_pose)
{
    assert(img_pts.size() == wld_pts.size());
    assert(img_pts.size() > 500);
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)img_pts.size();
    const int B = 500;
    
    vector<std::pair<Mat, Mat> > rt_candidate;
    for (int i = 0; i<num_iteration; i++) {
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point2d> sampled_img_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_img_pts.push_back(img_pts[k1]);
        sampled_img_pts.push_back(img_pts[k2]);
        sampled_img_pts.push_back(img_pts[k3]);
        sampled_img_pts.push_back(img_pts[k4]);
        
        sampled_wld_pts.push_back(wld_pts[k1]);
        sampled_wld_pts.push_back(wld_pts[k2]);
        sampled_wld_pts.push_back(wld_pts[k3]);
        sampled_wld_pts.push_back(wld_pts[k4]);
        
        Mat rvec;
        Mat tvec;
        bool is_solved = cv::solvePnP(Mat(sampled_wld_pts), Mat(sampled_img_pts), camera_matrix, dist_coeff, rvec, tvec, false, CV_P3P);
        if (is_solved) {
            rt_candidate.push_back(std::make_pair(rvec, tvec));
        }
        if (rt_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", rt_candidate.size());
    
    K = (int)rt_candidate.size();
    
    
    vector<HypotheseLoss> losses;
    for (int i = 0; i<rt_candidate.size(); i++) {
        HypotheseLoss hyp(0.0);
        hyp.rvec_ = rt_candidate[i].first;
        hyp.tvec_ = rt_candidate[i].second;
        losses.push_back(hyp);
    }
    
    double reproj_threshold = param.reproj_threshold;
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point2d> sampled_img_pts;
        vector<cv::Point3d> sampled_wld_pts;
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_img_pts.push_back(img_pts[index]);
            sampled_wld_pts.push_back(wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check reprojection error
            vector<cv::Point2d> projected_pts;
            cv::projectPoints(sampled_wld_pts, losses[i].rvec_, losses[i].tvec_, camera_matrix, dist_coeff, projected_pts);
            
            for (int j = 0; j<projected_pts.size(); j++) {
                cv::Point2d dif = projected_pts[j] - sampled_img_pts[j];
                double dis = cv::norm(dif);
                if (dis > reproj_threshold) {
                    losses[i].loss_ += 1.0;
                }
                else  {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                }
            }
        }
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point2d> inlier_img_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    inlier_img_pts.push_back(img_pts[index]);
                    inlier_wld_pts.push_back(wld_pts[index]);
                }
                
                Mat rvec = losses[i].rvec_;
                Mat tvec = losses[i].tvec_;
                bool is_solved = cv::solvePnP(Mat(inlier_wld_pts), Mat(inlier_img_pts), camera_matrix, dist_coeff, rvec, tvec, true, CV_EPNP);  // CV_ITERATIVE   CV_EPNP
                if (is_solved) {
                     losses[i].inlier_indices_.clear();
                    losses[i].rvec_ = rvec;
                    losses[i].tvec_ = tvec;
                }
            }
        }        
    }
    
    assert(losses.size() == 1);
    
    // change to camera to world transformation
    Mat rot;
    cv::Rodrigues(losses.front().rvec_, rot);
    Mat tvec = losses.front().tvec_;
    assert(tvec.rows == 3);
    assert(tvec.type() == CV_64FC1);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    for (int j = 0; j<3; j++) {
        for (int i = 0; i<3; i++) {
            camera_pose.at<double>(i, j) = rot.at<double>(i, j);
        }
    }
    camera_pose.at<double>(0, 3) = tvec.at<double>(0, 0);
    camera_pose.at<double>(1, 3) = tvec.at<double>(1, 0);
    camera_pose.at<double>(2, 3) = tvec.at<double>(2, 0);
    
    // camere to world coordinate
    camera_pose = camera_pose.inv();
    
    return true;
}

bool CvxPoseEstimation::preemptiveRANSAC2DOneToMany(const vector<cv::Point2d> & img_pts,
                                                    const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                                    const cv::Mat & camera_matrix,
                                                    const cv::Mat & dist_coeff,
                                                    const PreemptiveRANSACParameter & param,
                                                    cv::Mat & camera_pose,
                                                    cv::Mat & camera_rvec,
                                                    cv::Mat & camera_tvec)
{
    assert(img_pts.size() == candidate_wld_pts.size());
    if (img_pts.size() < 500) {
        return false;
    }
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)img_pts.size();
    int B = 500;
    if (img_pts.size() < 1000) {
        B = 300;
    }
    
    vector<std::pair<Mat, Mat> > rt_candidate;
    for (int i = 0; i<num_iteration; i++) {
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point2d> sampled_img_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_img_pts.push_back(img_pts[k1]);
        sampled_img_pts.push_back(img_pts[k2]);
        sampled_img_pts.push_back(img_pts[k3]);
        sampled_img_pts.push_back(img_pts[k4]);
        
        sampled_wld_pts.push_back(candidate_wld_pts[k1][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k2][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k3][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k4][0]);
        
        
        Mat rvec;
        Mat tvec;
        bool is_solved = cv::solvePnP(Mat(sampled_wld_pts), Mat(sampled_img_pts), camera_matrix, dist_coeff, rvec, tvec, false, CV_EPNP);
        if (is_solved) {
            rt_candidate.push_back(std::make_pair(rvec, tvec));
        }
        
        if (rt_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", rt_candidate.size());
    
    vector<HypotheseLoss> losses;
    for (int i = 0; i<rt_candidate.size(); i++) {
        HypotheseLoss hyp(0.0);
        hyp.rvec_ = rt_candidate[i].first;
        hyp.tvec_ = rt_candidate[i].second;
        losses.push_back(hyp);
    }
    
    double threshold = param.reproj_threshold;    
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point2d> sampled_img_pts;
        vector< vector<cv::Point3d> > sampled_wld_pts;  // one camera point may have multiple world points correspondences
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_img_pts.push_back(img_pts[index]);
            sampled_wld_pts.push_back(candidate_wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check re-projection error
            vector<vector<cv::Point2d> > all_projected_pts;
            Mat rvec = losses[i].rvec_;
            Mat tvec = losses[i].tvec_;
            // project all world points to image using estimated rotation and translation vector
            for (int j = 0; j<sampled_wld_pts.size(); j++) {
                vector<cv::Point2d> projected_pts;
                cv::projectPoints(sampled_wld_pts[j], rvec, tvec, camera_matrix, dist_coeff, projected_pts);
                all_projected_pts.push_back(projected_pts);
            }
            assert(all_projected_pts.size() == sampled_img_pts.size());
            
            // check reprojection error
            for (int j = 0; j<sampled_img_pts.size(); j++) {
                double min_dis = threshold * 2;
                int min_index = -1;
                cv::Point2d img_pt = sampled_img_pts[j];
                for (int k = 0; k<all_projected_pts[j].size(); k++) {
                    cv::Point2d dif = img_pt - all_projected_pts[j][k];
                    double dis = cv::norm(dif);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_index = k;
                    }
                } // end of k
                
                if (min_dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                    losses[i].inlier_candidate_world_pts_indices_.push_back(min_index);
                }
            } // end of j
            assert(losses[i].inlier_indices_.size() == losses[i].inlier_candidate_world_pts_indices_.size());
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        //getchar();
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        for (int i = 0; i<losses.size(); i++) {
        //    printf("after: loss is %lf\n", losses[i].loss_);
        //    printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        // printf("\n\n");
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point2d> inlier_img_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    int wld_index = losses[i].inlier_candidate_world_pts_indices_[j];
                    inlier_img_pts.push_back(img_pts[index]);
                    inlier_wld_pts.push_back(candidate_wld_pts[index][wld_index]);
                }
                
                Mat rvec;
                Mat tvec;
                bool is_solved = cv::solvePnP(Mat(inlier_wld_pts), Mat(inlier_img_pts), camera_matrix, dist_coeff, rvec, tvec, false, CV_EPNP);
                if (is_solved) {
                    losses[i].rvec_ = rvec;
                    losses[i].tvec_ = tvec;
                    losses[i].inlier_indices_.clear();
                    losses[i].inlier_candidate_world_pts_indices_.clear();
                }
            }
        }
    }
    assert(losses.size() == 1);
    
    // change to camera to world transformation
    Mat rot;
    cv::Rodrigues(losses.front().rvec_, rot);
    Mat tvec = losses.front().tvec_;
    assert(tvec.rows == 3);
    assert(tvec.type() == CV_64FC1);
    assert(rot.type() == CV_64FC1);
    assert(rot.rows == 3 && rot.cols == 3);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    rot.copyTo(camera_pose(cv::Rect(0, 0, 3, 3)));
    
    camera_pose.at<double>(0, 3) = tvec.at<double>(0, 0);
    camera_pose.at<double>(1, 3) = tvec.at<double>(1, 0);
    camera_pose.at<double>(2, 3) = tvec.at<double>(2, 0);
    
    // camere to world coordinate
    camera_pose = camera_pose.inv();
    camera_rvec = losses.front().rvec_;
    camera_tvec = losses.front().tvec_;
    
    if (isnan(camera_pose.at<double>(0, 0))) {
        return false;
    }
    
    return true;
}


double CvxPoseEstimation::minCameraDistanceUnderAngularThreshold(const vector<cv::Mat> & database_camera_poses,
                                                     const cv::Mat & query_pose,
                                                     const double angular_threshold)
{
    assert(database_camera_poses.size() > 0);
    vector<double> distance;    
    
    for(int i=0; i<database_camera_poses.size();i++)
    {
        double rot_dis = 0.0;
        double trans_dis = 0.0;        
        CvxPoseEstimation::poseDistance(query_pose,
                                        database_camera_poses[i],
                                        rot_dis,
                                        trans_dis);
        if(rot_dis<angular_threshold) {
            distance.push_back(trans_dis);
        }
    }
  
    if (distance.size() == 0) {
        return INT_MAX;
    }
    else {
        return *std::min_element(distance.begin(), distance.end());
    }
}


double CvxPoseEstimation::minCameraAngleUnderTranslationalThreshold(const vector<cv::Mat> & database_camera_poses,
                                                                    const cv::Mat & query_pose,
                                                                    const double translation_threshold)
{
    assert(database_camera_poses.size() > 0);
    vector<double> rot_distance;
    for(int i=0; i<database_camera_poses.size();i++)
    {
        double rot_dis = 0.0;
        double trans_dis = 0.0;        
        CvxPoseEstimation::poseDistance(query_pose,
                                        database_camera_poses[i],
                                        rot_dis,
                                        trans_dis);
        if(trans_dis<translation_threshold)
        {
            rot_distance.push_back(rot_dis);
        }
    }
    if (rot_distance.size() == 0) {
        return INT_MAX;
    }
    else {
        return *std::min_element(rot_distance.begin(), rot_distance.end());
    }
}

bool CvxPoseEstimation::preemptiveRANSAC3D(const vector<cv::Point3d> & camera_pts,
                                           const vector<cv::Point3d> & wld_pts,
                                           const PreemptiveRANSAC3DParameter & param,
                                           cv::Mat & camera_pose)
{
    assert(camera_pts.size() == wld_pts.size());
    if (camera_pts.size() < 500) {
        return false;
    }
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)camera_pts.size();
    const int B = 500;
    
    vector<cv::Mat > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_camera_pts.push_back(camera_pts[k1]);
        sampled_camera_pts.push_back(camera_pts[k2]);
        sampled_camera_pts.push_back(camera_pts[k3]);
        sampled_camera_pts.push_back(camera_pts[k4]);
        
        sampled_wld_pts.push_back(wld_pts[k1]);
        sampled_wld_pts.push_back(wld_pts[k2]);
        sampled_wld_pts.push_back(wld_pts[k3]);
        sampled_wld_pts.push_back(wld_pts[k4]);
        
        Mat affine;
        Mat inlier;
        //bool is_solved = cv::estimateAffine3D(Mat(sampled_camera_pts), Mat(sampled_wld_pts), affine, inlier, 0.9);
        CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts, affine);
        bool is_solved = true;
        if (is_solved)
        {
            affine_candidate.push_back(affine);
        }
        if (affine_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", affine_candidate.size());
    
    vector<HypotheseLoss> losses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        HypotheseLoss hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        losses.push_back(hyp);
    }
    
    double threshold = param.dis_threshold_;
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_camera_pts.push_back(camera_pts[index]);
            sampled_wld_pts.push_back(wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check transformation
            vector<cv::Point3d> transformed_pts;
            CvxCalib3D::rigidTransform(sampled_camera_pts, losses[i].affine_, transformed_pts);
            
            // reset inlier index?
            
            //losses[i].inlier_indices_.clear();
            for (int j = 0; j<transformed_pts.size(); j++) {
                cv::Point3d dif = transformed_pts[j] - sampled_wld_pts[j];
                double dis = cv::norm(dif);
              //  cout<<" distance is "<<dis<<endl;
                if (dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else  {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                }
            } // end of j
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        //getchar();
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        for (int i = 0; i<losses.size(); i++) {
        //    printf("after: loss is %lf\n", losses[i].loss_);
        //    printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
       // printf("\n\n");
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point3d> inlier_camera_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    inlier_camera_pts.push_back(camera_pts[index]);
                    inlier_wld_pts.push_back(wld_pts[index]);
                }
                Mat affine;
                Mat inlier;
           //     bool is_solved = cv::estimateAffine3D(Mat(inlier_camera_pts), Mat(inlier_wld_pts), affine, inlier, 0.9);
                CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, affine);
                bool is_solved = true;
                if (is_solved) {
                    losses[i].affine_ = affine;
                    losses[i].inlier_indices_.clear();
                }
            }
        }
    }
    assert(losses.size() == 1);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    losses[0].affine_.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    //cout<<"camera pose\n"<<camera_pose<<endl;    
    return true;
}

bool CvxPoseEstimation::preemptiveRANSAC3DOneToMany(const vector<cv::Point3d> & camera_pts,
                                                    const vector<vector<cv::Point3d>> & candidate_wld_pts,
                                                    const PreemptiveRANSAC3DParameter & param,
                                                    cv::Mat & camera_pose)
{
    assert(camera_pts.size() == candidate_wld_pts.size());
    if (camera_pts.size() < 500) {
        return false;
    }
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)camera_pts.size();
    const int B = param.sample_number_;
    
    vector<cv::Mat > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_camera_pts.push_back(camera_pts[k1]);
        sampled_camera_pts.push_back(camera_pts[k2]);
        sampled_camera_pts.push_back(camera_pts[k3]);
        sampled_camera_pts.push_back(camera_pts[k4]);
        
        sampled_wld_pts.push_back(candidate_wld_pts[k1][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k2][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k3][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k4][0]);
        
        Mat affine;    
        CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts, affine);
        affine_candidate.push_back(affine);
        if (affine_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", affine_candidate.size());
    
    vector<HypotheseLoss> losses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        HypotheseLoss hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        losses.push_back(hyp);
    }
    
    double threshold = param.dis_threshold_;
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point3d> sampled_camera_pts;
        vector< vector<cv::Point3d> > sampled_wld_pts;  // one camera point may have multiple world points correspondences
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_camera_pts.push_back(camera_pts[index]);
            sampled_wld_pts.push_back(candidate_wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check transformation
            vector<cv::Point3d> transformed_pts;
            CvxCalib3D::rigidTransform(sampled_camera_pts, losses[i].affine_, transformed_pts);
            
            // check minimum distance from transformed points to world coordiante
            for (int j = 0; j<transformed_pts.size(); j++) {
                double min_dis = threshold * 2;
                int min_index = -1;
                for (int k = 0; k<sampled_wld_pts[j].size(); k++) {
                    cv::Point3d dif = transformed_pts[j] - sampled_wld_pts[j][k];
                    double dis = cv::norm(dif);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_index = k;
                    }
                } // end of k
                
                if (min_dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                    losses[i].inlier_candidate_world_pts_indices_.push_back(min_index);
                }
            } // end of j
            assert(losses[i].inlier_indices_.size() == losses[i].inlier_candidate_world_pts_indices_.size());
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        //getchar();
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        for (int i = 0; i<losses.size(); i++) {
         //   printf("after: loss is %lf\n", losses[i].loss_);
         //   printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        // printf("\n\n");
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point3d> inlier_camera_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    int wld_index = losses[i].inlier_candidate_world_pts_indices_[j];
                    inlier_camera_pts.push_back(camera_pts[index]);
                    inlier_wld_pts.push_back(candidate_wld_pts[index][wld_index]);
                }
                Mat affine;
                
                CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, affine);
                losses[i].affine_ = affine;
                losses[i].inlier_indices_.clear();
                losses[i].inlier_candidate_world_pts_indices_.clear();  
            }
        }
    }
    assert(losses.size() == 1);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    losses[0].affine_.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    //cout<<"camera pose\n"<<camera_pose<<endl;
    return true;
}

// refine camera pose using all candidate
static Mat refineCameraPose(const vector<cv::Point3d> & camera_pts,
                            const vector<vector<cv::Point3d> > & candidate_wld_pts,
                            const double threshold,
                            const cv::Mat & initial_affine,
                            const int intial_inlier_num)
{
    Mat refined_affine;
    
    // estimate from all inliers agree with cur_affine
    vector<cv::Point3d> transformed_camera_pts;
    CvxCalib3D::rigidTransform(camera_pts, initial_affine, transformed_camera_pts);
    
    vector<cv::Point3d> inlier_camera_pts;
    vector<cv::Point3d> inlier_wld_pts;
    for (int j = 0; j<transformed_camera_pts.size(); j++) {
        double min_dis = threshold * 2;
        int min_index = -1;
        for (int k = 0; k<candidate_wld_pts[j].size(); k++) {
            cv::Point3d dif = transformed_camera_pts[j] - candidate_wld_pts[j][k];
            double dis = cv::norm(dif);
            if (dis < min_dis) {
                min_dis = dis;
                min_index = k;
            }
        } // end of k
        
        if (min_dis < threshold) {
            inlier_camera_pts.push_back(camera_pts[j]);
            inlier_wld_pts.push_back(candidate_wld_pts[j][min_index]);
        }
        
    } // end of j
    
    if (inlier_camera_pts.size() >= 4 && inlier_camera_pts.size() > intial_inlier_num) {
        CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, refined_affine);
    }
    else {
        refined_affine = initial_affine;
    }
    
    return refined_affine;
}

bool CvxPoseEstimation::preemptiveRANSAC3DAllInliers(const vector<cv::Point3d> & camera_pts,
                                                     const vector<vector<cv::Point3d>> & candidate_wld_pts,
                                                     const PreemptiveRANSAC3DParameter & param,
                                                     cv::Mat & camera_pose)
{
    assert(camera_pts.size() == candidate_wld_pts.size());
    if (camera_pts.size() < 500) {
        return false;
    }
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)camera_pts.size();
    const int B = 500;
    
    vector<cv::Mat > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_camera_pts.push_back(camera_pts[k1]);
        sampled_camera_pts.push_back(camera_pts[k2]);
        sampled_camera_pts.push_back(camera_pts[k3]);
        sampled_camera_pts.push_back(camera_pts[k4]);
        
        sampled_wld_pts.push_back(candidate_wld_pts[k1][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k2][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k3][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k4][0]);
        
        Mat affine;
        Mat inlier;
        CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts, affine);
        affine_candidate.push_back(affine);
        if (affine_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", affine_candidate.size());
    
    vector<HypotheseLoss> losses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        HypotheseLoss hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        losses.push_back(hyp);
    }
    
    const double threshold = param.dis_threshold_;
    const int refine_camera_num = param.refine_camera_num_;
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point3d> sampled_camera_pts;
        vector< vector<cv::Point3d> > sampled_wld_pts;  // one camera point may have multiple world points correspondences
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_camera_pts.push_back(camera_pts[index]);
            sampled_wld_pts.push_back(candidate_wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check transformation
            vector<cv::Point3d> transformed_pts;
            CvxCalib3D::rigidTransform(sampled_camera_pts, losses[i].affine_, transformed_pts);
            
            // check minimum distance from transformed points to world coordiante
            for (int j = 0; j<transformed_pts.size(); j++) {
                double min_dis = threshold * 2;
                int min_index = -1;
                for (int k = 0; k<sampled_wld_pts[j].size(); k++) {
                    cv::Point3d dif = transformed_pts[j] - sampled_wld_pts[j][k];
                    double dis = cv::norm(dif);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_index = k;
                    }
                } // end of k
                
                if (min_dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                    losses[i].inlier_candidate_world_pts_indices_.push_back(min_index);
                }
            } // end of j
            assert(losses[i].inlier_indices_.size() == losses[i].inlier_candidate_world_pts_indices_.size());
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        //getchar();
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        for (int i = 0; i<losses.size(); i++) {
            //   printf("after: loss is %lf\n", losses[i].loss_);
            //   printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        // printf("\n\n");
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point3d> inlier_camera_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    int wld_index = losses[i].inlier_candidate_world_pts_indices_[j];
                    inlier_camera_pts.push_back(camera_pts[index]);
                    inlier_wld_pts.push_back(candidate_wld_pts[index][wld_index]);
                }
                
                // estimate current affine from new inliers
                Mat cur_affine;
                CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, cur_affine);
                
                if (losses.size() < refine_camera_num) {
                    Mat refined_affine = refineCameraPose(camera_pts, candidate_wld_pts, threshold, cur_affine, (int)inlier_camera_pts.size());
                    losses[i].affine_ = refined_affine;
                    losses[i].inlier_indices_.clear();
                    losses[i].inlier_candidate_world_pts_indices_.clear();                   
                }                
            }
        }
    }
    assert(losses.size() == 1);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    losses[0].affine_.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    //cout<<"camera pose\n"<<camera_pose<<endl;
    return true;
}


bool CvxPoseEstimation::preemptiveRANSAC3D(const vector<cv::Point3d> & camera_pts,
                                           const vector<cv::Point3d> & wld_pts,
                                           const PreemptiveRANSAC3DParameter & param,
                                           cv::Mat & camera_pose,
                                           vector<bool> & inliers)
{
    assert(camera_pts.size() == wld_pts.size());
    if (camera_pts.size() < 500) {
        return false;
    }
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)camera_pts.size();
    const int B = 500;
    
    vector<cv::Mat > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_camera_pts.push_back(camera_pts[k1]);
        sampled_camera_pts.push_back(camera_pts[k2]);
        sampled_camera_pts.push_back(camera_pts[k3]);
        sampled_camera_pts.push_back(camera_pts[k4]);
        
        sampled_wld_pts.push_back(wld_pts[k1]);
        sampled_wld_pts.push_back(wld_pts[k2]);
        sampled_wld_pts.push_back(wld_pts[k3]);
        sampled_wld_pts.push_back(wld_pts[k4]);
        
        Mat affine;
        Mat inlier;
        //bool is_solved = cv::estimateAffine3D(Mat(sampled_camera_pts), Mat(sampled_wld_pts), affine, inlier, 0.9);
        CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts, affine);
        bool is_solved = true;
        if (is_solved)
        {
            affine_candidate.push_back(affine);
        }
        if (affine_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", affine_candidate.size());
    
    vector<HypotheseLoss> losses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        HypotheseLoss hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        losses.push_back(hyp);
    }
    
    double threshold = param.dis_threshold_;
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_camera_pts.push_back(camera_pts[index]);
            sampled_wld_pts.push_back(wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check transformation
            vector<cv::Point3d> transformed_pts;
            CvxCalib3D::rigidTransform(sampled_camera_pts, losses[i].affine_, transformed_pts);
            
            // reset inlier index?
            
            //losses[i].inlier_indices_.clear();
            for (int j = 0; j<transformed_pts.size(); j++) {
                cv::Point3d dif = transformed_pts[j] - sampled_wld_pts[j];
                double dis = cv::norm(dif);
                //  cout<<" distance is "<<dis<<endl;
                if (dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else  {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                }
            } // end of j
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        //getchar();
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        for (int i = 0; i<losses.size(); i++) {
            //    printf("after: loss is %lf\n", losses[i].loss_);
            //    printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        // printf("\n\n");
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point3d> inlier_camera_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    inlier_camera_pts.push_back(camera_pts[index]);
                    inlier_wld_pts.push_back(wld_pts[index]);
                }
                Mat affine;
                Mat inlier;
                //     bool is_solved = cv::estimateAffine3D(Mat(inlier_camera_pts), Mat(inlier_wld_pts), affine, inlier, 0.9);
                CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, affine);
                bool is_solved = true;
                if (is_solved && losses.size() > 1) {
                    losses[i].affine_ = affine;
                    losses[i].inlier_indices_.clear();
                }
            }
        }
    }
    assert(losses.size() == 1);
    
    // inliers
    inliers = vector<bool>(camera_pts.size(), false);
    for (int i = 0; i<losses[0].inlier_indices_.size(); i++) {
        inliers[losses[0].inlier_indices_[i]] = true;
    }
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    losses[0].affine_.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    //cout<<"camera pose\n"<<camera_pose<<endl;
    
    return true;
}


Mat CvxPoseEstimation::rotationToEularAngle(const cv::Mat & rot)
{
    assert(rot.rows == 3 && rot.cols == 3);
    assert(rot.type() == CV_64FC1);
    
    // https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles.pdf
    double m00 = rot.at<double>(0, 0);
    double m01 = rot.at<double>(0, 1);
    double m02 = rot.at<double>(0, 2);
    double m10 = rot.at<double>(1, 0);
    double m11 = rot.at<double>(1, 1);
    double m12 = rot.at<double>(1, 2);
    double m20 = rot.at<double>(2, 0);
    double m21 = rot.at<double>(2, 1);
    double m22 = rot.at<double>(2, 2);
    double theta1 = atan2(m12, m22);
    double c2 = sqrt(m00 * m00 + m01 * m01);
    double theta2 = atan2(-m02, c2);
    double s1 = sin(theta1);
    double c1 = cos(theta1);
    double theta3 = atan2(s1*m20 - c1 * m10, c1*m11 - s1*m21);
    
    double scale = 180.0/3.14159;
    theta1 *= scale;
    theta2 *= scale;
    theta3 *= scale;
    
    Mat eular_angle = cv::Mat::zeros(3, 1, CV_64FC1);
    eular_angle.at<double>(0, 0) = theta1;
    eular_angle.at<double>(1, 0) = theta2;
    eular_angle.at<double>(2, 0) = theta3;
    return eular_angle;
    //printf("Eular angle %lf %lf %lf\n", theta1, theta2, theta3);    
}

void CvxPoseEstimation::poseDistance(const cv::Mat & src_pose,
                                     const cv::Mat & dst_pose,
                                     double & angle_distance,
                                     double & euclidean_disance)
{
    // http://chrischoy.github.io/research/measuring-rotation/
    assert(src_pose.type() == CV_64F);
    assert(dst_pose.type() == CV_64F);
    
    Mat src_R = src_pose(cv::Rect(0, 0, 3, 3));
    Mat dst_R = dst_pose(cv::Rect(0, 0, 3, 3));
    
    double scale = 180.0/3.14159;    
    
    Mat q1 = CvxPoseEstimation::rotationToQuaternion(src_R);
    Mat q2 = CvxPoseEstimation::rotationToQuaternion(dst_R);
    double val_dot = fabs(q1.dot(q2));

    //double dot = r1.dot(r2);
    //angle_distance = acos(dot) * scale;
    angle_distance = 2.0 * acos(val_dot) * scale;
    
    euclidean_disance = 0.0;
    double dx = src_pose.at<double>(0, 3) - dst_pose.at<double>(0, 3);
    double dy = src_pose.at<double>(1, 3) - dst_pose.at<double>(1, 3);
    double dz = src_pose.at<double>(2, 3) - dst_pose.at<double>(2, 3);
    euclidean_disance += dx * dx;
    euclidean_disance += dy * dy;
    euclidean_disance += dz * dz;
    euclidean_disance = sqrt(euclidean_disance);
 //   printf("location distance are %f %f %f\n", dx, dy, dz);
}


Mat CvxPoseEstimation::rotationToQuaternion(const cv::Mat & rot)
{
    assert(rot.type() == CV_64FC1);
    assert(rot.rows == 3 && rot.cols == 3);
    
    Mat ret = cv::Mat::zeros(4, 1, CV_64FC1);
    
    float r11 = rot.at<double>(0, 0);
    float r12 = rot.at<double>(0, 1);
    float r13 = rot.at<double>(0, 2);
    float r21 = rot.at<double>(1, 0);
    float r22 = rot.at<double>(1, 1);
    float r23 = rot.at<double>(1, 2);
    float r31 = rot.at<double>(2, 0);
    float r32 = rot.at<double>(2, 1);
    float r33 = rot.at<double>(2, 2);
    
    float q0 = ( r11 + r22 + r33 + 1.0f) / 4.0f;
    float q1 = ( r11 - r22 - r33 + 1.0f) / 4.0f;
    float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
    float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
    if(q0 < 0.0f) q0 = 0.0f;
    if(q1 < 0.0f) q1 = 0.0f;
    if(q2 < 0.0f) q2 = 0.0f;
    if(q3 < 0.0f) q3 = 0.0f;
    q0 = sqrt(q0);
    q1 = sqrt(q1);
    q2 = sqrt(q2);
    q3 = sqrt(q3);
    if(q0 >= q1 && q0 >= q2 && q0 >= q3) {
        q0 *= +1.0f;
        q1 *= CvxPoseEstimation::SIGN(r32 - r23);
        q2 *= CvxPoseEstimation::SIGN(r13 - r31);
        q3 *= CvxPoseEstimation::SIGN(r21 - r12);
    } else if(q1 >= q0 && q1 >= q2 && q1 >= q3) {
        q0 *= CvxPoseEstimation::SIGN(r32 - r23);
        q1 *= +1.0f;
        q2 *= CvxPoseEstimation::SIGN(r21 + r12);
        q3 *= CvxPoseEstimation::SIGN(r13 + r31);
    } else if(q2 >= q0 && q2 >= q1 && q2 >= q3) {
        q0 *= CvxPoseEstimation::SIGN(r13 - r31);
        q1 *= CvxPoseEstimation::SIGN(r21 + r12);
        q2 *= +1.0f;
        q3 *= CvxPoseEstimation::SIGN(r32 + r23);
    } else if(q3 >= q0 && q3 >= q1 && q3 >= q2) {
        q0 *= CvxPoseEstimation::SIGN(r21 - r12);
        q1 *= CvxPoseEstimation::SIGN(r31 + r13);
        q2 *= CvxPoseEstimation::SIGN(r32 + r23);
        q3 *= +1.0f;
    } else {
        printf("q0, q1, q2, q3: %f %f %f %f\n", q0, q1, q2, q3);
        printf("Error: rotation matrix quaternion.\n");
        cout<<"rotation matrix is \n"<<rot<<endl;
        assert(0);
    }
    float r = CvxPoseEstimation::NORM(q0, q1, q2, q3);
    q0 /= r;
    q1 /= r;
    q2 /= r;
    q3 /= r;
    
    ret.at<double>(0, 0) = q0;
    ret.at<double>(1, 0) = q1;
    ret.at<double>(2, 0) = q2;
    ret.at<double>(3, 0) = q3;
    return ret;
}

Mat CvxPoseEstimation::quaternionToRotation(const cv::Mat & q)
{
    assert(q.type() == CV_64FC1);
    assert(q.rows == 4);
    assert(q.cols == 1);
    
    double x = q.at<double>(0, 0);
    double y = q.at<double>(1, 0);
    double z = q.at<double>(2, 0);
    double w = q.at<double>(3, 0);
    
    Eigen::Quaterniond quat(w, x, y, z);
    
    Eigen::Matrix<double, 3, 3> eig_mat = quat.matrix();
    cv::Mat rot = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int r = 0; r<3; r++) {
        for (int c = 0; c<3; c++) {
            rot.at<double>(r, c) = eig_mat(r, c);
        }
    }    
    return rot;
}




