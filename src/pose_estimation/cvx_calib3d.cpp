//
//  cvxCalib3d.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-12.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_calib3d.hpp"
#include "Kabsch.hpp"
#include <opencv2/calib3d/calib3d.hpp>

using cv::Mat;

void CvxCalib3D::rigidTransform(const vector<cv::Point3d> & src, const cv::Mat & affine, vector<cv::Point3d> & dst)
{
    cv::Mat rot   = affine(cv::Rect(0, 0, 3, 3));
    cv::Mat trans = affine(cv::Rect(3, 0, 1, 3));
    
    for (int i = 0; i<src.size(); i++) {
        cv::Mat p = rot * cv::Mat(src[i]) + trans;
        dst.push_back(cv::Point3d(p));
    }    
}

void CvxCalib3D::KabschTransform(const vector<cv::Point3d> & src, const vector<cv::Point3d> & dst, cv::Mat & affine)
{
    assert(src.size() == dst.size());
    assert(src.size() >= 4);
    
    Eigen::Matrix3Xd in(3, src.size());
    Eigen::Matrix3Xd out(3, dst.size());
    
    for (int i = 0; i<src.size(); i++) {
        in(0, i) = src[i].x;
        in(1, i) = src[i].y;
        in(2, i) = src[i].z;
        out(0, i) = dst[i].x;
        out(1, i) = dst[i].y;
        out(2, i) = dst[i].z;
    }
    Eigen::Affine3d aff = Find3DAffineTransformSameScale(in, out);
    affine = cv::Mat::zeros(3, 4, CV_64FC1);
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<4; j++) {
            affine.at<double>(i, j) = aff(i, j);
        }
    }
}

Eigen::Affine3d CvxCalib3D::KabschTransform(const vector<Eigen::Vector3d> & src, const vector<Eigen::Vector3d> & dst)
{
    assert(src.size() == dst.size());
    assert(src.size() >= 4);
    
    Eigen::Matrix3Xd in(3, src.size());
    Eigen::Matrix3Xd out(3, dst.size());
    
    for (int i = 0; i<src.size(); i++) {
        in(0, i) = src[i].x();
        in(1, i) = src[i].y();
        in(2, i) = src[i].z();
        out(0, i) = dst[i].x();
        out(1, i) = dst[i].y();
        out(2, i) = dst[i].z();
    }
    
    Eigen::Affine3d affine = Find3DAffineTransformSameScale(in, out);
    return affine;
}

cv::Mat CvxCalib3D::cameraDepthToWorldCoordinate(const cv::Mat & camera_depth_img,
                                     const cv::Mat & camera_to_world_pose,
                                     const cv::Mat & calibration_matrix,
                                     const double depth_factor,
                                     const double min_depth,
                                     const double max_depth,
                                     cv::Mat & camera_coordinate,
                                     cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    assert(camera_to_world_pose.type() == CV_64FC1);
    assert(calibration_matrix.type() == CV_64FC1);
    assert(min_depth < max_depth);
    assert(min_depth >= 0.0);
    
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    cv::Mat inv_K = calibration_matrix.inv();
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    cv::Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    camera_coordinate = cv::Mat::zeros(height, width, CV_64FC3);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/depth_factor; // to meter
            if (camera_depth < min_depth || max_depth > max_depth) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            cv::Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            // the x, y, z in camera coordininate 
            camera_coordinate.at<cv::Vec3d>(r,c)[0] = loc_camera_h.at<double>(0, 0);
            camera_coordinate.at<cv::Vec3d>(r,c)[1] = loc_camera_h.at<double>(1, 0);
            camera_coordinate.at<cv::Vec3d>(r,c)[2] = loc_camera_h.at<double>(2, 0);
            
            cv::Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    return world_coordinate_img;

}