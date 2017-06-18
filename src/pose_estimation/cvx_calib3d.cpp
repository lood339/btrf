//
//  cvxCalib3d.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-12.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_calib3d.hpp"
#include "Kabsch.h"
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

bool CvxCalib3D::EPnPL(const vector<cv::Point2d> & img_pts, const vector<cv::Point3d> & wld_pts,
                       const vector<cv::Point2d> & img_line_end_pts, const vector<cv::Point3d> & wld_line_end_pts,
                       const cv::Mat& camera_matrix, const cv::Mat& distortion_coeff,
                       const cv::Mat& init_rvec, const cv::Mat& init_tvec,
                       cv::Mat& final_rvec, cv::Mat& final_tvec)
{
    assert(img_pts.size() == wld_pts.size());
    assert(img_line_end_pts.size() == wld_line_end_pts.size());
   
    // change point on line constraint to point to point constraint
    vector<cv::Point2d> all_img_pts;
    vector<cv::Point3d> all_wld_pts;
    all_img_pts.insert(all_img_pts.end(), img_pts.begin(), img_pts.end());
    all_img_pts.insert(all_img_pts.end(), img_line_end_pts.begin(), img_line_end_pts.end());
    all_wld_pts.insert(all_wld_pts.end(), wld_pts.begin(), wld_pts.end());
    all_wld_pts.insert(all_wld_pts.end(), wld_line_end_pts.begin(), wld_line_end_pts.end());  
    
    init_rvec.copyTo(final_rvec);
    init_tvec.copyTo(final_tvec);
    bool is_solved = cv::solvePnP(Mat(all_wld_pts), Mat(all_img_pts), camera_matrix, distortion_coeff, final_rvec, final_tvec, true, CV_EPNP);
    return is_solved;
}

void CvxCalib3D::KabschTransform(const vector<cv::Point3d> & src, const vector<cv::Point3d> & dst,
                                 const vector<cv::Point3d> & line_end_src, const vector<cv::Point3d> & line_end_dst,
                                 cv::Mat & affine)
{
    assert(src.size() == dst.size());
    assert(line_end_src.size() == line_end_dst.size());
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
    
    Eigen::Matrix3Xd line_end_in(3, line_end_src.size());
    Eigen::Matrix3Xd line_end_out(3, line_end_dst.size());
    for (int i = 0; i<line_end_src.size(); i++) {
        line_end_in(0, i) = line_end_src[i].x;
        line_end_in(1, i) = line_end_src[i].y;
        line_end_in(2, i) = line_end_src[i].z;
        line_end_out(0, i) = line_end_dst[i].x;
        line_end_out(1, i) = line_end_dst[i].y;
        line_end_out(2, i) = line_end_dst[i].z;
    }    
   
    Eigen::Affine3d aff = find3DAffineTransform(in, out, line_end_in, line_end_out);
    affine = cv::Mat::zeros(3, 4, CV_64FC1);
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<4; j++) {
            affine.at<double>(i, j) = aff(i, j);
        }
    }
}

struct RandomLine3d
{
  //  vector<RandomPoint3d> pts;  //supporting collinear points
  //  cv::Point3d A, B;
  //  cv::Mat covA, covB;
 //   RandomPoint3d rndA, rndB;
    cv::Point3d u;  // unit direction, segment has direction
    cv::Point3d d;  // norm of d is the distance from the origin to line, the direction of d is parallel
    // the normal of plane containing the line and the origin
    // following the representation of Zhang's paper 'determining motion from...'
    
};


static cv::Mat vec2SkewMat (cv::Point3d vec)
{
    cv::Mat m = (cv::Mat_<double>(3,3) <<
                 0, -vec.z, vec.y,
                 vec.z, 0, -vec.x,
                 -vec.y, vec.x, 0);
    return m;
}

static cv::Mat q2r (cv::Mat q)
// input: unit quaternion representing rotation
// output: 3x3 rotation matrix
// note: q=(a,b,c,d)=a + b i + c j + d k, where (b,c,d) is the rotation axis
{
    double a = q.at<double>(0),	b = q.at<double>(1),
    c = q.at<double>(2), d = q.at<double>(3);
    double nm = sqrt(a*a+b*b+c*c+d*d);
    a = a/nm;
    b = b/nm;
    c = c/nm;
    d = d/nm;
    cv::Mat R = (cv::Mat_<double>(3,3)<<
                 a*a+b*b-c*c-d*d,	2*b*c-2*a*d,		2*b*d+2*a*c,
                 2*b*c+2*a*d,		a*a-b*b+c*c-d*d,	2*c*d-2*a*b,
                 2*b*d-2*a*c,		2*c*d+2*a*b,		a*a-b*b-c*c+d*d);
    return R.clone();
}

static cv::Mat cvpt2mat(const cv::Point3d& p, bool homo)
// this function is slow!
// return cv::Mat(3,1,CV_64,arrary) does not work!
{
    if (homo)
        return (cv::Mat_<double>(4,1)<<p.x, p.y, p.z, 1);
    else {
        return (cv::Mat_<double>(3,1)<<p.x, p.y, p.z);
        
    }
}

void computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
// input needs at least 2 correspondences of non-parallel lines
// the resulting R and t works as below: x'=Rx+t for point pair(x,x');
{
    if(a.size()<2)	{
        //cerr<<"Error in computeRelativeMotion_svd: input needs at least 2 pairs!\n";
        return;
    }
    cv::Mat A = cv::Mat::zeros(4,4,CV_64F);
    for(int i=0; i<a.size(); ++i) {
        cv::Mat Ai = cv::Mat::zeros(4,4,CV_64F);
        Ai.at<double>(0,1) = (a[i].u-b[i].u).x;
        Ai.at<double>(0,2) = (a[i].u-b[i].u).y;
        Ai.at<double>(0,3) = (a[i].u-b[i].u).z;
        Ai.at<double>(1,0) = (b[i].u-a[i].u).x;
        Ai.at<double>(2,0) = (b[i].u-a[i].u).y;
        Ai.at<double>(3,0) = (b[i].u-a[i].u).z;
        vec2SkewMat(a[i].u+b[i].u).copyTo(Ai.rowRange(1,4).colRange(1,4));
        A = A + Ai.t()*Ai;
    }
    cv::SVD svd(A);
    cv::Mat q = svd.u.col(3);
    //	cout<<"q="<<q<<endl;
    R = q2r(q);
    cv::Mat uu = cv::Mat::zeros(3,3,CV_64F),
    udr= cv::Mat::zeros(3,1,CV_64F);
    for(int i=0; i<a.size(); ++i) {
        uu = uu + vec2SkewMat(b[i].u)*vec2SkewMat(b[i].u).t();
        udr = udr + vec2SkewMat(b[i].u).t()* (cvpt2mat(b[i].d,0)-R*cvpt2mat(a[i].d,0));
    }
    t = uu.inv()*udr;
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