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

bool Ms7ScenesUtil::read_pose_7_scenes(const char *file_name, Eigen::Affine3d& affine)
{
    cv::Mat pose = Ms7ScenesUtil::read_pose_7_scenes(file_name);
    Eigen::Matrix4d eigen_pose;
    cv2eigen(pose, eigen_pose);
    
    Eigen::Matrix3d r = eigen_pose.block(0, 0, 3, 3);
    Eigen::Vector3d t(eigen_pose(0, 3), eigen_pose(1, 3), eigen_pose(2, 3));
    affine.linear() = r;
    affine.translation() = t;
    
    return true;
}


// return CV_64F
Mat Ms7ScenesUtil::camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    cv::Mat world_depth_img = cv::Mat::zeros(height, width, CV_64F);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0;
            if ((int)camera_depth == 65535) {
                // invalid depth
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_depth_img.at<double>(r, c) = x_world.at<double>(2, 0); // save depth in world coordinate
        }
    }
    return world_depth_img;
}

cv::Mat Ms7ScenesUtil::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if ((int)camera_depth == 65535 || camera_depth < 0.001) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    //world_coordinate_img /= 1000.0;
    return world_coordinate_img;
}



cv::Mat Ms7ScenesUtil::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img,
                                                        const cv::Mat & camera_to_world_pose,
                                                        cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    //cout<<"invet K is "<<inv_K<<endl;
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if (camera_depth == 65.535 || camera_depth < 0.1 || camera_depth > 10.0) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    return world_coordinate_img;
}

void
Ms7ScenesUtil::camera_depth_to_camera_and_world_coordinate(const cv::Mat & camera_depth,
                                                       const cv::Mat & camera_to_world_pose,
                                                       cv::Mat & camera_coord,
                                                       cv::Mat & world_coord,
                                                       cv::Mat & mask)
{
    assert(camera_depth.type() == CV_64FC1);
    
    const int width  = camera_depth.cols;
    const int height = camera_depth.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    //cout<<"invet K is "<<inv_K<<endl;
    
    camera_coord = cv::Mat::zeros(height, width, CV_64FC3);
    world_coord = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double depth = camera_depth.at<double>(r, c)/1000.0; // to meter
            if (depth == 65.535 || depth < 0.1 || depth > 10.0) {
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            // camera coordinate
            camera_coord.at<cv::Vec3d>(r, c)[0] = loc_camera_h.at<double>(0, 0);
            camera_coord.at<cv::Vec3d>(r, c)[1] = loc_camera_h.at<double>(1, 0);
            camera_coord.at<cv::Vec3d>(r, c)[2] = loc_camera_h.at<double>(2, 0);
            
            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coord.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coord.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coord.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
}

cv::Mat
Ms7ScenesUtil::camera_matrix()
{
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;    
    return K;   
}

cv::Mat
Ms7ScenesUtil::camera_depth_to_camera_coordinate(const cv::Mat & camera_depth_img,
                                                 cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    //cout<<"invet K is "<<inv_K<<endl;
    
    cv::Mat camera_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if (camera_depth == 65.535 || camera_depth < 0.1 || camera_depth > 10.0) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            camera_coordinate_img.at<cv::Vec3d>(r, c)[0] = loc_camera.at<double>(0, 0) * scale;
            camera_coordinate_img.at<cv::Vec3d>(r, c)[1] = loc_camera.at<double>(1, 0) * scale;
            camera_coordinate_img.at<cv::Vec3d>(r, c)[2] = loc_camera.at<double>(2, 0) * scale;
           
        }
    }
    return camera_coordinate_img;
}

cv::Mat Ms7ScenesUtil::cameraDepthToWorldCoordinate(const cv::Mat & camera_depth_img,
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
            if (camera_depth < min_depth || camera_depth > max_depth ) {
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


bool Ms7ScenesUtil::load_prediction_result(const char *file_name, string & rgb_img_file, string & depth_img_file, string & camera_pose_file,
                                              vector<cv::Point2d> & img_pts,
                                              vector<cv::Point3d> & wld_pts_pred,
                                              vector<cv::Point3d> & wld_pts_gt)
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
        fscanf(pf, "%s", buf);
        camera_pose_file = string(buf);
    }
    
    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        printf("%s\n", dummy_buf);
    }
    
    while (1) {
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        // filter out zero points
        img_pts.push_back(cv::Point2f(val[0], val[1]));
        wld_pts_pred.push_back(cv::Point3f(val[2], val[3], val[4]));
        wld_pts_gt.push_back(cv::Point3f(val[5], val[6], val[7]));
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    
    return true;
}

bool Ms7ScenesUtil::load_prediction_result_with_color(const char *file_name,
                                                         string & rgb_img_file,
                                                         string & depth_img_file,
                                                         string & camera_pose_file,
                                                         vector<cv::Point2d> & img_pts,
                                                         vector<cv::Point3d> & wld_pts_pred,
                                                         vector<cv::Point3d> & wld_pts_gt,
                                                         vector<cv::Vec3d> & color_pred,
                                                         vector<cv::Vec3d> & color_sample)
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
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        
        // 2D , 3D position
        img_pts.push_back(cv::Point2d(val[0], val[1]));
        wld_pts_pred.push_back(cv::Point3d(val[2], val[3], val[4]));
        wld_pts_gt.push_back(cv::Point3d(val[5], val[6], val[7]));
        
        double val2[6] = {0.0};
        ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf",
                     &val2[0], &val2[1], &val2[2],
                     &val2[3], &val2[4], &val2[5]);
        if (ret != 6) {
            break;
        }
        color_pred.push_back(cv::Vec3d(val2[0], val2[1], val2[2]));
        color_sample.push_back(cv::Vec3d(val2[3], val2[4], val2[5]));
        assert(img_pts.size() == color_pred.size());
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    return true;
}

bool Ms7ScenesUtil::load_prediction_result_with_color(const char *file_name,
                                                      string & rgb_img_file,
                                                      string & depth_img_file,
                                                      string & camera_pose_file,
                                                      vector<cv::Point2d> & img_pts,
                                                      vector<cv::Point3d> & wld_pts_gt,
                                                      vector<vector<cv::Point3d> > & candidate_wld_pts_pred,
                                                      vector<cv::Vec3d> & color_sample,
                                                      vector<vector<cv::Vec3d> > & candidate_color_pred)
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
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        img_pts.push_back(cv::Point2d(val[0], val[1]));
        wld_pts_gt.push_back(cv::Point3d(val[2], val[3], val[4]));
        color_sample.push_back(cv::Vec3d(val[5], val[6], val[7]));
        
        int num = 0;
        ret = fscanf(pf, "%d", &num);
        assert(ret == 1);
        
        vector<cv::Point3d> wld_pts_pred;
        vector<cv::Vec3d> color_pred;
        for (int i = 0; i<num; i++) {
            ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5]);
            assert(ret == 6);
            wld_pts_pred.push_back(cv::Point3d(val[0], val[1], val[2]));
            color_pred.push_back(cv::Vec3d(val[3], val[4], val[5]));
        }
        candidate_wld_pts_pred.push_back(wld_pts_pred);
        candidate_color_pred.push_back(color_pred);
    }
    fclose(pf);
    
    assert(img_pts.size() == wld_pts_gt.size());
    assert(img_pts.size() == candidate_wld_pts_pred.size());
    assert(img_pts.size() == candidate_color_pred.size());
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    
    return true;
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

bool Ms7ScenesUtil::load_prediction_result_with_uncertainty(const char *file_name,
                                             string & rgb_img_file,
                                             string & depth_img_file,
                                             string & camera_pose_file,
                                             vector<Eigen::Vector2d> & img_pts,
                                             vector<Eigen::Vector3d> & wld_pts_gt,
                                             vector<vector<Eigen::Vector3d> > & candidate_wld_pts_pred,
                                             vector<vector<Eigen::Matrix3d> > & candidate_wld_pts_pred_covariance,
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
        img_pts.push_back(Eigen::Vector2d(val[0], val[1]));
        wld_pts_gt.push_back(Eigen::Vector3d(val[2], val[3], val[4]));
        
        
        int num = 0;
        ret = fscanf(pf, "%d", &num);
        assert(ret == 1);
        
        vector<Eigen::Vector3d> wld_pts_pred;
        vector<Eigen::Matrix3d> wld_pts_pred_cov;
        vector<double> feat_dists;
        for (int i = 0; i<num; i++) {
            ret = fscanf(pf, "%lf %lf %lf %lf", &val[0], &val[1], &val[2], &val[3]);
            assert(ret == 4);
            wld_pts_pred.push_back(Eigen::Vector3d(val[0], val[1], val[2]));
            feat_dists.push_back(val[3]);
            
            // read covariance matrix
            Eigen::Matrix3d cov;
            for (int j = 0; j<3; j++) {
                for (int k = 0; k<3; k++) {
                    double cur_val = 0.0;
                    ret = fscanf(pf, "%lf", &cur_val);
                    assert(ret == 1);
                    cov(j, k) = cur_val;
                }
            }
            wld_pts_pred_cov.push_back(cov);
        }
        candidate_wld_pts_pred.push_back(wld_pts_pred);
        candidate_wld_pts_pred_covariance.push_back(wld_pts_pred_cov);
        candidate_feature_dists.push_back(feat_dists);
    }
    fclose(pf);
    
    assert(img_pts.size() == wld_pts_gt.size());
    assert(img_pts.size() == candidate_wld_pts_pred.size());
    assert(img_pts.size() == candidate_wld_pts_pred_covariance.size());
    assert(img_pts.size() == candidate_feature_dists.size());
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    return true;
}

bool Ms7ScenesUtil::load_estimated_camera_pose(const char *file_name,
                                                  string & rgb_img_file,
                                                  string & depth_img_file,
                                                  string & camera_pose_file,
                                                  cv::Mat & estimated_pose)
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
    
    estimated_pose = cv::Mat::eye(4, 4, CV_64FC1);
    
    for (int r = 0; r<4; r++) {
        for (int c = 0; c<4; c++) {
            double val = 0.0;
            int ret = fscanf(pf, "%lf", &val);
            assert(ret == 1);
            estimated_pose.at<double>(r, c) = val;
        }
    }
    
    fclose(pf);
    return true;
}

vector<string> Ms7ScenesUtil::read_file_names(const char *file_name)
{
    vector<string> file_names;
    FILE *pf = fopen(file_name, "r");
    assert(pf);
    while (1) {
        char line[1024] = {NULL};
        int ret = fscanf(pf, "%s", line);
        if (ret != 1) {
            break;
        }
        file_names.push_back(string(line));
    }
    printf("read %lu lines\n", file_names.size());
    fclose(pf);
    return file_names;
}
