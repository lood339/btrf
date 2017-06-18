//  Created by jimmy on 2016-05-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_io.hpp"
#include <iostream>
#include <dirent.h>

using cv::Mat;
using std::cout;
using std::endl;

bool CvxIO::imread_depth_16bit_to_32f(const char *file, cv::Mat & depth_img)
{
    depth_img = cv::imread(file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depth_img.empty()) {
        printf("Error: can not read image from %s\n", file);
        return false;
    }
    assert(depth_img.type() == CV_16UC1);
    depth_img.convertTo(depth_img, CV_32F);
    return true;
}

bool CvxIO::imread_depth_16bit_to_64f(const char *filename, cv::Mat & depth_img)
{
    depth_img = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depth_img.empty()) {
        printf("Error: can not read image from %s\n", filename);
        return false;
    }
    assert(depth_img.type() == CV_16UC1);
    depth_img.convertTo(depth_img, CV_64F);
    return true;
}

bool CvxIO::imread_rgb_8u(const char *file_name, cv::Mat & rgb_img)
{
    rgb_img = cv::imread(file_name, CV_LOAD_IMAGE_COLOR);
    if (rgb_img.empty()) {
        printf("Error: can not read image from %s\n", file_name);
        return false;
    }
    assert(rgb_img.type() == CV_8UC3);
    return true;
}

bool CvxIO::imread_gray_8u(const char *file_name, cv::Mat & grey_img)
{
    grey_img = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (grey_img.empty()) {
        printf("Error: can not read image from %s\n", file_name);
        return false;
    }
    assert(grey_img.type() == CV_8UC1);
    return true;
}

void CvxIO::imwrite_depth_8u(const char *file, const cv::Mat & depth_img)
{
    assert(depth_img.type() == CV_32F || depth_img.type() == CV_64F);
    assert(depth_img.channels() == 1);
    
    double minv = 0.0;
    double maxv = 0.0;
    cv::minMaxLoc(depth_img, &minv, &maxv);
    
    printf("min, max values are: %lf %lf\n", minv, maxv);
    
    
    cv::Mat shifted_depth_map;
    depth_img.convertTo(shifted_depth_map, CV_32F, 1.0, -minv);
    cv::Mat depth_8u;
    shifted_depth_map.convertTo(depth_8u, CV_8UC1, 255/(maxv - minv));
    
    cv::imwrite(file, depth_8u);
    printf("save to: %s\n", file);
}

void CvxIO::imwrite_xyz_to_8urgb(const char *file, const cv::Mat & xyz_img)
{
    assert(xyz_img.type() == CV_32FC3 || xyz_img.type() == CV_64FC3);
    
    cv::Mat single_channel = xyz_img.reshape(1);
    
    double minv = 0.0;
    double maxv = 0.0;
    cv::minMaxLoc(single_channel, &minv, &maxv);
    
    printf("min, max values are: %lf %lf\n", minv, maxv);
    cv::Mat shifted_image;
    single_channel.convertTo(shifted_image, CV_32F, 1.0, -minv);
    
    cv::Mat single_channel_8u;
    shifted_image.convertTo(single_channel_8u, CV_8UC1, 255/(maxv - minv));
    
    cv::Mat bgr_image = single_channel_8u.reshape(3, xyz_img.rows);
    cv::Mat rgb_img;
    cv::cvtColor(bgr_image, rgb_img, CV_BGR2RGB);
    
    cv::imwrite(file, rgb_img);
    printf("save to: %s\n", file);
    
}

bool CvxIO::save_mat(const char *txtfile, const cv::Mat & mat)
{
    assert(mat.type() == CV_64FC1);
    FILE * pf = fopen(txtfile, "w");
    if (!pf) {
        printf("Error: can not write to %s \n", txtfile);
        return false;
    }
    fprintf(pf, "%d %d\n", mat.rows, mat.cols);
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x< mat.cols; x++) {
            fprintf(pf, "%lf ", mat.at<double>(y, x));
        }
        fprintf(pf, "\n");
    }
    fclose(pf);
    printf("save to %s\n", txtfile);
    return true;
}

bool CvxIO::load_mat(const char *txtfile, cv::Mat & mat)
{
    FILE *pf = fopen(txtfile, "r");
    if (!pf) {
        printf("Error: can not read from %s \n", txtfile);
        return false;
    }
    int h = 0;
    int w = 0;
    int num = fscanf(pf, "%d %d", &h, &w);
    assert(num == 2);
    mat = cv::Mat::zeros(h, w, CV_64FC1);
    for (int y = 0; y<h; y++) {
        for (int x = 0; x<w; x++) {
            double val = 0;
            num = fscanf(pf, "%lf", &val);
            assert(num ==1);
            mat.at<double>(y, x) = val;
        }
    }
    fclose(pf);
    return true;
}

bool CvxIO::write_mat(FILE *pf, const cv::Mat & mat)
{
    assert(pf);
    fprintf(pf, "%d %d\n", mat.rows, mat.cols);
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x< mat.cols; x++) {
            fprintf(pf, "%lf ", mat.at<double>(y, x));
        }
        fprintf(pf, "\n");
    }
    return true;
    
}
bool CvxIO::read_mat(FILE *pf, cv::Mat & mat)
{
    int h = 0;
    int w = 0;
    int num = fscanf(pf, "%d %d", &h, &w);
    assert(num == 2);
    mat = cv::Mat::zeros(h, w, CV_64FC1);
    for (int y = 0; y<h; y++) {
        for (int x = 0; x<w; x++) {
            double val = 0;
            num = fscanf(pf, "%lf", &val);
            assert(num ==1);
            mat.at<double>(y, x) = val;
        }
    }
    return true;
}


vector<string> CvxIO::read_files(const char *dir_name)
{    
    const char *post_fix = strrchr(dir_name, '.');
    string pre_str(dir_name);
    pre_str = pre_str.substr(0, pre_str.rfind('/') + 1);
    //printf("pre_str is %s\n", pre_str.c_str());
    
    assert(post_fix);
    vector<string> file_names;
    DIR *dir = NULL;
    struct dirent *ent = NULL;
    if ((dir = opendir (pre_str.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            const char *cur_post_fix = strrchr( ent->d_name, '.');
            //printf("cur post_fix is %s %s\n", post_fix, cur_post_fix);
            
            if (!strcmp(post_fix, cur_post_fix)) {
                file_names.push_back(pre_str + string(ent->d_name));
              //  cout<<file_names.back()<<endl;
            }
            
            //printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    }
    printf("read %lu files\n", file_names.size());
    return file_names;
}

vector<string> CvxIO::read_file_names(const char *file_name)
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

