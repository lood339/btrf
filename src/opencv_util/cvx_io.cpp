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

