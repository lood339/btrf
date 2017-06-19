//  Created by jimmy on 2017-01-20.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef RGBD_RF_dataset_param_h
#define RGBD_RF_dataset_param_h

#include <unordered_map>
#include <string>

using std::string;
using std::unordered_map;

// 4 scenes and 7 scenes dataset parameter
class DatasetParameter
{
public:
    double depth_factor_;
    double k_focal_length_x_;
    double k_focal_length_y_;
    double k_camera_centre_u_;
    double k_camera_centre_v_;
    double min_depth_;
    double max_depth_;
    
    DatasetParameter()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 585.0;
        k_focal_length_y_ = 585.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
        min_depth_ = 0.05;
        max_depth_ = 6.0;
    }
    
    cv::Mat camera_matrix() const
    {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
        K.at<double>(0, 0) = k_focal_length_x_;
        K.at<double>(1, 1) = k_focal_length_y_;
        K.at<double>(0, 2) = k_camera_centre_u_;
        K.at<double>(1, 2) = k_camera_centre_v_;
        
        return K;
    }
    
    void as4Scenes()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 572.0;
        k_focal_length_y_ = 572.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
        min_depth_ = 0.05;
        max_depth_ = 10.0;
    }
    
    void as7Scenes()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 585.0;
        k_focal_length_y_ = 585.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
        min_depth_ = 0.05;
        max_depth_ = 6.0;
    }
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        const double param_num = 7;
        std::unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0.0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 7);
        
        depth_factor_ = imap[string("depth_factor")];
        k_focal_length_x_ = imap[string("k_focal_length_x")];
        k_focal_length_y_ = imap[string("k_focal_length_y")];
        k_camera_centre_u_ = imap[string("k_camera_centre_u")];
        k_camera_centre_v_ = imap[string("k_camera_centre_v")];
        min_depth_ = imap[string("min_depth")];
        max_depth_ = imap[string("max_depth")];
        return true;
    }
    
    
    bool readFromFileDataParameter(const char* file_name)
    {
        FILE *pf = fopen(file_name, "r");
        if (!pf) {
            printf("Error: can not open %s \n", file_name);
            return false;
        }
        
        const double param_num = 7;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0.0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 7);
        fclose(pf);
        
        depth_factor_ = imap[string("depth_factor")];
        k_focal_length_x_ = imap[string("k_focal_length_x")];
        k_focal_length_y_ = imap[string("k_focal_length_y")];
        k_camera_centre_u_ = imap[string("k_camera_centre_u")];
        k_camera_centre_v_ = imap[string("k_camera_centre_v")];
        min_depth_ = imap[string("min_depth")];
        max_depth_ = imap[string("max_depth")];
        
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "depth_factor %lf\n", depth_factor_);
        fprintf(pf, "k_focal_length_x %lf\n", k_focal_length_x_);
        fprintf(pf, "k_focal_length_y %lf\n", k_focal_length_y_);
        
        fprintf(pf, "k_camera_centre_u %lf\n", k_camera_centre_u_);
        fprintf(pf, "k_camera_centre_v %lf\n", k_camera_centre_v_);
        fprintf(pf, "min_depth %f\n", min_depth_);
        fprintf(pf, "max_depth %f\n\n", max_depth_);
        
        return true;
    }
    
    void printSelf() const
    {
        printf("Dataset parameters:\n");
        printf("depth_factor: %lf\n", depth_factor_);
        printf("k_focal_length_x: %lf\t k_focal_length_y: %lf\n", k_focal_length_x_, k_focal_length_y_);
        printf("k_camera_centre_u: %lf\t k_camera_centre_v_: %lf\n", k_camera_centre_u_, k_camera_centre_v_);
        printf("min depth: %f\t max depth: %f\n", min_depth_, max_depth_);
    }
    
};

#endif
