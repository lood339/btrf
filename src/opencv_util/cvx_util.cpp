//
//  cvxUtil.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-27.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_util.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>


vector<double>
CvxUtil::generateRandomNumbers(double min_val, double max_val, int rnd_num)
{
    assert(rnd_num > 0);
    
    cv::RNG rng;
    vector<double> data;
    for (int i = 0; i<rnd_num; i++) {
        data.push_back(rng.uniform(min_val, max_val));
    }
    return data;
}

void
CvxUtil::splitFilename (const string& str, string &path, string &file)
{
    assert(!str.empty());
    unsigned int found = (unsigned int )str.find_last_of("/\\");
    path = str.substr(0, found);
    file = str.substr(found + 1);
}


unsigned
CvxUtil::valueToBinNumber(double v_min, double interval, double value, const unsigned nBin)
{
    int num = (value - v_min)/interval;
    if (num < 0) {
        return 0;
    }
    if (num >= nBin) {
        return nBin - 1;
    }
    return (unsigned)num;
}

double
CvxUtil::binNumberToValue(double v_min, double interval, int bin)
{
    return v_min + bin * interval;
}

