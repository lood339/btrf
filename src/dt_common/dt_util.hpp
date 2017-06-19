//
//  dt_util.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__dt_util__
#define __Classifer_RF__dt_util__

// decision tree util
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <string>

using std::vector;
using std::string;
using Eigen::VectorXf;
using Eigen::VectorXd;
using std::vector;

class DTUtil
{
public:
    // spatial variance objective
    template <class T>
    static double spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices);
    
    
    // mean and standard deviation
    template <class T>
    static void meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma);
    
    // mean value of data
    // mask: index of data
    template <class T>
    static T mean(const vector<T> & data,
                  const vector<unsigned int> & mask);
    
    // mean value of data
    template <class T>
    static T mean(const vector<T> & data);
    
    // mean value and median value of errors
    // median value: each dimension is independently computed
    template <class T>
    static void meanMedianError(const vector<T> & errors, T & mean, T & median);
    
    // balance objective
    static double balanceLoss(const int leftNodeSize, const int rightNodeSize);
    
    // [start, step, end)
    template <class T>
    static vector<T> range(int start, int end, int step)
    {
        assert((end - start) * step >= 0);
        vector<T> ret;
        for (int i = start; i < end; i += step) {
            ret.push_back((T)i);
        }
        return ret;
    }

    
};

#endif /* defined(__Classifer_RF__dt_util__) */
