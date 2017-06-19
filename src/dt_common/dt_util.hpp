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
    template <class T>
    static double spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices);
    
    // full variance of Gaussian model
    template <class T>
    static double fullVariance(const vector<T>& labels, const vector<unsigned int> & indices);
    
    template <class MatrixType>
    static double sumOfVariance(const vector<MatrixType> & labels, const int row_index,
                                const vector<unsigned int> & indices);
    
    template <class Type1, class Type2>
    static double spatialVariance(const vector<Type1> & labels, const vector<unsigned int> & indices, const vector<Type2> & wt);
    
    template <class T>
    static void meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma);
    
    template <class T>
    static T mean(const vector<T> & data, const vector<unsigned int> & indices);
    
    template <class T>
    static T mean(const vector<T> & data);
    
    // mean and standard of particular row
    template <class matrixType, class vectorType>
    static void rowMeanStddev(const vector<matrixType> & labels, const vector<unsigned int> & indices,
                              const int row_index, vectorType & mean,   vectorType & sigma);
    
    template <class T>
    static void meanMedianError(const vector<T> & errors, T & mean, T & median);
    
    // mean error of each row of a list of matrixes
    template <class MatrixType>
    static void matrixMeanError(const vector<MatrixType> & errors, MatrixType & mean);
    
   
    static double crossEntropy(const VectorXd & prob);
    
    static double balanceLoss(const int leftNodeSize, const int rightNodeSize);
    
    static bool isSameLabel(const vector<unsigned int> & labels, const vector<unsigned int> & indices);
    
    
    static Eigen::MatrixXd confusionMatrix(const vector<unsigned int> & predictions,
                                           const vector<unsigned int> & labels,
                                           const int category_num,
                                           bool normalize);
    
    // accuracy of each category and average
    static Eigen::VectorXd accuracyFromConfusionMatrix(const Eigen::MatrixXd & conf);
    
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
