//
//  dt_util.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "dt_util.h"
#include <Eigen/QR>
#include <iostream>

using std::cout;
using std::endl;

template <class T>
double DTUtil::spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices)
{
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);
    
    T mean = T::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        T dif = labels[index] - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j];
        }
    }
    return var;
}

template<class T>
double DTUtil::fullVariance(const vector<T>& labels, const vector<unsigned int> & indices)
{
    assert(indices.size() > 1);
    typedef typename T::Scalar Scalar;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    
    double loss = 0.0;
    
    const int length = (int)labels[0].size();
    MatrixType sampled_data(indices.size(), length);
    for (unsigned i = 0; i<indices.size(); i++) {
        unsigned int index = indices[i];
        assert(index >= 0 && index < labels.size());
        sampled_data.row(i) = labels[index];
    }    
    
    MatrixType centered = sampled_data.rowwise() - sampled_data.colwise().mean();
    MatrixType cov = (centered.adjoint() * centered) / sampled_data.rows();
    
    Eigen::ColPivHouseholderQR<MatrixType> qr(cov);
    if (qr.rank() == length) {
        loss = qr.logAbsDeterminant();
    }
    else {
        loss = qr.logAbsDeterminant();
        // avoid underflow
        if (std::isnan(loss) || std::isinf(loss)) {
            loss = log(0.0000001);
        }
        //printf("Warning: full variance underflow, use a small number log(0.0000001) instead. Sample number %ld\n", indices.size());
        //cout<<"covariance matrix \n"<<cov<<endl;
        //printf("logAbsDeterminant vs log(0.0000001): %lf, %lf\n", qr.logAbsDeterminant(), log(0.0000001));
    }
    
    return loss;
}


template <class MatrixType>
double DTUtil::sumOfVariance(const vector<MatrixType> & labels, const int row_index, const vector<unsigned int> & indices)
{
    typedef typename MatrixType::Scalar Scalar;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ScalarVector;
    
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);
    
    ScalarVector mean = ScalarVector::Zero(labels[0].row(0).size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index].row(row_index);
    }
    mean /= indices.size();
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        ScalarVector dif = labels[index].row(row_index) - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j];
        }
    }
    return var;
}

template<class Type1, class Type2>
double DTUtil::spatialVariance(const vector<Type1> & labels, const vector<unsigned int> & indices, const vector<Type2> & wt)
{
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);
    assert(wt.size() == labels.front().size());
    
    Type1 mean = Type1::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        Type1 dif = labels[index] - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j] * fabs(double(wt[j]));
        }
    }
    return var;
}

template <class T>
void DTUtil::meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma)
{
    assert(indices.size() > 0);
    
    mean = T::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    sigma = T::Zero(labels[0].size());
    if (indices.size() == 1) {
        return;
    }
    for (int i = 0; i<indices.size(); i++) {
        T dif = labels[indices[i]] - mean;
        for (int j = 0; j<sigma.size(); j++) {
            sigma[j] += dif[j] * dif[j];
        }
    }
    for (int j = 0; j<sigma.size(); j++) {
        sigma[j] = sqrt(fabs(sigma[j])/indices.size());
    }
}

template <class T>
T DTUtil::mean(const vector<T> & data, const vector<unsigned int> & indices)
{
    assert(indices.size() > 0);
    
    T m = T::Zero(data[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < data.size());
        m += data[index];
    }
    m /= indices.size();
    
    return m;
}

template <class T>
T DTUtil::mean(const vector<T> & data)
{
    assert(data.size() > 0);
    
    T m = T::Zero(data[0].size());
    
    for (int i = 0; i<data.size(); i++) {
        m += data[i];
    }
    m /= data.size();
    return m;
}

template <class matrixType, class vectorType>
void DTUtil::rowMeanStddev(const vector<matrixType> & labels, const vector<unsigned int> & indices, const int row_index, vectorType & mean, vectorType & sigma)
{
    assert(indices.size() > 0);
    
    mean = vectorType::Zero(labels[0].row(0).size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index].row(row_index);
    }
    mean /= indices.size();
    
    sigma = vectorType::Zero(labels[0].row(0).size());
    if (indices.size() == 1) {
        return;
    }
    for (int i = 0; i<indices.size(); i++) {
        vectorType dif = labels[indices[i]].row(row_index) - mean;
        for (int j = 0; j<sigma.size(); j++) {
            sigma[j] += dif[j] * dif[j];
        }
    }
    for (int j = 0; j<sigma.size(); j++) {
        sigma[j] = sqrt(fabs(sigma[j])/indices.size());
    }
}

template <class T>
void DTUtil::meanMedianError(const vector<T> & errors,
                                  T & mean,
                                  T & median)
{
    assert(errors.size() > 0);
    const int dim = (int)errors[0].size();
    mean = T::Zero(dim);
    median = T::Zero(dim);
    
    vector<vector<double> > each_dim_data(dim);
    for (int i = 0; i<errors.size(); i++) {
        T err = errors[i].cwiseAbs();
        mean += err;
        for (int j = 0; j<err.size(); j++) {
            each_dim_data[j].push_back(err[j]);
        }
    }
    mean /= errors.size();
    
    for (int i = 0; i<each_dim_data.size(); i++) {
        std::sort(each_dim_data[i].begin(), each_dim_data[i].end());
        median[i] = each_dim_data[i][each_dim_data[i].size()/2];
    }
}

template <class MatrixType>
void DTUtil::matrixMeanError(const vector<MatrixType> & errors, MatrixType & mean)
{
    assert(errors.size() > 0);
    
    const int cols  = (int)errors[0].cols();
    const int rows = (int)errors[0].rows();
    mean  = MatrixType::Zero(rows, cols);
    
    for (int i = 0; i<errors.size(); i++) {
        mean += errors[i];
    }
    mean /= errors.size();
}


double
DTUtil::crossEntropy(const VectorXd & prob)
{
    double entropy = 0.0;
    for (int i = 0; i<prob.size(); i++) {
        double p = prob[i];
        assert(p > 0 && p <= 1);
        entropy += - p * std::log(p);
    }
    return entropy;
}


double DTUtil::balanceLoss(const int leftNodeSize, const int rightNodeSize)
{
    double dif = leftNodeSize - rightNodeSize;
    double num = leftNodeSize + rightNodeSize;
    double loss = fabs(dif)/num;
    assert(loss >= 0);
    return loss;
}

bool
DTUtil::isSameLabel(const vector<unsigned int> & labels, const vector<unsigned int> & indices)
{
    assert(indices.size() >= 1);
    unsigned label = labels[indices[0]];
    for (int i = 1; i<indices.size(); i++) {
        if (label != labels[indices[i]]) {
            return false;
        }
    }
    return true;
}

Eigen::MatrixXd
DTUtil::confusionMatrix(const vector<unsigned int> & preds, const vector<unsigned int> & labels,
                        const int category_num,
                        bool normalize)
{
    assert(preds.size() == labels.size());
    assert(category_num > 0);
    
    Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(category_num, category_num);
    for (int i = 0; i<preds.size(); i++) {
        confusion(labels[i], preds[i]) += 1.0;
    }
    if (normalize) {
        confusion = 1.0 / preds.size() * confusion;
    }
    return confusion;
}

Eigen::VectorXd DTUtil::accuracyFromConfusionMatrix(const Eigen::MatrixXd & conf)
{
    assert(conf.rows() == conf.cols());
    
    Eigen::VectorXd acc = Eigen::VectorXd(conf.rows() + 1, 1);
    Eigen::VectorXd row_sum = conf.rowwise().sum();
    double all_sum = conf.sum();
    double trace = conf.trace();
    for (int r = 0; r<conf.rows(); r++) {
        acc[r] = conf(r ,r)/row_sum[r];
    }
    acc[conf.rows()] = trace/all_sum;
    return acc;
}




template double
DTUtil::spatialVariance(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices);

template double
DTUtil::fullVariance(const vector<Eigen::VectorXf>& labels, const vector<unsigned int> & indices);

template double
DTUtil::sumOfVariance(const vector<Eigen::MatrixXf> & labels, const int row_index, const vector<unsigned int> & indices);

template double
DTUtil::spatialVariance(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices, const vector<int> & wt);

template void
DTUtil::meanStddev(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices, Eigen::VectorXf & mean, Eigen::VectorXf & sigma);

template Eigen::VectorXf
DTUtil::mean(const vector<Eigen::VectorXf> & data, const vector<unsigned int> & indices);

template Eigen::VectorXf
DTUtil::mean(const vector<Eigen::VectorXf> & data);

template void
DTUtil::rowMeanStddev(const vector<Eigen::MatrixXf> & labels, const vector<unsigned int> & indices,
                      const int row_index, Eigen::VectorXf & mean, Eigen::VectorXf & sigma);

template void
DTUtil::meanMedianError(const vector<Eigen::VectorXf> & errors, Eigen::VectorXf & mean, Eigen::VectorXf & median);

template void
DTUtil::matrixMeanError(const vector<Eigen::MatrixXf> & errors, Eigen::MatrixXf & mean);





