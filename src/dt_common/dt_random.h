//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRandom__
#define __Classifer_RF__DTRandom__

#include <stdio.h>
#include <vector>

using std::vector;

class vnl_random;

class DTRandom
{
    vnl_random* rnd_generator_;
public:
    DTRandom();
    ~DTRandom();
    
    double getRandomNumber(const double min_v, const double max_v) const;
    
    vector<double> getRandomNumbers(const double min_v, const double max_v, int num) const;
    
    
    
public:
    // out of bagging sampling, the random number generator is related to the machine time
    static void outof_bag_sampling(const unsigned int N,
                                   vector<unsigned int> & bootstrapped,
                                   vector<unsigned int> & outof_bag);
    
    static vector<double>
    generateRandomNumber(const double min_v, const double max_v, int num);
    
    // generate one random number
    static double randomNumber(const double min_v, const double max_v);
    
};

#endif /* defined(__Classifer_RF__DTRandom__) */
