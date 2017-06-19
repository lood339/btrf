//  Created by jimmy on 2016-05-27.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef __cvx_util__
#define __cvx_util__

#include <stdio.h>
#include <vector>
#include <string>

using std::vector;
using std::string;

class CvxUtil
{
public:
    
    static inline bool isInside(const int width, const int height, const int x, const int y)
    {
        return x >= 0 && y >= 0 && x < width && y < height;
    }
    
    static void splitFilename (const string& str, string &path, string &file);
    
    // quantilization method
    // interval: resolution, the width of bin
    // nBin: tobal number of bins
    static unsigned valueToBinNumber(double v_min, double interval, double value, const unsigned nBin);
    
    // bin: bin index
    static double binNumberToValue(double v_min, double interval, int bin);
    
    template <typename T>
    static vector<size_t> sortIndices(const vector<T> &v) {
        
        // initialize original index locations
        vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i){
            idx[i] = i;
        }
        
        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
        
        return idx;
    }
    
};


#endif /* __cvx_util__ */
