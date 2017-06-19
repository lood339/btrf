//  Created by jimmy on 2016-06-14.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef __Kabsch__
#define __Kabsch__

// This code is modifed from released in public domain from
// https://github.com/oleg-alexandrov/projects/blob/master/eigen/Kabsch.cpp

#include <Eigen/Geometry>

// This is general Kabsh algorithm,
// The input 3D points are stored as columns.
Eigen::Affine3d Find3DAffineTransform(Eigen::Matrix3Xd input_pts, Eigen::Matrix3Xd output_pts);

// This is Kabsh algorithm for camera pose estimation
// Assume the points in camera coordiantes and world coordinates have same scale
// The input 3D points are stored as columns. No scale effect
Eigen::Affine3d Find3DAffineTransformSameScale(Eigen::Matrix3Xd input_pts, Eigen::Matrix3Xd output_pts);




#endif /* __Kabsch__ */
