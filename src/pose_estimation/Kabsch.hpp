//
//  Kabsch.h
//  RGB_RF
//
//  Created by jimmy on 2016-06-14.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef Kabsch_h
#define Kabsch_h

#include <Eigen/Geometry>

// The input 3D points are stored as columns.
Eigen::Affine3d Find3DAffineTransform(Eigen::Matrix3Xd input_pts, Eigen::Matrix3Xd output_pts);

// The input 3D points are stored as columns. No scale effect
Eigen::Affine3d Find3DAffineTransformSameScale(Eigen::Matrix3Xd input_pts, Eigen::Matrix3Xd output_pts);

// The input 3D points are stored as columns.
// rot_input_pts, rot_output_pts: only constrain rotation of the affine transformation
Eigen::Affine3d find3DAffineTransform(Eigen::Matrix3Xd input_pts, Eigen::Matrix3Xd output_pts,
                                      Eigen::Matrix3Xd rot_input_pts, Eigen::Matrix3Xd rot_output_pts);




#endif /* Kabsch_h */
