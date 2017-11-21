/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        defs.h                                                       *
 *                                                                           *
 * description: The project's common definitions.                            *
 *****************************************************************************/

#ifndef _defs_h_
#define _defs_h_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>

//////////////////////////////////// DATA TYPES ///////////////////////////////////////

typedef int booleanT;


/////////////////////// PROTOTYPES OF PUBLIC FUNCTIONS /////////////////////////////

void exitWithError(char *message);

#ifdef __cplusplus
}
#endif


#endif