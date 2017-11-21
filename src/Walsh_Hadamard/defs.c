/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        defs.c                                                       *
 *                                                                           *
 * description: The project's common definitions.                            *
 *****************************************************************************/

#include "defs.h"

// Prints the given error message and halts the program.
void exitWithError(char *message) {
    printf("%s\n",message);  
    exit(1);
}
