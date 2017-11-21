/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        wh.h                                                         *
 *                                                                           *
 * description: A utility for performing pattern matching using projection   *
 *              on Walsh Hadamard kernels.                                   *
 *              A WHSetup struct should be created first, defining the       *
 *              sizes of the source image and the pattern, and the required  *
 *              amount of WH kernels to be used.                             *
 *              Next, the desired source image and pattern for matching      *
 *              should be specified.                                         *
 *              Now the pattern matching may be performed, and the results   *
 *              may be obtained from the WHSetup using the available         *
 *              macros. The results include an array of Matches (each match  *
 *              contains the location of the match in the source image and   *
 *              the euclidean distance of the match from the pattern) and    *
 *              number of matches that were found.                           *
 *****************************************************************************/

#ifndef _walsh_h_
#define _walsh_h_

#ifdef __cplusplus
extern "C" {
#endif
    
#include "defs.h"
#include "matrix.h"
#include "image.h"

//////////////////////////////// COMPILATION DEFINITIONS //////////////////////////////

// define only one of the following three definitions according to the required pattern size:
//#define MAX_PATTERN_32 // define if max pattern size is 32x32
#define MAX_PATTERN_64 // define if max pattern size is 64x64
//#define MAX_PATTERN_128 // define if max pattern size is 128x128

#define MAX_BASIS_1024 // define if up to 1024 basis are used (which is always the case in 32x32)

// the following values should only be changed if they produce better running time
// on the specific machine.
#define BUFFER_SIZE_32 16832 // the buffer size to use for patterns sizes up to 32x32
#define BUFFER_SIZE_64 34664 // the buffer size to use for patterns sizes up to 64x64
#define BUFFER_SIZE_128 69328 // the buffer size to use for patterns sizes up to 128x128

//////////////////////////////////// DATA TYPES ///////////////////////////////////////

typedef signed int basisT;

typedef unsigned int depthT;

typedef int signT;

// A branch in the WH tree. Every node contains a matrix.
// The top matrix is the original image, and the leaf is the convolution of the
// original image with the corresponding WH basis.
typedef struct {
	depthT depth;  // number of levels in the branch (excluding the top original image)
	basisT basisInd; // current basis index of the branch
	Matrix **matrices; // array of depth size: the matrices in each level
	coordT *delta; // array of depth size: the delta between pixels that are added/subtracted in each level
	booleanT *horizontal; // array of depth size: the direction of addition/subtraction in each level (true=horizontal,false=vertical)
	signT **sign; // array [depth][numOfBasis]: the sign (addition/subtraction) for each level and basis index
	basisT **nodeID; // array [depth + 1][numOfBasis]: the ID of each node in the WH tree (in post order)
	basisT *pixelNodeIDValues; // array of the original image size: the node ID to which each pixel in the matrix is updated for.
	basisT **pixelNodeIDRowsPtr; // rows pointer for the pixelNodeIDValues array.
#ifdef MAX_BASIS_1024
	depthT **climbLevels; // array [numOfBasis][numOfBasis]: the levels to climb in the tree when moving from one basis index to another.
#else 
	depthT *climbLevels; // array of numOfBasis size: the levels to climb in the tree when moving from a basis index to the following basis.
#endif
} Branch;

// A match between the pattern and the source image.
typedef struct {
	coordT y, x; // the location of the match in the source image
	distanceT distance; // the euclidean distance between the pattern and the match
} Match;

// A setup for pattern matching. A setup is created per source image size, pattern size &
// required number of WH basis. Whenever the source image or pattern are 
// changed (and their size remain the same), the setup should be updated. It is also
// possible to set the percentage that control the pattern matching method.
// The result of the pattern matching are updated in the setup (the matches array and
// numOfMatches).
typedef struct {
	coordT sourceRows; // number of rows in the source image
	coordT sourceCols; // number of cols in the source image
	coordT patternRows; // number of rows (and cols) in the pattern image
	basisT numOfBasis; // number of supported WH basis
	Matrix *sourceImage; // source image
	Matrix *patternImage; // pattern image
	cellValueT *patternProjections; // array of numOfBasis size: the projections of the pattern on the first numOfBasis WH basis
	Branch *branch; // a branch in the WH tree
	float startBottomUpPercent; // for suspected windows under this percentage, the bottom up method should be used
	float startDistancePercent; // for suspected windows under this percentage, the direct distance method should be used
	Match *matches; // an array of matches that were found in the pattern matching process
	matrixSizeT numOfMatches; // number of matches found in the pattern matching process
} WHSetup;

//////////////////////////////////// CONSTANTS ///////////////////////////////////////

#define PLUS 1 // represents the plus sign
#define MINUS -1 // represents the minus sign

#define DEFAULT_BOTTOM_UP_PERCENT 10.0 // default percentage under which the bottom up method should be used
#define DEFAULT_DISTANCE_PERCENT 2.0 // default percentage under which the direct distance method should be used

#ifdef MAX_PATTERN_32
	#define BUFFER_SIZE BUFFER_SIZE_32
#else
	#ifdef MAX_PATTERN_64
		#define BUFFER_SIZE BUFFER_SIZE_64
	#else
		#define BUFFER_SIZE BUFFER_SIZE_128
	#endif
#endif



//////////////////////////////////////// MACROS //////////////////////////////////////

// Branch
#define branchMat(branch, matNo)		(branch->matrices[matNo])

#define branchDepth(branch)		    (branch->depth)

#define branchBasisInd(branch)		(branch->basisInd)

#define branchDelta(branch, matNo)	(branch->delta[matNo])

#define branchHorizontal(branch, matNo)	   (branch->horizontal[matNo])

#define branchSign(branch, matNo, basisInd) (branch->sign[matNo][basisInd])

#define branchNodeID(branch, matNo, basisInd) (branch->nodeID[matNo][basisInd])

#define branchPixelNodeID(branch, y, x) (branch->pixelNodeIDRowsPtr[y][x])

#define branchPixelNodeIDPtr(branch) (branch->pixelNodeIDValues)

#ifdef MAX_BASIS_1024
	#define branchClimbLevels(branch, toBasisInd) (branch->climbLevels[branch->basisInd][toBasisInd])
#else
	#define branchClimbLevels(branch, toBasisInd) ((toBasisInd - branch->basisInd == 1) ? branch->climbLevels[branch->basisInd] : getClimbLevels(toBasisInd, branch->basisInd))
#endif

#define branchMatVal(branch, matNo, y, x)    (matVal(branch->matrices[matNo], y, x))

// Match
#define matchY(match)        (match->y)
#define matchX(match)        (match->x)
#define matchDistance(match) (match->distance)

// WHSetup
#define matches(setup)      (setup->matches)
#define numOfMatches(setup) (setup->numOfMatches)
#define sourceImage(setup)  (setup->sourceImage)
#define patternImage(setup) (setup->patternImage)

/////////////////////// PROTOTYPES OF PUBLIC FUNCTIONS /////////////////////////////

WHSetup *createWHSetup(coordT sourceRows, coordT sourceCols, coordT patternRows, basisT numOfBasis);
void destroyWHSetup(WHSetup *setup);

void setPatternImage(WHSetup *setup, Image *pattern);
void setSourceImage(WHSetup *setup, Image *source);
void setMethodStartPercent(WHSetup *setup, float bottomUpStartPercent, float distanceStartPercent);

void whPatternMatch(WHSetup *setup, distanceT rejectThresh);
    
#ifdef __cplusplus
}
#endif

#endif