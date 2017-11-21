/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        wh.c                                                         *
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

#include "wh.h"
#include <stdlib.h>
#include <math.h>
#include <memory.h>

#pragma inline_recursion(on)

#ifdef MAX_PATTERN_32
	#pragma inline_depth(10)
#else
	#ifdef MAX_PATTERN_64
		#pragma inline_depth(12)
	#else
		#pragma inline_depth(14)
	#endif
#endif


/////////////////////////////// GLOBALS /////////////////////////////////////////

// holds the Walsh-Hadamard tree signs order
signT signs[8] = {PLUS,MINUS,PLUS,MINUS,MINUS,PLUS,MINUS,PLUS};

/////////////////////////////////////// GLOBALS /////////////////////////////////////

cellValueT buffer[BUFFER_SIZE]; // buffer for heavy memory operations

////////////////////////////// PROTOTYPES ///////////////////////////////////////

Branch * allocBranch(coordT rows, coordT cols, depthT depth, basisT numOfBasis);
void destoryBranch(Branch *branch);

void addH(Matrix *sourceMat, Matrix *destMat, coordT delta);
void subH(Matrix *sourceMat, Matrix *destMat, coordT delta);
void addV(Matrix *sourceMat, Matrix *destMat, coordT delta);
void subV(Matrix *sourceMat, Matrix *destMat, coordT delta);

signT getSign(basisT basisInd, depthT levelFromBottom);
depthT getClimbLevels(basisT sourceBasisInd, basisT destBasisInd);
cellValueT computePixel(Branch *branch, basisT basisInd, depthT currentMatNo, coordT y, coordT x);

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// FUNCTIONS //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

// added
static
__inline unsigned int min(unsigned int a, unsigned int b)
{
    if (a < b) {
        return a;
    }
    else {
        return b;
    }    
}

//////////////////////////////////// ALLOCATIONS //////////////////////////////////////

/*
Allocates and returns an Branch of the given depth with the given matrix sizes.
The number of matrices created in the branch is (2 * depth) + 1:
 - The top matrix is reserved for the source matrix, and is not allocated.
 - each unit of depth allocates two matrices, one for the horizontal convolution and one
   for the vertical convolution.
The basis index of the branch is initialized to -1.
*/
Branch *allocBranch(coordT rows, coordT cols, depthT depth, basisT numOfBasis)
{
	Branch *branch;
	coordT delta, row;
	depthT numOfMatrices = (depth << 1) + 1;
	basisT numOfLeaves = (basisT)pow(2, numOfMatrices - 1);
	basisT divider = 2;
	depthT matInd;
	signT *signPtr;
	basisT *nodeIDPtr, *pixelIDPtr;
	depthT *climbPtr = (depthT *)1; // init with dummy value
	basisT i = 0, j = 0; // init with dummy values

	// allocate branch
	branch = (Branch *)malloc(sizeof(Branch));
	if (!branch)
		exitWithError("ERROR in allocBranch: can't allocate branch.");

	// allocate matrices
	branch->matrices = (Matrix **)malloc(numOfMatrices * sizeof(Matrix *));
	if (!branch->matrices)
		exitWithError("ERROR in allocBranch: can't allocate matrices array mapping.");

	branch->depth = depth << 1;
	branch->basisInd = -1;

	// allocate branch data
	branch->delta = (coordT *)malloc(branch->depth * sizeof(coordT));
	branch->horizontal = (booleanT *)malloc(branch->depth * sizeof(booleanT));
	branch->sign = (signT **)malloc(branch->depth * sizeof(signT *));
	signPtr = (signT *)malloc(branch->depth * numOfBasis * sizeof(signT));
	branch->nodeID = (basisT **)malloc(numOfMatrices * sizeof(basisT *));
	nodeIDPtr = (basisT *)malloc(numOfMatrices * numOfBasis * sizeof(basisT));
	branch->pixelNodeIDRowsPtr = (basisT **)malloc(rows * sizeof(basisT *));
	branch->pixelNodeIDValues = pixelIDPtr = (basisT *)malloc(rows * cols * sizeof(basisT));

#ifdef MAX_BASIS_1024
	branch->climbLevels = (depthT **)malloc(numOfBasis * sizeof(depthT *));
	climbPtr = (depthT *)malloc(numOfBasis * numOfBasis * sizeof(depthT));
#else
	branch->climbLevels = (depthT *)malloc(numOfBasis * sizeof(depthT *));
#endif
	
	if (!branch->delta || !branch->horizontal || !branch->sign || !branch->climbLevels
		|| !branch->nodeID || !branch->pixelNodeIDRowsPtr || !signPtr || !climbPtr || !nodeIDPtr
		|| !pixelIDPtr)
		exitWithError("ERROR in allocBranch: can't allocate branch data.");

	// create pixel node ID 2-d array (uninitialized)
	for (row = 0; row < rows ; row++, pixelIDPtr += cols)
		branch->pixelNodeIDRowsPtr[row] = pixelIDPtr;

	branch->matrices[0] = NULL;
	
	// allocate matrices, where each matrix has 2^i less rows than the previous matrix
	// matrices[0] is not allocated at this point.
	// Also sets the appropriate delta, horizontal & sign array for each level.
	for (matInd = 1, delta = 1; matInd < numOfMatrices; matInd += 2, delta <<= 1) {
		depthT levelFromBottom = numOfMatrices - matInd;

		// fill horizontal & delta array
		branch->horizontal[matInd - 1] = 1;
		branch->horizontal[matInd] = 0;
		branch->delta[matInd - 1] = delta;
		branch->delta[matInd] = delta;

		// fill sign 2-d array
		branch->sign[matInd - 1] = signPtr;
		branch->sign[matInd] = signPtr + numOfBasis;
		i = 0;
		while (i < numOfBasis) {
			*signPtr = getSign(i, levelFromBottom);
			*(signPtr + numOfBasis) = getSign(i, (depthT)(levelFromBottom - 1));
			signPtr++;
			i++;
		}
		signPtr += numOfBasis;

		// fill matrices array
		branch->matrices[matInd] = allocMatrix(rows, cols - delta);
		branch->matrices[matInd + 1] = allocMatrix(rows - delta, cols - delta);
		rows -= delta;
		cols -= delta;
	}
	
	// fill node ID 2-d array

	branch->nodeID[0] = nodeIDPtr;
	
	for (i = 0; i < numOfBasis; i++) // init node ID of first level (level 0)
		*(nodeIDPtr++) = 0;

	for (matInd = 1, divider = 2; matInd < numOfMatrices; matInd++, divider *= 2, nodeIDPtr += numOfBasis) {
		basisT limit = numOfLeaves / divider;
		basisT addedValue = 1;
		booleanT exit = 0;
		basisT basisInd = 0;

		branch->nodeID[matInd] = nodeIDPtr;

		for (i = 0; i < divider && !exit; i++) {
			for (j = 0; j < limit && !exit; j++, basisInd++) {
				exit = (basisInd >= numOfBasis);
				if (!exit)
					branch->nodeID[matInd][basisInd] = branch->nodeID[matInd - 1][basisInd] + addedValue;
			}
			addedValue = limit * 2 - addedValue + 1;
		}
	}

	// fill climb levels array
	i = 0;
	while (i < numOfBasis) {

#ifdef MAX_BASIS_1024
		branch->climbLevels[i] = climbPtr;  // fill 2-d array
		j = 0;
		while (j < numOfBasis) {
			*(climbPtr++) = getClimbLevels(i, j);
			j++;
		}
#else
		branch->climbLevels[i] = getClimbLevels(i, i + 1); // fill 1-d array
#endif

		i++;
	}

	return(branch);
}

/*
Frees the given branch.
*/
void destoryBranch(Branch *branch) {
	depthT i;

	for (i = 1; i <= branch->depth; i++)
		destroyMatrix(branch->matrices[i]);

    free(branch->matrices);  // add by Jimmy chen, to avoid memory leak
	free(branch->delta);
	free(branch->horizontal);
	free(branch->sign[0]);
	free(branch->sign);
	free(branch->nodeID[0]);
	free(branch->nodeID);
	free(branch->pixelNodeIDValues);
	free(branch->pixelNodeIDRowsPtr);

#ifdef MAX_BASIS_1024
	free(branch->climbLevels[0]);
#endif

	free(branch->climbLevels);

	free(branch);
}

///////////////////////////////////// LOW LEVEL ////////////////////////////////////////////

/*
Updates destMat such that destMat[y][x] = sourceMat[y][x] + sourceMat[y][x+delta].
destMat should have delta columns less than sourceMat.
*/
__inline void addH(Matrix *sourceMat, Matrix *destMat, coordT delta) {
	cellValueT *sourcePtr = matValPtr(sourceMat),
		       *destPtr = matValPtr(destMat);
	coordT destCols = matCols(destMat),
		   sourceCols = matCols(sourceMat);
	matrixSizeT total = matSize(sourceMat);
	unsigned int maxBuffer = (BUFFER_SIZE / sourceCols) * sourceCols;

	while (total) {
        unsigned int count = min(total, maxBuffer);
		cellValueT *bufferPtr = memcpy(buffer, sourcePtr, count << 2);
		sourcePtr += count;
		total -= count;
		while (count) {
			coordT col = destCols;
			while (col--)
				*(destPtr++) = *(bufferPtr) + *(bufferPtr++ + delta);
			bufferPtr += delta;
			count -= sourceCols;
		}
	}
}

/*
Updates destMat such that destMat[y][x] = sourceMat[y][x] - sourceMat[y][x+delta].
destMat should have delta columns less than sourceMat.
*/
__inline void subH(Matrix *sourceMat, Matrix *destMat, coordT delta) {
	cellValueT *sourcePtr = matValPtr(sourceMat),
			   *destPtr = matValPtr(destMat);
	coordT destCols = matCols(destMat),
		   sourceCols = matCols(sourceMat);
	matrixSizeT total = matSize(sourceMat);
	unsigned int maxBuffer = (BUFFER_SIZE / sourceCols) * sourceCols;

	while (total) {
		unsigned int count = min(total, maxBuffer);
		cellValueT *bufferPtr = memcpy(buffer, sourcePtr, count << 2);
		sourcePtr += count;
		total -= count;
		while (count) {
			coordT col = destCols;
			while (col--)
				*(destPtr++) = *(bufferPtr) - *(bufferPtr++ + delta);
			bufferPtr += delta;
			count -= sourceCols;
		}
	}
}

/*
Updates destMat such that destMat[y][x] = sourceMat[y][x] + sourceMat[y+delta][x].
destMat should have delta rows less than sourceMat.
*/
__inline void addV(Matrix *sourceMat, Matrix *destMat, coordT delta) {
	matrixSizeT total = matSize(destMat);
	unsigned int d = delta * matCols(destMat),
		         maxBuffer = BUFFER_SIZE - d;
	cellValueT *sourcePtr = matValPtr(sourceMat),
		       *destPtr = matValPtr(destMat),
		       *nextBufferPtr = buffer + d;

	while (total) {
		unsigned int count = min(total, maxBuffer);
		cellValueT *bufferPtr = memcpy(buffer, sourcePtr, (count + d) << 2),
			   *bufferPtr1 = nextBufferPtr;
		sourcePtr += count;
		total -= count;
		while (count--)
			*(destPtr++) = *(bufferPtr++) + *(bufferPtr1++);
	}
}

/*
Updates destMat such that destMat[y][x] = sourceMat[y][x] - sourceMat[y+delta][x].
destMat should have delta rows less than sourceMat.
*/
__inline void subV(Matrix *sourceMat, Matrix *destMat, coordT delta) {
	matrixSizeT total = matSize(destMat);
	unsigned int d = delta * matCols(destMat),
		         maxBuffer = BUFFER_SIZE - d;
	cellValueT *sourcePtr = matValPtr(sourceMat),
		       *destPtr = matValPtr(destMat),
		       *nextBufferPtr = buffer + d;

	while (total) {
		unsigned int count = min(total, maxBuffer);
		cellValueT *bufferPtr = memcpy(buffer, sourcePtr, (count + d) << 2),
				   *bufferPtr1 = nextBufferPtr;
		sourcePtr += count;
		total -= count;
		while (count--)
			*(destPtr++) = *(bufferPtr++) - *(bufferPtr1++);
	}
}

/*
Returns the convolution sign according to the basis index and level from the bottom
of the tree (leafs are considered in level 0).
The returned value is PLUS or MINUS.
*/
__inline signT getSign(basisT basisInd, depthT levelFromBottom) {
	return signs[(basisInd >> (levelFromBottom - 1)) & 7]; 
}

/*
Finds the number of levels to climb in the wh tree from the source basis index to the dest
basis index in order to find the lowest common matrix of the two given basis.
*/
__inline depthT getClimbLevels(basisT sourceBasisInd, basisT destBasisInd) {
	// XOR the current and new basis indice
	int xorInd = sourceBasisInd ^ destBasisInd;
	depthT result;

	// find num of levels to go up.
	for (result = 0; xorInd > 0; xorInd >>= 1, result++); 

	return result;
}

/*
Computes, sets and returns the (y,x) pixel of matrix number matNo of the given basis
in the given branch.
*/
__inline cellValueT computePixel(Branch *branch, basisT basisInd, depthT matNo, coordT y, coordT x) {

	cellValueT secondVal, result;
	depthT nextMatNo = matNo - 1;
	basisT nodeID = branchNodeID(branch, nextMatNo, basisInd);

	// pixel is not updated for the current node - find value recursively.
	if (nodeID > branchPixelNodeID(branch, y, x)) {
		// find second value according to horizontal/vertical
		if (branchHorizontal(branch, nextMatNo))
			secondVal = computePixel(branch, basisInd, nextMatNo, 
									 y, (coordT)(x + branchDelta(branch, nextMatNo)));
		else 
			secondVal = computePixel(branch, basisInd, nextMatNo,
									 (coordT)(y + branchDelta(branch, nextMatNo)), x);

		// add/sub second value to/from base value according to sign
		result = computePixel(branch, basisInd, nextMatNo, y, x)
			     + branchSign(branch, nextMatNo, basisInd) * secondVal;

		branchPixelNodeID(branch, y, x) = nodeID;
	}
	// pixel is updated for the current node - just add/sub.
	else {
		// find second value according to horizontal/vertical
		if (branchHorizontal(branch, nextMatNo))
			secondVal = branchMatVal(branch, nextMatNo, y, (coordT)(x + branchDelta(branch, nextMatNo)));
		else 
			secondVal = branchMatVal(branch, nextMatNo, (coordT)(y + branchDelta(branch, nextMatNo)), x);

		// add/sub second value to/from base value according to sign
		result = branchMatVal(branch, nextMatNo, y, x) + secondVal * branchSign(branch, nextMatNo, basisInd);
	}

	branchMatVal(branch, matNo, y, x) = result;
	return result;
}

///////////////////////////////////// HIGH LEVEL ////////////////////////////////////////////

// notifies the branch that all the pixels are updated according to the branch's basis index.
// (updates the nodeID of all pixels with the nodeID of the current branch's leaf).
void updateNodeID(Branch *branch) {
	basisT nodeID = branchNodeID(branch, branchDepth(branch), branchBasisInd(branch));
	basisT *ptr = branchPixelNodeIDPtr(branch);
	matrixSizeT count = matSize(branchMat(branch, 0));

	while (count--)
		*(ptr++) = nodeID;
}

/*
Computes and returns the value of the (y,x) pixel of leaf number basisInd.
If the currentBranch is not of basisInd, it will become such.
*/
cellValueT whPixel(Branch *branch, basisT basisInd, coordT y, coordT x) {
	return computePixel(branch, basisInd, branchDepth(branch), y, x);
}

/*
Updates the current branch to contain the matrix branch of the given basis index.
The leaf matrix of the updated branch is returned.
*/
Matrix * whMatrix(Branch *branch, basisT basisInd) {
	depthT climbLevels, level;

	// if no basis was created in the branch yet, the whole branch is created assuming
	// that the top matrix exists
	if (branchBasisInd(branch) == -1) {
		level = 0;
		climbLevels = branchDepth(branch);
	}
	// a basis was created in this branch before, compute the required starting level
	else {
		// find num of levels to go up.
		climbLevels = branchClimbLevels(branch, basisInd);

		// find level in which the convolution should start
		level = branchDepth(branch) - climbLevels; 
	}

	// replace required matrices in branch starting from the starting level
	for (; level < branchDepth(branch); level++, climbLevels--) {
		
		if (branchHorizontal(branch, level)) {
			// add/sub the current matrix into the next one horizontally (using the current delta)
			if (branchSign(branch, level, basisInd) == PLUS)
				addH(branchMat(branch, level), branchMat(branch, level + 1),
				     branchDelta(branch, level));
			else
				subH(branchMat(branch, level), branchMat(branch, level + 1),
				     branchDelta(branch, level));
		}
		else {
			// add/sub the current matrix into the next one vertically (using the current delta)
			if (branchSign(branch, level, basisInd) == PLUS)
				addV(branchMat(branch, level), branchMat(branch, level + 1),
					 branchDelta(branch, level));
			else
				subV(branchMat(branch, level), branchMat(branch, level + 1),
					 branchDelta(branch, level));
		}
	}

	// set the new basis index.
	branchBasisInd(branch) = basisInd;

	// return the bottom matrix - the projection value of the given basis index
	return branchMat(branch, level);
}

/*
Returns basisNo values which are the projections of win onto basisNo Walsh-Hadamard
basis vectors. The size of the basis vectors is the same as the size of win.
The size of the given win should be 2^k (a power of two).
*/
cellValueT * whWin(Matrix *win, basisT basisNo) {

	cellValueT *result = (cellValueT *)malloc(basisNo * sizeof(cellValueT));
	depthT depth = (depthT)(log10(win->rows) / log10(2));

	Branch *branch;
	basisT basis;

	branch = allocBranch(matRows(win), matCols(win), depth, basisNo);
	branchMat(branch, 0) = win;

	for (basis = 0; basis < basisNo; basis++)
		result[basis] = matVal(whMatrix(branch, basis), 0, 0);
	
	destoryBranch(branch);

	return result;
}


//////////////////////////////////// SETUP //////////////////////////////////////////////

/******************************************************************************
 * Creates and returns a WHSetup that supports source matrices of size        *
 * (sourceRows,sourceCols), patterns of size (patternRows, patternRows) and   *
 * the given number of WH kernels to be used for projection.                  *
 ******************************************************************************/
WHSetup *createWHSetup(coordT sourceRows, coordT sourceCols, coordT patternRows,
							 basisT numOfBasis) {
	WHSetup *result = (WHSetup *)malloc(sizeof(WHSetup));

	if (!result)
		exitWithError("ERROR in createWHSetup: can't allocate setup.");

	result->sourceRows = sourceRows;
	result->sourceCols = sourceCols;
	result->patternRows = patternRows;
	result->numOfBasis = numOfBasis;
	
	result->startBottomUpPercent = DEFAULT_BOTTOM_UP_PERCENT;
	result->startDistancePercent = DEFAULT_DISTANCE_PERCENT;

	result->branch = allocBranch(sourceRows, sourceCols,
									  (depthT)(log10(patternRows) / log10(2)), numOfBasis);
	if (!result->branch)
		exitWithError("ERROR in createWHSetup: can't allocate branch.");

	result->matches = (Match *)malloc((sourceRows - patternRows + 1) *
									  (sourceCols - patternRows + 1) * sizeof(Match));
	if (!result->matches)
		exitWithError("ERROR in createWHSetup: can't allocate matches area.");

	return result;
}

/******************************************************************************
 * Notifies the setup on the given pattern. The pattern must be of the        *
 * size that the setup was created with.                                      *
 ******************************************************************************/
void setPatternImage(WHSetup *setup, Image *pattern) {
	setup->patternImage = imageToMatrix(pattern);
	setup->patternProjections = whWin(setup->patternImage, setup->numOfBasis);
}

/******************************************************************************
 * Notifies the setup on the given source image. The image must be of the     *
 * size that the setup was created with.                                      *
 ******************************************************************************/
void setSourceImage(WHSetup *setup, Image *source) {
	setup->sourceImage = imageToMatrix(source);
	branchMat(setup->branch, 0) = setup->sourceImage;
	branchBasisInd(setup->branch) = -1;
}

/******************************************************************************
 * Notifies the setup on the percentage that should be used for determining   *
 * the pattern matching method. In the pattern matching process, when the     *
 * percentage of suspected windows goes below bottomUpStartPercent, the       *
 * method changes from top down to bottom up. when the percentage of          *
 * suspected windows goes below distanceStartPercent, the method changes from *
 * bottom up to direct distance.                                              *
 ******************************************************************************/
void setMethodStartPercent(WHSetup *setup, float bottomUpStartPercent, float distanceStartPercent) {
	setup->startBottomUpPercent = bottomUpStartPercent;
	setup->startDistancePercent = distanceStartPercent;
}

/*****************************************************************************
 * Destroys the given WHSetup.                                               *
 *****************************************************************************/
void destroyWHSetup(WHSetup *setup) {
	destoryBranch(setup->branch);
	free(setup->patternProjections);
	free(setup->matches);
	free(setup);
}

/////////////////////////////////// PATTERN MATCHING ////////////////////////////////////////

/*****************************************************************************
 * Performs pattern matching on the given setup. The given rejectThresh      *
 * specifies the maximum mean difference (in pixels) between the pattern and *
 * a match.																	 *
 *																			 *
 * After the execution of this function, the resulting matches array and the *
 * number of matches that were found are available in the setup, and can be  *
 * obtained using matches(setup) & numOfMatches(setup).						 *
 *																			 *
 * The pattern matching process is done using three methods, in this order:  *
 * Top Down - the whole image is projected on the WH kernels, starting with  *
 *            the lowest frequency kernel. Each such projection rejects some *
 *            of the suspected windows.	After each projection, the percentage*
 *            of remaining suspected windows is measured.					 *
 * Bottom Up - when the percentage of suspected windows goes under a certain *
 *             margin, the Top Down method is replaced by Bottom Up. In this *
 *             method, only the remaining suspected windows are projected,   *
 *             instead of the whole image. Here as well, each projection     *
 *             rejects some more suspected windows, effecting the percentage *
 *			   of remaining windows. The percentage margin for moving to     *
 *             Bottom Up is initially 10%, and it can be changed using the   *
 *             setMethodStartPercent() function.						     *
 * Direct Distance - when the percentage of suspected windows goes under     *
 *                   a second margin, the Bottom Up method is replaced by    *
 *                   Direct Distance. In this method, the eculidean distance *
 *                   between each of the remaining suspected windows and the *
 *                   pattern is computed, enabling the final rejection of    *
 *                   all the windows with distance above the threshold. The  *
 *				     percentage margin for moving to Direct Distance is      *
 *				     initially 2%, and it can be changed using the           *
 *                   setMethodStartPercent() function.						 *
 *****************************************************************************/
void whPatternMatch(WHSetup *setup, distanceT rejectThresh) {

	coordT y, x, projCols;
	unsigned int dif;
	matrixSizeT numOfWins, suspectedCount, total;
	Matrix *proj;
	Branch *branch = setup->branch;
	Match *currentMatch = matches(setup),
		  *baseMatch = matches(setup);
	cellValueT *projPtr;
	float suspectedPercent, percentFactor;

	booleanT performBottomUp = 0;
	unsigned int distanceShiftAmount = branch->depth;
	cellValueT *patternProj  = setup->patternProjections;  /*pre-processing result from pattern*/
	basisT currentBasis = 1;
	distanceT threshold = rejectThresh * matSize(setup->patternImage); // don't square yet - for first scan

	// Perform initial projection of basis 0
	proj = whMatrix(branch, 0);

	numOfWins = matSize(proj);
	percentFactor = (float)100 / (float)numOfWins;

	// save the locations & distance of all suspected windows in the matches array.
	projPtr = matValPtr(proj);
	total = matSize(proj);
	projCols = matCols(proj);
	x = 0; y = 0;
	while (total) {
		unsigned int count = min(total, BUFFER_SIZE);
		cellValueT *bufferPtr = memcpy(buffer, projPtr, count << 2);
		projPtr += count;
		total -= count;
		while (count--) {
			if ((dif = abs(*bufferPtr - *patternProj)) <= threshold) {
				matchY(currentMatch) = y;
				matchX(currentMatch) = x;
				matchDistance(currentMatch++) = dif * dif;
			}
			bufferPtr++;
			if (++x == projCols) {
				x = 0;
				y++;
			}
		}
	}

	suspectedCount = currentMatch - baseMatch;
	suspectedPercent = (float)suspectedCount * percentFactor;
	threshold *= rejectThresh * matSize(setup->patternImage); // square thresh - for next scans
	patternProj++;

	// Top Down phase: perform full projections for each basis
	// and udpate the "suspected" windows list according to the threshhold.
	while (suspectedPercent > setup->startBottomUpPercent && currentBasis < setup->numOfBasis) {
		total = 0;
		proj = whMatrix(branch, currentBasis);
		currentMatch = baseMatch;

		while (suspectedCount--) {
			dif = matVal(proj, matchY(currentMatch), matchX(currentMatch)) - *patternProj;
			if ((matchDistance(currentMatch) += dif * dif) <= threshold)
				baseMatch[total++] = *currentMatch;
			currentMatch++;
		}

		suspectedCount = total;
		suspectedPercent = (float)suspectedCount * percentFactor;
		currentBasis++;
		patternProj++;
	}

	// check whether Bottom Up phase is needed, or maybe we should go directly to the 
	// Direct Distance phase.
	performBottomUp = suspectedPercent > setup->startDistancePercent &&
					  currentBasis < setup->numOfBasis && suspectedCount;

	// when Top Down phase has ended, the nodeID of the last updated leaf should be updated
	// for all pixels, as a preparation for the Bottom Up phase (if necessary)
	if (performBottomUp)
		updateNodeID(branch);

	// Bottom Up phase: Scan basis, for each one find the projections of the "suspected" windows
	// and udpate the "suspected" windows list according to the threshhold.
	while (performBottomUp) {
		total = 0;
		currentMatch = baseMatch;

		while (suspectedCount--) {
			// increase lower bound by the distance of the current basis
			dif = whPixel(branch, currentBasis, matchY(currentMatch), matchX(currentMatch)) - *patternProj;
			// if lower bound is under threshold, save the current win in the updated win list
			if ((matchDistance(currentMatch) += dif * dif) <= threshold)
	  			baseMatch[total++] = *currentMatch;

			currentMatch++;
		}

		suspectedCount = total;
		suspectedPercent = (float)suspectedCount * percentFactor;
		currentBasis++;
		patternProj++;
		performBottomUp = suspectedPercent > setup->startDistancePercent &&
						  currentBasis < setup->numOfBasis && suspectedCount;
	}

	// Direct Distance phase: compute the direct distance of each suspected window and update
	// the suspected windows list accordingly.
	if (suspectedCount) {
		total = 0;
		threshold /= matSize(setup->patternImage);
		currentMatch  = baseMatch;

		while (suspectedCount--) {
			// if lower bound is under threshold, save the current win in the updated win list            
			if ((matchDistance(currentMatch) = findMatrixDistance(setup->sourceImage,
												 			      matchY(currentMatch),
										 						  matchX(currentMatch),
																  setup->patternImage))
					<= threshold)
	  			baseMatch[total++] = *currentMatch;

			currentMatch++;
		}

		suspectedCount = total;
	}

	numOfMatches(setup) = suspectedCount;
}


#pragma inline_recursion(off)
#pragma inline_depth()

