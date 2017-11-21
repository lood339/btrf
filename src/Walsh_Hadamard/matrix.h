/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        matrix.h                                                     *
 *                                                                           *
 * description: Utilities for handling 2-D matrices with 32-bit values.      *
 *              A Matrix is allocated using it's dimensions (rows,cols).     *
 *              It is also possible to create a matrix from a given image.   *
 *              The Matrix data should be accessed only using the available  *
 *              macros.                                                      *
 *****************************************************************************/

#ifndef _matrix_h_
#define _matrix_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"
#include "image.h"

//////////////////////////////////// DATA TYPES ///////////////////////////////////////

typedef signed int cellValueT;

typedef unsigned int matrixSizeT;

typedef unsigned int distanceT;

typedef struct {
	coordT rows, cols;
	matrixSizeT size;
	cellValueT *cellValues;
	cellValueT **rowsPtr;
} Matrix;

/////////////////////////////////////// CONSTANTS ///////////////////////////////////

#define HIGHEST_CELL_VALUE 0x7fffffff

#define LOWEST_CELL_VALUE 0x80000000

//////////////////////////////////////// MACROS //////////////////////////////////////

#define matVal(mat, y, x)	(mat->rowsPtr[y][x])

#define matValD(mat, i)		(mat->cellValues[i])

#define matValPtr(mat)      (mat->cellValues)

#define matCols(mat)		(mat->cols)

#define matRows(mat)		(mat->rows)

#define matSize(mat)		(mat->size)


/////////////////////////////// inline functions //////////////////////////////////////

/*
Returns the distance between the given win and the segment from the given matrix
that starts in (startRow, startCol). The segment's size will be as the win's size.
*/
    distanceT findMatrixDistance(Matrix *mat, coordT startRow, coordT startCol, Matrix *win);
    /* modified as non-inline functions
{
	cellValueT *matPtr = matValPtr(mat) + startRow * matCols(mat) + startCol,
		   *winPtr = matValPtr(win);
	coordT rows = matRows(win), cols = matCols(win), col;
	coordT colsDif = matCols(mat) - cols;
	distanceT result = 0;
	while (rows--) {
		col = cols;
		while (col--) {
			distanceT dif = *(matPtr++) - *(winPtr++);
			result += dif * dif;
		}

		matPtr += colsDif;
	}

	return result;
}
     */

/////////////////////// PROTOTYPES OF PUBLIC FUNCTIONS /////////////////////////////

Matrix * allocMatrix(coordT rows, coordT cols);
void destroyMatrix(Matrix *mat);
Matrix * cloneMatrix(Matrix *mat);

Matrix * imageToMatrix(Image *image);
Image * matrixToImage(Matrix *mat);

void fillMatrix(Matrix *mat, cellValueT cellValue);

void normalizeMatrix(Matrix *mat);

void addMatrix(Matrix *mat1, Matrix *mat2);
void subMatrix(Matrix *mat1, Matrix *mat2);
void mulMatrix(Matrix *mat1, Matrix *mat2);
void divMatrix(Matrix *mat1, Matrix *mat2);
int dotProductMatrix(Matrix *mat1, Matrix *mat2);
void mulMatrixByConst(Matrix *mat, cellValueT constant);
void divMatrixByConst(Matrix *mat, cellValueT constant);
void addConstToMatrix(Matrix *mat, cellValueT constant);

int sumMatrix(Matrix *mat);
cellValueT minMatrix(Matrix *mat, coordT *y, coordT *x);
cellValueT maxMatrix(Matrix *mat, coordT *y, coordT *x);

Matrix * convoluteMatrix(Matrix *mat, Matrix *mask);
Matrix * zeroPadMatrix(Matrix *mat, coordT rows, coordT cols, coordT startRow, coordT startCol);
Matrix * rotate180Matrix(Matrix *mat);

void copyMatrixSegment(Matrix *source, Matrix *dest,
					   coordT sourceStartRow, coordT destStartRow, coordT numOfRows,
					   coordT sourceStartCol, coordT destStartCol, coordT numOfCols);

#ifdef __cplusplus
}
#endif

#endif