/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        matrix.c                                                     *
 *                                                                           *
 * description: Utilities for handling 2-D matrices with 32-bit values.      *
 *              A Matrix is allocated using it's dimensions (rows,cols).     *
 *              It is also possible to create a matrix from a given image.   *
 *              The Matrix data should be accessed only using the available  *
 *              macros.                                                      *
 *****************************************************************************/

#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

/*****************************************************************************
 * Allocates and returns an uninitialized Matrix of the given size.          *
 *****************************************************************************/
Matrix *allocMatrix(coordT rows, coordT cols)
{
	Matrix *mat;
	cellValueT *ptr;
	coordT row;

	mat = (Matrix *)malloc(sizeof(Matrix));
	if (!mat)
		exitWithError("ERROR in allocMatrix: can't allocate matrix.");

	mat->cellValues = ptr = (cellValueT *)malloc(rows * cols * sizeof(cellValueT));
	if (!mat->cellValues)
		exitWithError("ERROR in allocMatrix: can't allocate cellValues array.");

	mat->rows = rows;
	mat->cols = cols;
	mat->size = rows * cols;
	mat->rowsPtr = (cellValueT **)malloc(rows * sizeof(ptr));
	if (!mat->rowsPtr)
		exitWithError("ERROR in allocMatrix: can't allocate cellValues rows mapping.");

	for (row = 0; row < rows ; row++) {
		mat->rowsPtr[row] = ptr;
		ptr += cols;
	}

	return(mat);
}

/*****************************************************************************
 * Allocates and returns a matrix that contains the given image.             *
 *****************************************************************************/
Matrix * imageToMatrix(Image *image) {
	Matrix *mat = allocMatrix(imRows(image), imCols(image));
	pixelT *pixels = pixelsPtr(image);
	cellValueT *ptr = mat->cellValues;
	matrixSizeT i = matSize(mat);

	while (i--) 
 		*(ptr++) = *(pixels++);

	return mat;
}

/*****************************************************************************
 * Allocates and returns an image that contains the given matrix.            *
 * The values of the given matrix must be in the range 0..255                *
 *****************************************************************************/
Image * matrixToImage(Matrix *mat) {
	Image *image;
	pixelT *pixels;
	cellValueT *ptr = mat->cellValues;
	matrixSizeT i = matSize(mat);

	pixels = (pixelT *)malloc(matRows(mat) * matCols(mat) * sizeof(pixelT));
	if (!pixels)
		exitWithError("ERROR in matrixToImage: can't allocate pixels array");

	image = createImage(pixels, matRows(mat), matCols(mat));

	while (i--) 
		*(pixels++) = *(ptr++);

	return image;
}

/*****************************************************************************
 * Destroys the given matrix.                                                *
 *****************************************************************************/
void destroyMatrix(Matrix *mat) {
	free(mat->cellValues);
	free(mat->rowsPtr);
	free(mat);
}

/*****************************************************************************
 * Fills the given matrix with the given cellValue.                          *
 *****************************************************************************/
void fillMatrix(Matrix *mat, cellValueT cellValue) {
	cellValueT *ptr = mat->cellValues;
	matrixSizeT i = matSize(mat);

	while (i--) 
		*(ptr++) = cellValue;
}

/*****************************************************************************
 * Adds the two matrices together and returns the reuslt in the first one.   *
 * The two matrices must be of the same size.                                *
 *****************************************************************************/
void addMatrix(Matrix *mat1, Matrix *mat2) {
	cellValueT *ptr1 = mat1->cellValues, *ptr2 = mat2->cellValues; 
	matrixSizeT i = matSize(mat1);

	while (i--) 
		*(ptr1++) += *(ptr2++);
}

/*****************************************************************************
 * Subtracts the second matrix from the first one and returns the reuslt in  *
 * the first one.                                                            *
 * The two matrices must be of the same size.                                *
 *****************************************************************************/
void subMatrix(Matrix *mat1, Matrix *mat2) {
	cellValueT *ptr1 = mat1->cellValues, *ptr2 = mat2->cellValues; 
	matrixSizeT i = matSize(mat1);

	while (i--) 
		*(ptr1++) -= *(ptr2++);
}

/*****************************************************************************
 * Multiplies the two matrices together and returns the reuslt in the first  *
 * one.																		 *
 * The two matrices must be of the same size.								 *
 *****************************************************************************/
void mulMatrix(Matrix *mat1, Matrix *mat2) {
	cellValueT *ptr1 = mat1->cellValues, *ptr2 = mat2->cellValues; 
	matrixSizeT i = matSize(mat1);

	while (i--) 
		*(ptr1++) *= *(ptr2++);
}

/*****************************************************************************
 * Divides the first matrix by the second and returns the reuslt in the      *
 * first.																	 *
 * The two matrices must be of the same size.								 *
 *****************************************************************************/
void divMatrix(Matrix *mat1, Matrix *mat2) {
	cellValueT *ptr1 = mat1->cellValues, *ptr2 = mat2->cellValues; 
	matrixSizeT i = matSize(mat1);

	while (i--) 
		*(ptr1++) /= *(ptr2++);
}

/*****************************************************************************
 * Sums the given matrix and returns the result.						     *
 *****************************************************************************/
int sumMatrix(Matrix *mat) {
	cellValueT *ptr = mat->cellValues; 
	matrixSizeT i = matSize(mat);
	int sum = 0;

	while (i--) 
		sum += *(ptr++);

	return sum;
}

/*****************************************************************************
 * Returns the minimal cell value of the given matrix.                       *
 *****************************************************************************/
cellValueT minMatrix(Matrix *mat, coordT *y, coordT *x) {
	cellValueT *ptr = mat->cellValues; 
	matrixSizeT i = matSize(mat);
	int min = HIGHEST_CELL_VALUE;
	int updateCoord = (y != NULL && x != NULL);

	while (i--) {
		if (*ptr < min) {
			min = *ptr;
			if (updateCoord) {
				*y = i / matCols(mat);
				*x = i % matCols(mat);
			}
		}
		ptr++;
	}

	return min;
}

/*****************************************************************************
 * Returns the maximal cell value of the given matrix.                       *
 *****************************************************************************/
cellValueT maxMatrix(Matrix *mat, coordT *y, coordT *x) {
	cellValueT *ptr = mat->cellValues; 
	matrixSizeT i = matSize(mat);
	int max = LOWEST_CELL_VALUE;
	int updateCoord = (y != NULL && x != NULL);

	while (i--) {
		if (*ptr > max) {
			max = *ptr;
			if (updateCoord) {
				*y = i / matCols(mat);
				*x = i % matCols(mat);
			}
		}
		ptr++;
	}

	return max;
}

/*****************************************************************************
 * Multiplies each cell value of the given matrix with the given constant.   *
 *****************************************************************************/
void mulMatrixByConst(Matrix *mat, cellValueT constant) {
	cellValueT *ptr = mat->cellValues; 
	matrixSizeT i = matSize(mat);

	while (i--) 
		*(ptr++) *= constant;
}

/*****************************************************************************
 * Divides each cell value of the given matrix by the given constant.        *
 *****************************************************************************/
void divMatrixByConst(Matrix *mat, cellValueT constant) {
	cellValueT *ptr = mat->cellValues; 
	matrixSizeT i = matSize(mat);

	while (i--) 
		*(ptr++) /= constant;
}

/*****************************************************************************
 * Adds the given constant to each cell value of the given matrix.           *
 *****************************************************************************/
void addConstToMatrix(Matrix *mat, cellValueT constant) {
	cellValueT *ptr = mat->cellValues; 
	matrixSizeT i = matSize(mat);

	while (i--) 
		*(ptr++) += constant;
}

/*****************************************************************************
 * Returns the dot product of the two given matrices.                        *
 * The two matrices must be of the same size.                                *
 *****************************************************************************/
int dotProductMatrix(Matrix *mat1, Matrix *mat2) {
	cellValueT *ptr1 = mat1->cellValues, *ptr2 = mat2->cellValues; 
	matrixSizeT i = matSize(mat1);
	int sum = 0;

	while (i--) 
		sum += *(ptr1++) * *(ptr2++);

	return sum;
}

/*****************************************************************************
 * Creates a new matrix of size (rows, cols) filled with zeros, with the     *
 * given matrix copied into it at location (startRow, startCol).             *
 *****************************************************************************/
Matrix * zeroPadMatrix(Matrix *mat, coordT rows, coordT cols, coordT startRow, coordT startCol) {
	Matrix *newMat = allocMatrix(rows, cols);

	fillMatrix(newMat, 0);
	copyMatrixSegment(mat, newMat, 0, startRow, matRows(mat), 0, startCol, matCols(mat));

	return newMat;
}

/*****************************************************************************
 * Returns a new matrix containing the given matrix rotated by 180 degrees.  *
 *****************************************************************************/
Matrix * rotate180Matrix(Matrix *mat) {
	Matrix *newMat = allocMatrix(matRows(mat), matCols(mat));

	cellValueT *matPtr = matValPtr(mat),
			   *newMatPtr = matValPtr(newMat) + matSize(newMat) - 1;

	matrixSizeT count = matSize(mat);

	while (count--)
		*(newMatPtr--) = *(matPtr++);

	return newMat;
}


/*****************************************************************************
 * Convolutes the given matrix with the given mask and returns the result in *
 * a new matrix. The new matrix will be of the size:                         *
 * (rows(mat) - rows(mask) + 1, rows(mat) - rows(mask) + 1).                 *
 *****************************************************************************/
Matrix * convoluteMatrix(Matrix *mat, Matrix *mask) {
	coordT maskDif = matCols(mask) - 1;
	Matrix *dest = allocMatrix(matRows(mat) - maskDif, matCols(mat) - maskDif);

	cellValueT *startMatPtr = mat->cellValues,
			   *startMaskPtr = mask->cellValues + matSize(mask) - 1,
	           *destPtr = dest->cellValues;

	coordT colsSpace = matCols(mat) - matCols(mask);
	coordT destRows = matRows(dest);

	while (destRows--) {
		coordT destCols = matCols(dest);
		while (destCols--) {
			int sum = 0;
			coordT maskRows = matRows(mask);
			cellValueT *maskPtr = startMaskPtr;
			cellValueT *matPtr = startMatPtr++;
			while (maskRows--) {
				coordT maskCols = matCols(mask);
				while (maskCols--)
					sum += *(maskPtr--) * *(matPtr++);
				matPtr += colsSpace;
			}

			*(destPtr++) = sum;
		}

		startMatPtr += maskDif;
	}

	return dest;
}

/*****************************************************************************
 * Creates and returns another copy of the given matrix.                     *
 *****************************************************************************/
Matrix * cloneMatrix(Matrix *mat) {
	Matrix *newMat = allocMatrix(matRows(mat), matCols(mat));

	memcpy(matValPtr(newMat), matValPtr(mat), matRows(mat) * matCols(mat) * sizeof(cellValueT));

	return newMat;
}

/*****************************************************************************
 * Translates the matrix into the 0..255 range.                              *
 * All cell values are taken as absolute values.                             *
 *****************************************************************************/
void normalizeMatrix(Matrix *mat) {
	cellValueT lowest = minMatrix(mat, NULL, NULL);
	cellValueT highest = maxMatrix(mat, NULL, NULL);
	cellValueT *ptr = mat->cellValues;
	cellValueT val;
	double interval;
	matrixSizeT i = matSize(mat);

	interval = highest - lowest;

	// normalize
	ptr = mat->cellValues;
	i = matSize(mat);
	while (i--) {
		val = abs(*ptr);
		*(ptr++) = (cellValueT)(((double)(val - lowest) / interval) * 255.0);
	}
}

/*****************************************************************************
 * Copies a window in the source matrix into a window in the dest matrix.    *
 *****************************************************************************/
void copyMatrixSegment(Matrix *source, Matrix *dest,
					   coordT sourceStartRow, coordT destStartRow, coordT numOfRows,
					   coordT sourceStartCol, coordT destStartCol, coordT numOfCols) {

	cellValueT *sourcePtr = source->cellValues + matCols(source) * sourceStartRow + sourceStartCol;
	cellValueT *destPtr = dest->cellValues + matCols(dest) * destStartRow + destStartCol;
	coordT sourceDif = matCols(source) - numOfCols;
	coordT destDif = matCols(dest) - numOfCols;
	coordT cols;

	while (numOfRows--) {
		cols = numOfCols;
		while (cols--)
			*(destPtr++) = *(sourcePtr++);

		sourcePtr += sourceDif;
		destPtr += destDif;
	}
}

distanceT findMatrixDistance(Matrix *mat, coordT startRow, coordT startCol, Matrix *win)
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
