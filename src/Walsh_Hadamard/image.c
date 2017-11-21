/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        image.c                                                      *
 *                                                                           *
 * description: Utilities for handling 8-bit grey level images.              *
 *              An Image is created using an array of pixels (each pixel     *
 *              indicates a grey level) and the image's dimensions.          *
 *              The pixel array should be in rows order, i.e. the top row    *
 *              from left to right, then the second row from left to right,  *
 *              etc.                                                         *
 *              The image data should be accessed only using the available   *
 *              macros.                                                      *
 *****************************************************************************/

#include "image.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/*****************************************************************************
 * creates and returns an Image of the given size, with the given pixels     *
 * array.                                                                    *
 *****************************************************************************/
Image *createImage(pixelT *pixels, coordT rows, coordT cols)
{
	Image *image;

	image = (Image *)malloc(sizeof(Image));
	if (!image)
		exitWithError("ERROR in allocImage: can't allocate image.");

	imRows(image) = rows;
	imCols(image) = cols;
	pixelsPtr(image) = pixels;

	return(image);
}

/* creat image head
 */
Image *createImageHead(coordT rows, coordT cols)
{
    Image *image;
    
    image = (Image *)malloc(sizeof(Image));
    if (!image)
        exitWithError("ERROR in allocImage: can't allocate image.");
    
    imRows(image) = rows;
    imCols(image) = cols;
    
    return image;
}

void destroyImageHead(Image *image) {    
    free(image);
}

/* assign image content
 */
void assignImageData(Image * image, pixelT * pixels)
{
    assert(image != NULL);
    assert(image->rows != 0 && image->cols != 0);
    pixelsPtr(image) = pixels;
}

/*****************************************************************************
 * Destroys the given Image.                                                 *
 *****************************************************************************/
void destroyImage(Image *image) {
	free(pixelsPtr(image));
	free(image);
}

/*****************************************************************************
 * Copies a window in the source image into a window in the dest image.      *
 *****************************************************************************/
void copyImageSegment(Image *source, Image *dest,
			   	      coordT sourceStartRow, coordT destStartRow, coordT numOfRows,
					  coordT sourceStartCol, coordT destStartCol, coordT numOfCols) {

	pixelT *sourcePtr = pixelsPtr(source) + imCols(source) * sourceStartRow + sourceStartCol;
	pixelT *destPtr = pixelsPtr(dest) + imCols(dest) * destStartRow + destStartCol;
	coordT sourceDif = imCols(source) - numOfCols;
	coordT destDif = imCols(dest) - numOfCols;
	coordT cols;

	while (numOfRows--) {
		cols = numOfCols;
		while (cols--)
			*(destPtr++) = *(sourcePtr++);

		sourcePtr += sourceDif;
		destPtr += destDif;
	}
}

/*****************************************************************************
 * Performes log2 on the given image.                                        *
 *****************************************************************************/
void logImage(Image *image) {
	double val;
	double logOf2 = log10(2.0);
	pixelT *ptr = pixelsPtr(image);
	unsigned int i = imRows(image) * imCols(image);

	while (i--) {
		val = log10(*ptr) / logOf2;
		*(ptr++) = (pixelT)val << 5;
	}
}
