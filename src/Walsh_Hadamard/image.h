/*****************************************************************************
 *           Real Time Pattern Matching Using Projection Kernels             *
 *****************************************************************************
 * file:        image.h                                                      *
 *                                                                           *
 * description: Utilities for handling 8-bit grey level images.              *
 *              An Image is created using an array of pixels (each pixel     *
 *              indicates a grey level) and the image's dimensions.          *
 *              The pixel array should be in rows order, i.e. the top row    *
 *              from left to right, then the second row from left to right,  *
 *              etc.                                                         *
 *              The image data should be accessed only using the available   *
 *              macros.                                                      *
 ****************************************************************************/

#ifndef _image_h_
#define _image_h_

#ifdef __cplusplus
extern "C" {
#endif
    
#include "defs.h"

//////////////////////////////////// DATA TYPES ///////////////////////////////////////

typedef unsigned char pixelT;

typedef unsigned int coordT;

typedef struct {
	coordT rows, cols;
	pixelT *pixels; // P0,0..P0,width  P1,0..P1,width ... Pheigth,0..Pheight,width
} Image;


//////////////////////////////////////// MACROS //////////////////////////////////////

#define imRows(im)    (im->rows)
#define imCols(im)    (im->cols)

#define pixelsPtr(im) (im->pixels)

#define pixelVal(im, y, x) (im->pixels[y * im->cols + x])
#define pixelValD(im, i)   (im->pixels[i])

/////////////////////// PROTOTYPES OF PUBLIC FUNCTIONS /////////////////////////////

Image * createImage(pixelT *pixels, coordT rows, coordT cols);
void destroyImage(Image *image);
    
Image *createImageHead(coordT rows, coordT cols);
void destroyImageHead(Image *image);
void assignImageData(Image * image, pixelT * pixels);

void logImage(Image *image);

void copyImageSegment(Image *source, Image *dest,
					  coordT sourceStartRow, coordT destStartRow, coordT numOfRows,
					  coordT sourceStartCol, coordT destStartCol, coordT numOfCols);
#ifdef __cplusplus
}
#endif

#endif