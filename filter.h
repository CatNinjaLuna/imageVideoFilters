/*
Name : Carolina Li
Date : Sep /29 / 2024
Purpose : The objecive of this file contains several elements:
   1. Create a function to cconvert RGB to greyscale
   2. Create a function to apply a Sepia tone filter
   3. Create a Sepia tone filter
   4. Create a naive 5x5 blur fliter
   5. Create a faster naive 5x5 blur fliter
   6. Create a gradient magnitude filter
   7. Create a filter that applying a quantization filter

*/
#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

/*
 * Converts the source image to a grayscale-like image where each pixel's red channel is subtracted from 255.
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be converted.
 *   - cv::Mat &dst: The destination image where the grayscale result will be stored.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a sepia-tone effect to the source image.
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be transformed.
 *   - cv::Mat &dst: The destination image where the sepia effect will be applied and stored.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur filter to the source image.
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be blurred.
 *   - cv::Mat &dst: The destination image where the blurred result will be stored.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur using separable filters for horizontal and vertical passes.
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be blurred.
 *   - cv::Mat &dst: The destination image where the blurred result will be stored.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 3x3 Sobel filter in the X direction (horizontal edge detection).
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to detect horizontal edges from.
 *   - cv::Mat &dst: The destination image where the Sobel X filtered result will be stored.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 3x3 Sobel filter in the Y direction (vertical edge detection).
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to detect vertical edges from.
 *   - cv::Mat &dst: The destination image where the Sobel Y filtered result will be stored.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/*
 * Computes the gradient magnitude from Sobel X and Sobel Y derivatives for each pixel.
 * Arguments:
 *   - cv::Mat &sx: The source image containing Sobel X (horizontal) gradients.
 *   - cv::Mat &sy: The source image containing Sobel Y (vertical) gradients.
 *   - cv::Mat &dst: The destination image where the computed gradient magnitude will be stored.
 * Return value:
 *   - Returns 0 on success, or -1 if either the sx or sy image is empty.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur to the source image, followed by color quantization to reduce the number of color levels.
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be blurred and quantized.
 *   - cv::Mat &dst: The destination image where the blurred and quantized result will be stored.
 *   - int levels: The number of quantization levels to reduce the color channels to.
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

#endif // FILTER_H
