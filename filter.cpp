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
#include <opencv2/opencv.hpp>

/*
 * Converts the source image to a grayscale-like image where each pixel's red channel is subtracted from 255.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be converted.
 *   - cv::Mat &dst: The destination image where the grayscale result will be stored.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int greyscale(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1; // return -1 if the source is empty
   }

   dst = src.clone(); // Copy the source image to destination

   // Iterate through each pixel
   for (int i = 0; i < src.rows; i++)
   {
      for (int j = 0; j < src.cols; j++)
      {
         // Access the pixel at (i, j)
         cv::Vec3b &pixel = dst.at<cv::Vec3b>(i, j);

         // Create a new grayscale value using the custom formula
         int grayValue = 255 - pixel[2]; // Subtract the red channel from 255

         // Apply the same value to all channels (B, G, R)
         pixel[0] = grayValue; // Blue channel
         pixel[1] = grayValue; // Green channel
         pixel[2] = grayValue; // Red channel
      }
   }

   return 0; // Success
}

/*
 * Applies a sepia-tone effect to the source image.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be transformed.
 *   - cv::Mat &dst: The destination image where the sepia effect will be applied and stored.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int sepia(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1; // Return error if source image is empty
   }

   // Create a destination image with the same size and type as the source
   dst.create(src.size(), src.type());

   // Iterate over each pixel and apply the sepia transformation
   for (int y = 0; y < src.rows; ++y)
   {
      for (int x = 0; x < src.cols; ++x)
      {
         // Access the original pixel values (R, G, B)
         cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
         uchar oriBlue = pixel[0];
         uchar oriGreen = pixel[1];
         uchar oriRed = pixel[2];

         // Calculate the new Blue, Green, and Red values using the original pixel values
         uchar newBlue = cv::saturate_cast<uchar>(0.272 * oriRed + 0.534 * oriGreen + 0.131 * oriBlue);
         uchar newGreen = cv::saturate_cast<uchar>(0.349 * oriRed + 0.686 * oriGreen + 0.168 * oriBlue);
         uchar newRed = cv::saturate_cast<uchar>(0.393 * oriRed + 0.769 * oriGreen + 0.189 * oriBlue);

         // Assign the new values to the destination image
         dst.at<cv::Vec3b>(y, x) = cv::Vec3b(newBlue, newGreen, newRed);
      }
   }
   return 0;
}

/*
 * Applies a 5x5 Gaussian blur filter to the source image.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be blurred.
 *   - cv::Mat &dst: The destination image where the blurred result will be stored.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1;
   }

   dst = src.clone();

   // 5x5 Gaussian kernel
   int kernel[5][5] = {
       {1, 2, 4, 2, 1},
       {2, 4, 8, 4, 2},
       {4, 8, 16, 8, 4},
       {2, 4, 8, 4, 2},
       {1, 2, 4, 2, 1}};
   int kernelSum = 128;

   // Loop through each pixel, skipping the outer two rows and columns
   for (int y = 2; y < src.rows - 2; ++y)
   {
      for (int x = 2; x < src.cols - 2; ++x)
      {
         int sumB = 0, sumG = 0, sumR = 0;

         // Apply the 5x5 filter to the neighborhood of the current pixel
         for (int ky = -2; ky <= 2; ++ky)
         {
            for (int kx = -2; kx <= 2; ++kx)
            {
               // Get the pixel at (x + kx, y + ky)
               cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);

               // Multiply the pixel's color channels by the corresponding kernel value
               int weight = kernel[ky + 2][kx + 2];
               sumB += pixel[0] * weight; // Blue channel
               sumG += pixel[1] * weight; // Green channel
               sumR += pixel[2] * weight; // Red channel
            }
         }

         // Obtain a normalized result by deviding the kernelSum
         dst.at<cv::Vec3b>(y, x)[0] = sumB / kernelSum;
         dst.at<cv::Vec3b>(y, x)[1] = sumG / kernelSum;
         dst.at<cv::Vec3b>(y, x)[2] = sumR / kernelSum;
      }
   }

   return 0;
}

/*
 * Applies a 5x5 Gaussian blur using separable filters for horizontal and vertical passes.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be blurred.
 *   - cv::Mat &dst: The destination image where the blurred result will be stored.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1;
   }

   dst = src.clone();
   cv::Mat temp = src.clone();

   // implement seperable filter
   int kernel[5] = {1, 2, 4, 2, 1};
   int kernelSum = 16;

   // horizontal filtering
   for (int y = 2; y < src.rows - 2; ++y)
   {
      // Get pointers to the current row in src and temp
      cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(y);
      cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(y);

      for (int x = 2; x < src.cols - 2; ++x)
      {
         int sumB = 0, sumG = 0, sumR = 0;

         // Apply horizontal 1x5 filter
         for (int k = -2; k <= 2; ++k)
         {
            sumB += srcRow[x + k][0] * kernel[k + 2];
            sumG += srcRow[x + k][1] * kernel[k + 2];
            sumR += srcRow[x + k][2] * kernel[k + 2];
         }

         // Normalization
         tempRow[x][0] = sumB / kernelSum;
         tempRow[x][1] = sumG / kernelSum;
         tempRow[x][2] = sumR / kernelSum;
      }
   }

   // Vertical pass (5x1)
   for (int y = 2; y < src.rows - 2; ++y)
   {
      // Get pointers to the current rows in temp and dst
      cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(y);

      for (int x = 2; x < src.cols - 2; ++x)
      {
         int sumB = 0, sumG = 0, sumR = 0;

         // Apply the vertical 5x1 filter
         for (int k = -2; k <= 2; ++k)
         {
            cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(y + k);

            sumB += tempRow[x][0] * kernel[k + 2];
            sumG += tempRow[x][1] * kernel[k + 2];
            sumR += tempRow[x][2] * kernel[k + 2];
         }

         // Normalize and store the result in the destination matrix
         dstRow[x][0] = sumB / kernelSum;
         dstRow[x][1] = sumG / kernelSum;
         dstRow[x][2] = sumR / kernelSum;
      }
   }

   return 0;
}

/*
 * Applies a 3x3 Sobel filter in the X direction (horizontal edge detection).
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to detect horizontal edges from.
 *   - cv::Mat &dst: The destination image where the Sobel X filtered result will be stored, using 16-bit signed integers.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1; // Return error if source image is empty
   }

   dst = cv::Mat::zeros(src.size(), CV_16SC3);

   // Apply horizontal kernel [-1, 0, 1]
   cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
   for (int y = 1; y < src.rows - 1; ++y)
   {
      for (int x = 1; x < src.cols - 1; ++x)
      {
         for (int c = 0; c < 3; ++c)
         {
            temp.at<cv::Vec3s>(y, x)[c] = src.at<cv::Vec3b>(y, x - 1)[c] * -1 +
                                          src.at<cv::Vec3b>(y, x + 1)[c] * 1;
         }
      }
   }

   // Apply vertical kernel [1, 2, 1]
   for (int y = 1; y < src.rows - 1; ++y)
   {
      for (int x = 1; x < src.cols - 1; ++x)
      {
         for (int c = 0; c < 3; ++c)
         {
            dst.at<cv::Vec3s>(y, x)[c] = temp.at<cv::Vec3s>(y - 1, x)[c] * 1 +
                                         temp.at<cv::Vec3s>(y, x)[c] * 2 +
                                         temp.at<cv::Vec3s>(y + 1, x)[c] * 1;
         }
      }
   }

   return 0;
}

/*
 * Applies a 3x3 Sobel filter in the Y direction (vertical edge detection).
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to detect vertical edges from.
 *   - cv::Mat &dst: The destination image where the Sobel Y filtered result will be stored, using 16-bit signed integers.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1; // Return error if source image is empty
   }

   dst = cv::Mat::zeros(src.size(), CV_16SC3);

   // Apply vertical kernel [1, 2, 1]
   cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
   for (int y = 1; y < src.rows - 1; ++y)
   {
      for (int x = 1; x < src.cols - 1; ++x)
      {
         for (int c = 0; c < 3; ++c)
         {
            temp.at<cv::Vec3s>(y, x)[c] = src.at<cv::Vec3b>(y - 1, x)[c] * 1 +
                                          src.at<cv::Vec3b>(y, x)[c] * 2 +
                                          src.at<cv::Vec3b>(y + 1, x)[c] * 1;
         }
      }
   }

   // Apply horizontal kernel [-1, 0, 1]
   for (int y = 1; y < src.rows - 1; ++y)
   {
      for (int x = 1; x < src.cols - 1; ++x)
      {
         for (int c = 0; c < 3; ++c)
         {
            dst.at<cv::Vec3s>(y, x)[c] = temp.at<cv::Vec3s>(y, x - 1)[c] * -1 +
                                         temp.at<cv::Vec3s>(y, x + 1)[c] * 1;
         }
      }
   }

   return 0;
}

/*
 * Computes the gradient magnitude from Sobel X and Sobel Y derivatives for each pixel.
 *
 * Arguments:
 *   - cv::Mat &sx: The source image containing Sobel X (horizontal) gradients (in 16-bit signed format).
 *   - cv::Mat &sy: The source image containing Sobel Y (vertical) gradients (in 16-bit signed format).
 *   - cv::Mat &dst: The destination image where the computed gradient magnitude (in 8-bit unsigned format) will be stored.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if either the sx or sy image is empty.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
   if (sx.empty() || sy.empty())
   {
      return -1; // Return error if any source image is empty
   }

   dst = cv::Mat::zeros(sx.size(), CV_8UC3);

   for (int y = 0; y < sx.rows; ++y)
   {
      for (int x = 0; x < sx.cols; ++x)
      {
         for (int c = 0; c < 3; ++c)
         {
            short sx_val = sx.at<cv::Vec3s>(y, x)[c];
            short sy_val = sy.at<cv::Vec3s>(y, x)[c];
            uchar magnitude = static_cast<uchar>(std::sqrt(sx_val * sx_val + sy_val * sy_val));
            dst.at<cv::Vec3b>(y, x)[c] = magnitude;
         }
      }
   }

   return 0;
}

/*
 * Applies a 5x5 Gaussian blur to the source image, followed by color quantization to reduce the number of color levels.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be blurred and quantized.
 *   - cv::Mat &dst: The destination image where the blurred and quantized result will be stored.
 *   - int levels: The number of quantization levels to reduce the color channels to.
 *
 * Return value:
 *   - Returns 0 on success, or -1 if the source image is empty.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
   if (src.empty())
   {
      return -1; // Return error if source image is empty
   }

   // Blur the image using a 5x5 Gaussian filter
   cv::Mat blurred;
   cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);

   // Quantize the image
   dst = blurred.clone();
   int b = 255 / levels;
   for (int y = 0; y < blurred.rows; ++y)
   {
      for (int x = 0; x < blurred.cols; ++x)
      {
         for (int c = 0; c < 3; ++c)
         {
            int x_val = blurred.at<cv::Vec3b>(y, x)[c];
            int xt = x_val / b;
            int xf = xt * b;
            dst.at<cv::Vec3b>(y, x)[c] = xf;
         }
      }
   }

   return 0;
}
