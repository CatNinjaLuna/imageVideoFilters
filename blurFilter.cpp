#include <opencv2/opencv.hpp>

int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
   if (src.empty())
   {
      return -1;
   }

   // Copy the source image to the destination image
   dst = src.clone();

   // Define the 5x5 Gaussian kernel
   int kernel[5][5] = {
       {1, 2, 4, 2, 1},
       {2, 4, 8, 4, 2},
       {4, 8, 16, 8, 4},
       {2, 4, 8, 4, 2},
       {1, 2, 4, 2, 1}};

   // Traverse each pixel
   for (int y = 2; y < src.rows - 2; ++y)
   {
      for (int x = 2; x < src.cols - 2; ++x)
      {
         int sumB = 0, sumG = 0, sumR = 0;
         int weightSum = 0;

         // Apply the 5x5 kernel
         for (int ky = -2; ky <= 2; ++ky)
         {
            for (int kx = -2; kx <= 2; ++kx)
            {
               // The at method is used to access the pixel value
               cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
               int weight = kernel[ky + 2][kx + 2];

               sumB += pixel[0] * weight;
               sumG += pixel[1] * weight;
               sumR += pixel[2] * weight;
               weightSum += weight;
            }
         }

         // Normalization of the sum
         dst.at<cv::Vec3b>(y, x) = cv::Vec3b(sumB / weightSum, sumG / weightSum, sumR / weightSum);
      }
   }
   return 0;
}

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

   return 0; // Return 0 on success
}