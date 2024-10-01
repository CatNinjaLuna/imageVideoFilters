
/*
Name : Carolina Li
Date : Sep /29 / 2024
Purpose : The objecive of this file contains several elements:
   1. Display live video, and save an image to a file
   2. Display greyscale of live video
   3. Display alternative greyscale live video
   4. Implement a Sepia tone filter
   5. Implement a naive 5x5 blur fliter
   6. Implement a faster naive 5x5 blur fliter
   7. implement a gradient magnitude filter
   8. implement a quantization filter
   9. implement a negation effect on the image
   10. implement a filter that makes face colorful and convert other area into greyscale
   11. implement a filter to blur the image outside of found faces

*/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"

// Converts the source image to grayscale.
int greyscale(cv::Mat &src, cv::Mat &dst);
// Applies a sepia-tone effect to the source image.
int sepia(cv::Mat &src, cv::Mat &dst);
// Applies a 5x5 Gaussian blur filter.
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
    // Applies a 5x5 Gaussian blur using separable filters.
    int blur5x5_2(cv::Mat &src, cv::Mat &dst);
// Applies a 3x3 Sobel filter to detect horizontal edges.
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
// Applies a 3x3 Sobel filter to detect vertical edges.
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
// Computes the gradient magnitude from the horizontal and vertical Sobel derivatives.
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
// Blurs the source image and then quantizes it into a fixed number of levels.
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/*
 * Applies a negative effect to the source image by inverting all pixel values.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) to be inverted.
 *   - cv::Mat &dst: The destination image where the negative effect will be stored.
 *
 * Return value:
 *   - This function does not return a value, but the negative effect is applied to the destination image.
 */
void applyNegativeEffect(cv::Mat &src, cv::Mat &dst)
{
   cv::bitwise_not(src, dst);
}

/*
 * Makes the face regions in the source image colorful while converting the rest of the image to grayscale.
 *
 * Arguments:
 *   - cv::Mat &src: The source image (in BGR format) where faces will remain in color.
 *   - cv::Mat &dst: The destination image where the grayscale and colorful face effect will be stored.
 *   - std::vector<cv::Rect> &faces: A vector containing the bounding boxes of detected faces in the image.
 *
 * Return value:
 *   - This function does not return a value, but the effect is applied to the destination image.
 */
void applyFaceColorfulEffect(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces)
{
   cv::Mat gray;
   cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
   cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);

   for (const auto &face : faces)
   {
      src(face).copyTo(dst(face));
   }
}

int main(int argc, char *argv[])
{
   cv::VideoCapture *capdev;

   // open the video device
   capdev = new cv::VideoCapture(0);
   if (!capdev->isOpened())
   {
      std::cerr << "Unable to open video device" << std::endl;
      return -1;
   }

   // Load the Haar cascade file for face detection
   cv::CascadeClassifier face_cascade;
   if (!face_cascade.load(FACE_CASCADE_FILE))
   {
      std::cerr << "Error loading Haar cascade file" << std::endl;
      return -1;
   }

   // get some properties of the image
   cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                 (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
   std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

   cv::namedWindow("Video", 1); // identifies a window
   cv::Mat frame, altGreyFrame;
   cv::Mat sobelxFrame;
   cv::Mat displayFrame;
   cv::Mat sobelyFrame;
   cv::Mat magnitudeFrame;
   cv::Mat blurQuantizeFrame;
   cv::Mat negativeFrame;
   cv::Mat faceColorfulFrame;

   bool isGrayscale = false;
   bool isAltGrayscale = false;
   bool isSepia = false;
   bool isblurred = false;
   bool sobelx = false;
   bool sobely = false;
   bool isMagnitude = false;
   bool isQuantized = false;
   bool isFaceDetected = false;
   bool isNegative = false;
   bool isFaceColorful = false;
   bool isBlurOutsideFaces = false;

   for (;;)
   {
      *capdev >> frame; // get a new frame from the camera

      if (frame.empty())
      {
         std::cerr << "Frame is empty" << std::endl;
         break;
      }

      // check for key press
      char lastkey = 0;
      char key = cv::waitKey(10);
      if (key != -1)
      {
         lastkey = key; // Update the last key press
         std::cout << "Key pressed: " << lastkey << std::endl;
      }

      if (isGrayscale)
      {
         cv::Mat greyFrame;
         cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
         cv::imshow("Video", greyFrame);
      }
      else if (isAltGrayscale)
      {
         // Use alternative grayscale function
         if (greyscale(frame, altGreyFrame) == 0)
         {
            cv::imshow("Video", altGreyFrame);
         }
         else
         {
            std::cerr << "Error processing the frame" << std::endl;
         }
      }
      else if (isSepia)
      {
         cv::Mat sepiaFrame;
         sepia(frame, sepiaFrame);
         cv::imshow("Video", sepiaFrame);
         std::cout << "Displaying sepia frame" << std::endl;
      }

      else if (isblurred)
      {
         cv::Mat blurredFrame;
         blur5x5_2(frame, blurredFrame);
         cv::imshow("Video", blurredFrame);
         std::cout << "Displaying blurred frame" << std::endl;
      }

      else if (sobelx)
      {
         sobelX3x3(frame, sobelxFrame);
         cv::convertScaleAbs(sobelxFrame, displayFrame);
         cv::imshow("Video", displayFrame);
         // cv::imshow("Video", sobelxFrame);
         std::cout << "Displaying sobelxFrame" << std::endl;
      }

      else if (sobely)
      {
         sobelY3x3(frame, sobelyFrame);
         cv::convertScaleAbs(sobelyFrame, displayFrame);
         cv::imshow("Video", displayFrame);
         // cv::imshow("Video", sobelyFrame);
         std::cout << "Displaying sobelyFrame" << std::endl;
      }
      else if (isMagnitude)
      {
         magnitude(sobelxFrame, sobelyFrame, magnitudeFrame);
         cv::imshow("Video", magnitudeFrame);
         std::cout << "Displaying gradient magnitude frame" << std::endl;
      }
      else if (isQuantized)
      {
         blurQuantize(frame, blurQuantizeFrame, 10);
         cv::imshow("Video", blurQuantizeFrame);
         std::cout << "Displaying blurred and quantized frame" << std::endl;
      }
      else if (isFaceDetected)
      {
         std::vector<cv::Rect> faces;
         cv::Mat grayFrame;
         cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
         if (detectFaces(grayFrame, faces) == 0)
         {
            drawBoxes(frame, faces);
            cv::imshow("Video", frame);
            std::cout << "Displaying face detection frame" << std::endl;
         }
      }

      else if (isNegative)
      {
         applyNegativeEffect(frame, negativeFrame);
         cv::imshow("Video", negativeFrame);
         std::cout << "Displaying negative frame" << std::endl;
      }

      else if (isFaceColorful)
      {
         std::vector<cv::Rect> faces;
         cv::Mat grayFrame;
         cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
         if (detectFaces(grayFrame, faces) == 0)
         {
            applyFaceColorfulEffect(frame, faceColorfulFrame, faces);
            cv::imshow("Video", faceColorfulFrame);
            std::cout << "Displaying face colorful frame" << std::endl;
         }
      }

      else if (isBlurOutsideFaces)
      {
         std::vector<cv::Rect> faces;
         cv::Mat grayFrame;
         cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
         if (detectFaces(grayFrame, faces) == 0)
         {
            // Create a mask for the faces
            cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
            for (const auto &face : faces)
            {
               cv::rectangle(mask, face, cv::Scalar(255, 255, 255), cv::FILLED);
            }

            // Blur the entire frame
            cv::Mat blurredFrame;
            blur5x5_2(frame, blurredFrame);

            // Combine the blurred frame with the original frame using the mask
            cv::Mat resultFrame = frame.clone();
            blurredFrame.copyTo(resultFrame, ~mask);
            frame.copyTo(resultFrame, mask);

            cv::imshow("Video", resultFrame);
            std::cout << "Displaying frame with blur outside faces" << std::endl;
         }
      }

      else
      {
         cv::imshow("Video", frame);
      }

      if (lastkey == 'q')
      {
         break;
      }
      if (lastkey == 's')
      {
         cv::imwrite("captured_image.jpg", frame);
      }
      if (lastkey == 'g')
      {
         isGrayscale = true;
         // isAltGrayscale = false;
         // isSepia = false;
         std::cout << "Switching to standard grayscale mode" << std::endl;
      }
      if (lastkey == 'c')
      {
         isGrayscale = false;
         // isAltGrayscale = false;
         // isSepia = false;
         std::cout << "Switching to color mode" << std::endl;
      }

      if (lastkey == 'h')
      {
         isAltGrayscale = true;
         isGrayscale = false;
         isSepia = false;
         std::cout << "Switching to alternative grayscale mode" << std::endl;
      }
      if (lastkey == 'j')
      {
         isSepia = true;
         isGrayscale = false;
         isAltGrayscale = false;
         std::cout << "Switching to sepia mode" << std::endl;
      }

      if (lastkey == 'b')
      {
         // call the blur version of the video stream
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = true;
         std::cout << "Switching to blurred mode" << std::endl;
      }

      if (lastkey == 'x')
      {
         // call the blur version of the video stream
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = true;

         cv::convertScaleAbs(sobelYFrame, displayFrame);
         cv::imshow("Video", displayFrame);
         std::cout << "Switching to sobelx mode" << std::endl;
      }

      if (lastkey == 'y')
      {
         // call the blur version of the video stream
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobely = true;
         std::cout << "Switching to sobely mode" << std::endl;
      }

      if (lastkey == 'm')
      {
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = true;
         sobely = true;
         isMagnitude = true;

         std::cout
             << "Switching to magnitude mode" << std::endl;
      }

      if (lastkey == 'l')
      {
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = false;
         sobely = false;
         isMagnitude = false;
         isQuantized = true;

         std::cout
             << "Switching to quantization mode" << std::endl;
      }

      if (lastkey == 'f')
      {
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = false;
         sobely = false;
         isMagnitude = false;
         isQuantized = false;
         isFaceDetected = true;

         std::cout
             << "Switching to face detection mode" << std::endl;
      }

      if (lastkey == 'n')
      {
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = false;
         sobely = false;
         isMagnitude = false;
         isQuantized = false;
         isFaceDetected = false;
         isNegative = true;

         std::cout
             << "Switching to negative mode" << std::endl;
      }

      if (lastkey == 'a')
      {
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = false;
         sobely = false;
         isMagnitude = false;
         isQuantized = false;
         isFaceDetected = false;
         isNegative = false;
         isFaceColorful = true;

         std::cout
             << "Switching to colorful mode" << std::endl;
      }

      if (lastkey == 'z')
      {
         isSepia = false;
         isAltGrayscale = false;
         isGrayscale = false;
         isblurred = false;
         sobelx = false;
         sobely = false;
         isMagnitude = false;
         isQuantized = false;
         isFaceDetected = false;
         isNegative = false;
         isFaceColorful = false;
         isBlurOutsideFaces = true;

         std::cout
             << "Switching to blur outside face mode" << std::endl;
      }

      delete capdev;
      return 0;
   }
