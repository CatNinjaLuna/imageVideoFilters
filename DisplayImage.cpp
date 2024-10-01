#include <cstdio>
#include <cstring>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
int main(int argc, char *argv[])
{
   // Check if filename is provided
   if (argc < 2)
   {
      std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
      return -1;
   }

   // Read the image
   cv::Mat image = cv::imread(argv[1]);
   if (image.empty())
   {
      std::cout << "Could not open or find the image!" << std::endl;
      return -1;
   }

   // Display the image
   cv::imshow("Display Image", image);

   // Wait for a key press
   while (true)
   {
      char key = cv::waitKey(10);
      if (key == 'q')
         break; // Quit on 'q' key press
   }
   return 0;
}
