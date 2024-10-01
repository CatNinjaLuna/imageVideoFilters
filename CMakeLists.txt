# cmake_minimum_required(VERSION 2.8)
# project( DisplayImage )
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( DisplayImage DisplayImage.cpp )
# target_link_libraries( DisplayImage ${OpenCV_LIBS} )

# # add your source file
# set(MY_SOURCE_FILES
#    src/DisplayImage.cpp
#    set(CMAKE_C_COMPILER "/usr/bin/clang")
#    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

#    #src/othercppfiles
# )

# #add_executable( DisplayImage ${MY_SOURCE_FILES})
# target_link_libraries( DisplayImage ${OpenCV_LIBS})

cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add your source file
set(MY_SOURCE_FILES
   src/DisplayImage.cpp
   #src/othercppfiles
)
add_executable( DisplayImage ${MY_SOURCE_FILES})
target_link_libraries( DisplayImage ${OpenCV_LIBS})