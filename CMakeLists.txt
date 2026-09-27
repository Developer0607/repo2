cmake_minimum_required(VERSION 3.16)
project(task2_fit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(task2_fit src/common/task2_fit/main.cpp)
add_executable(task1_image src/common/task1_image/main.cpp)
target_link_libraries(task2_fit PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(task2_fit PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task1_image PRIVATE Eigen3::Eigen opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)
target_include_directories(task1_image PRIVATE ${OpenCV_INCLUDE_DIRS})


if(MSVC)
  target_compile_options(task2_fit PRIVATE /W4)
else()
  target_compile_options(task2_fit PRIVATE -Wall -Wextra -Wpedantic )
endif()
if(MSVC)
  target_compile_options(task1_image PRIVATE /W4)
else()
  target_compile_options(task1_image PRIVATE -Wall -Wextra -Wpedantic)
endif()
