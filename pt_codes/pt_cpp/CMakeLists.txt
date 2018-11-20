cmake_minimum_required(VERSION 3.6)
project(ptcpp)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_PREFIX_PATH /media/jintain/sg/ai/tools/tfboys/pt_codes/pt_cpp/libtorch)
set(CMAKE_PREFIX_PATH ./libtorch)


include_directories(
        libtorch/include
)

link_directories(
        libtorch/lib
)

find_package(Torch REQUIRED)
if (NOT Torch_FOUND)
    message(FATAL "Torch not found, you should set CMAKE_PREFIX_PATH mannually.")
endif ()

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL "opencv needed but not found.")
endif ()

add_executable(ptcpp classifier.cpp)
target_link_libraries(ptcpp ${OpenCV_LIBS} ${TORCH_LIBRARIES})