# CMakeList.txt : CMake project for VisionNode
# Works for Windows 10 with Darknet and OpenCV installed.
# Works for Linux (Jetson Nano) with TensorRT and OpenCV (with GPU support) installed. 

# About the programs:
# TrackDetection: Run track detection stand alone to detect tracks in a video. 
# Vision: Run track detection and object detection combined on either a video or with a camera.
# VisionLibrary: Runtime Library for the vision program to be used in a ROS program (Robot Operating System).
# DrawLocalizationMap: Draw a localization map from localization data (from a localization program in ROS).
# ExplainYOLOImageGenerator: Generate images explaining how YOLO works.


cmake_minimum_required (VERSION 3.8)

project ("VisionNode")

set (CMAKE_CXX_STANDARD 17)


# Find packages
find_package(OpenCV REQUIRED) # Variable OpenCV_DIR must be declared to <path>/opencv/build/ on Windows
if (WIN32)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif (WIN32)
if (UNIX)
    find_package(Threads REQUIRED)
endif(UNIX)


# Set variables
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/")
if (WIN32)
    set(DARKNET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Dependencies/darknet/")  # path to Darknet installation
    set(VCPKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg/installed/x64-windows/") # some libraries needed by darknet
endif (WIN32)
if (UNIX)
    #set(DARKNET_DIR "/home/jetson/darknet/") # path to Darknet installation
    set(DARKNET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Dependencies/darknet/")  # path to Darknet installation
    set(TENSORRT_LIBS "/usr/lib/aarch64-linux-gnu/") # path to TensorRT library files
endif (UNIX)


# Include directories
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${SRC})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/yolo-plugin/") # include yolo_layer.h and yolo_v2_class.hpp
if (WIN32)
    include_directories("${DARKNET_DIR}include/") # include darknet.h
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg/installed/x64-windows/include") # include header files for libraries needed by darknet
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-6.0.1.5/include/") # path to TensorRT/include
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/") # path to CUDA/include
endif (WIN32)
if (UNIX)
    include_directories("/usr/local/cuda-10.2/include") # path to CUDA/include
    include_directories("${DARKNET_DIR}include/") # include darknet.h and yolo_v2_class.hpp
endif (UNIX)


# Add executables

# Run Track Detection independently
add_executable (TrackDetection "src/TrackDetection.cpp" "src/RunTrackDetection.cpp" "src/LoadParameters.cpp")

# Draw localization map from localization data
add_executable (DrawLocalizationMap "src/DrawLocalizationMap.cpp")

if (WIN32)
    add_executable (Vision "src/RunVisionSystem.cpp" "src/VisionSystem.cpp" "src/TrackDetection.cpp" "src/ObjectDetection.cpp" "src/LoadParameters.cpp")
    add_executable (ExplainYOLOImageGenerator "src/ExplainYOLOImageGenerator.cpp")
endif (WIN32)
if (UNIX)
    add_executable (Vision "src/RunVisionSystem.cpp" "src/VisionSystem.cpp" "src/TrackDetection.cpp" "src/ObjectDetection.cpp" "src/YOLO-TensorRT.cpp" "src/LoadParameters.cpp")
    add_library (VisionLibrary SHARED "src/VisionSystem.cpp" "src/TrackDetection.cpp" "src/ObjectDetection.cpp" "src/YOLO-TensorRT.cpp" "src/LoadParameters.cpp")
endif(UNIX)

# Link libraries
target_link_libraries(TrackDetection PUBLIC ${OpenCV_LIBS} "-pthread")
target_link_libraries(DrawLocalizationMap PUBLIC ${OpenCV_LIBS})

if (WIN32)
    # Different darknet libraries depending on build type (Release/Debug)
    IF(CMAKE_BUILD_TYPE MATCHES Release)
        target_link_libraries(ExplainYOLOImageGenerator PUBLIC ${OpenCV_LIBS} "${DARKNET_DIR}darknet.lib")
        target_link_libraries(Vision PUBLIC ${OpenCV_LIBS} "${DARKNET_DIR}darknet.lib")
    ELSE()
        target_link_libraries(ExplainYOLOImageGenerator PUBLIC ${OpenCV_LIBS} "${DARKNET_DIR}darknet.lib")
        target_link_libraries(Vision PUBLIC ${OpenCV_LIBS} "${DARKNET_DIR}darknetd.lib")
    ENDIF()
endif (WIN32)
if (UNIX)
    target_link_libraries(Vision PUBLIC ${OpenCV_LIBS} "-pthread" "-lcudart" "${TENSORRT_LIBS}libnvinfer.so" "${TENSORRT_LIBS}libnvinfer_plugin.so" "${CMAKE_CURRENT_SOURCE_DIR}/yolo-plugin/libyolo_layer.so" ) # "${DARKNET_DIR}libdarknet.so"
    target_link_libraries(VisionLibrary PUBLIC ${OpenCV_LIBS} "-pthread" "-lcudart" "${TENSORRT_LIBS}libnvinfer.so" "${TENSORRT_LIBS}libnvinfer_plugin.so" "${CMAKE_CURRENT_SOURCE_DIR}/yolo-plugin/libyolo_layer.so" ) # "${DARKNET_DIR}libdarknet.so"
endif (UNIX)