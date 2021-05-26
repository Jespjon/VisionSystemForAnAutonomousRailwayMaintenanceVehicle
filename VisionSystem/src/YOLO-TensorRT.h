#pragma once

#include <opencv2/opencv.hpp>

// Only run TenorRT on linux (possible to change this, but than the TensorRT model must be optimized for the new system) 
#ifdef linux
#define TENSORRT
#endif

struct BoundingBox {
    int x, y, w, h;       // (x,y) - top-left corner in pixels, (w, h) - width & height of bounded box in pixels
    float confidence;     // probability that the box contains an object
    int classId;
    float classProbability; // probability of the class
    int distance = -1;      // distance to the object in meters (can be calculated in ObjectDetection.cpp)

    // Constructors
    BoundingBox() {}
    BoundingBox(int x, int y, int w, int h, float confidence, int classId, float classProbability)
        : x(x), y(y), w(w), h(h), confidence(confidence), classId(classId), classProbability(classProbability)
    { }
};

namespace yolo_tensorrt {
    // Build
	void BuildNetwork(std::string enginePath, int nClasses, float confidenceThreshold, int imageWidth, int imageHeight);
	
    // Detect
    std::vector<BoundingBox> Detect(cv::Mat frame);
    
    // Destroy
    void Destroy();
	
}