#pragma once

#ifndef OBJECT_DETECTION
#define OBJECT_DETECTION

#include <opencv2/opencv.hpp>
#include "YOLO-TensorRT.h"

namespace object_detection {

	// Struct for detected trains
	struct Train {
		bool detected = false;
		int sideOfTrack = 0;
		bool insideWarningZone = false;
	};

	// Setup
	void SetupNeuralNetworks();

	// Detect
	void DetectObjects(cv::Mat frameImage);

	// Process after track detection
	void ProcessObjectsAfterTrackDetection();
	void OutputGlobalInformation();
	void DrawBoundingBoxes(cv::Mat image);

	// Cleanup
	void Cleanup();

	// Other public functions
	Train GetDetectedTrain();
}

#endif