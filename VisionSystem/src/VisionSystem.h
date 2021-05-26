#pragma once
#include "GlobalInformation.h"
#include "opencv2/opencv.hpp"

namespace vision {
	// Setup
	void SetupVision(bool windowLess = false, bool useROSCameras_ = false);

	// Detect
	bool VisionDetect(bool windowLess = false, cv::Mat nextFrame_ = cv::Mat()); // returns false if load image failed

	// Destroy
	void VisionDestroy();

	// Getters
	GlobalInformation GetGlobalInformation(); // returns struct with the detection results

	// Setters
	void SetDrivingMode(int drivingMode);
	void SetDrawVisionOutput(bool newShow);
}