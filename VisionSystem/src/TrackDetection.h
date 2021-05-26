#pragma once

#include <opencv2/opencv.hpp>
namespace track_detection {

	// Getters
	float GetExportDetectionWidthRatio();
	float GetExportDetectionHeightRatio();
	int GetTrackMiddle(int row);
	int GetTrackWidth(int row);
	int GetTrackHighestRow();
	int GetLaneStatus();
	int GetDistance(int row);
	void GetTrackPosition(int row, int& trackWidth, int& trackMiddle);

	// Run Track Detection combined with object detection
	void InizializeArrays();
	void DetectTracks(cv::Mat frameImage, bool draw);
	void OutputGlobalInformation();
	void DrawOverlayImages(cv::Mat outputImage, bool alwaysDrawWarningZone = false, bool contours = false);

	// Run Track Detection independently
	void RunTrackDetection();

	// Other functions
	int FindIfObjectIsInsideWarningZone(cv::Point leftPoint, cv::Point rightPoint, int OUTPUT_IMAGE_WIDTH, int OUTPUT_IMAGE_HEIGHT);
	void CalibrationAid(cv::Mat outputImage);

}

