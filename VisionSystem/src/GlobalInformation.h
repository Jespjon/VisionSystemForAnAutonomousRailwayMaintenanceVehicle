#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// GlobalEnumerationConstants.h <-- including the enumeration constants for some variables

// Struct for detected vehicles and persons
struct Object {
	int x; // Top left corner
	int y; // Top left corner
	int width;
	int height;
	int objectType; // of type OBJECTS
	int insideWarningZone; // of type WARNING_ZONE
	int distance;
};

struct GlobalInformation {
	int fps;
	int frameNumber;

	int maxTrackViewDistance;
	int minTrackViewDistance;

	int laneStatus; // of type LANE_STATUS

	bool switchDetected;
	int switchedLaneStatus; // of type SWITCH
	int incomingSwitchStatus; // of type SWITCH

	int currentMainSignalMessage; // of type MAIN_SIGNAL_MESSAGE
	int nextMainSignalMessage; // of type MAIN_SIGNAL_MESSAGE
	int currentRoadCrossingMessage; // of type ROAD_CROSSING_MESSAGE
	int nextRoadCrossingMessage; // of type ROAD_CROSSING_MESSAGE

	bool roadCrossingDetected;
	bool roadCrossingBarriersDetected;
	int distanceToRoadCrossing;

	int maxSpeedLimit;

	bool ATCSpeedUpSignDetected;
	bool ATCSpeedDownSignDetected;
	bool warningSignDetected;
	bool distantSignalSignDetected;
	bool signVDetected;

	int distanceToMainSignal;
	int distanceToRoadCrossingSignal;
	int distanceToSpeedSign;

	int leftPolesCounter;
	int rightPolesCounter;

	std::vector<Object> objectList;

	std::string errors;

	cv::Mat image;
};

inline GlobalInformation globalInformation;