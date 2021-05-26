// About
/*
* VisionSystem.cpp
* 
* Vision system for detecting tracks and objects in a camera/video frame.
* 
* First setup the vision system with SetupVision().
* Run VisionDetect() to detect tracks and objects in a new frame. 
* Automatically loads the frame from a video or camera.
* Get the vision output, i.e. a struct with variables (GlobalInformation.h) with GetGlobalInformation().
* Exit with VisionDestroy(). 
*/

// Structure
/*
* Parameters
* 
* Utilities Functions
* 
* Draw Functions
* 
* Load Frame
* 
* Vision Detect <-- Main section
* 
* Setup
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>
#include <thread>
#include <future>
#include <atomic>
#include "yolo_v2_class.hpp"

#include "SharedParameters.h"
#include "LoadParameters.h"
#include "GlobalEnumerationConstants.h"
#include "GlobalInformation.h"
#include "ObjectDetection.h"
#include "TrackDetection.h"

using std::cout;
using std::endl;
using std::string;


namespace vision {

	// ######## ######## ######## ######## Parameters ######## ######## ######## ######## 
#pragma region Parameters

	// Main parameters
	bool drawWarningZoneLines = false; // Draw contour lines of the warning zone instead of the filled warning zone.
	bool drawCalibrationAid = false; // Draw a circle for the horizon height and lines for the track width
	bool useROSCameras;

	// Camera settings
	int captureWidth = 1920; // TODO: same as output sizes?
	int captureHeight = 1080;
	int displayWidth = 1920;
	int displayHeight = 1080;
	int framerate = 30;
	int flipMethod = 0; // 2 for raspberry pi camera, 0 for arducam

	// Global variables
	int framesPerSecond = 0;
	int outputImageArea = OUTPUT_IMAGE_HEIGHT * OUTPUT_IMAGE_HEIGHT;
	int bottomPixelsToCrop = 0; // Pixels to crop at the bottom
	string windowName;
	cv::VideoWriter video;
	cv::Size outputImageSize;
	cv::Mat outputImage;
	cv::Mat nextOutputImage;
	cv::Mat saveOutputImage;
	cv::Mat frameImage;
	cv::Mat nextFrameImage;
	cv::Mat infoBoxOverlayImage;
	cv::VideoCapture cap;
	cv::VideoCapture capForwardCamera;
	cv::VideoCapture capBackwardCamera;

	// Threads
	std::atomic<bool> pause(false);
	std::atomic<bool> workDone(false);
	std::thread commandInputWorkerThread;
	std::thread saveVideoWorkerThread;

	// Drawing
	float textSize = 0.7;
	cv::Scalar textColor = cv::Scalar(255, 255, 255);	// White
	float textThickness = 1.0;
	int lineType = cv::LINE_AA;
	int textHorizontalShift = 10;

#pragma endregion Parameters


	// ######## ######## ######## ######## Utilities Functions ######## ######## ######## ######## 
#pragma region Utilities Functions

	// Camera setup, used if frames are loaded from camera
	std::string gstreamer_pipeline(int sensor_id, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
		return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
			std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
			"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false";
	}
	
	/*
	* Can be called from external programs (i.e. a ROS program) to set the driving mode
	*/
	void SetDrivingMode(int drivingMode) {
		if (useCameraInput) {
			if (drivingMode == DRIVING_MODE::FORWARD) {
				cap = capForwardCamera;
			}
			else if (drivingMode == DRIVING_MODE::BACKWARD) {
				cap = capBackwardCamera;
			}
			else {
				globalInformation.errors += "[ERROR] Driving mode unknown: " + std::to_string(drivingMode) + "\n";
				return;
			}
		}
	}

	/*
	* Can be called from external programs (i.e. a ROS program) to set the show parameter
	*/
	void SetDrawVisionOutput(bool newDrawVisionOutput) {
		drawVisionOutput = newDrawVisionOutput;
	}

	// Used to retrieve globalInformation (by ROS for example)
	GlobalInformation GetGlobalInformation() {
		return globalInformation;
	}

	void ClearGlobalInformation() {
		globalInformation.objectList.clear();
		globalInformation.incomingSwitchStatus = SWITCH::NONE;
		globalInformation.switchedLaneStatus = SWITCH::NONE;
		globalInformation.switchDetected = false;
		globalInformation.errors = "";
		return;
	}

	/*
	* Runs in a thread to parse user keyboard input.
	* Pauses the program for any input, or sets a parameter.
	*/
	void CommandInputsWorker(std::atomic<bool>& pause, std::atomic<bool>& workDone) {
		while (!workDone.load()) {
			// Pause
			//std::cin.get();
			//pause.store(true);

			// Set parameter
			/*int drivingMode;
			std::cin >> drivingMode;
			SetDrivingMode(drivingMode);*/

		}
	}

	/*
	* Runs in a thread to save video frames.
	*/
	void SaveVideoFrame() {
		if (saveVideo) {
			video.write(saveOutputImage);
		}
	}

#pragma endregion Utilities Functions


	// ######## ######## ######## ######## Draw Functions ######## ######## ######## ######## 
#pragma region Draw Functions

	/*
	* Draw an info box on an image to show the output logics from the track and object detection. 
	*/
	void DrawInfoBoxVision(cv::Mat image) {
		std::vector<std::string> outputStringList;
		outputStringList.reserve(10);
		string outputString = "";

		// Message of fps
		outputString = "FPS: ";
		if (framesPerSecond != 0)
			outputString += std::to_string(globalInformation.fps);
		outputStringList.push_back(outputString);

		// Message of frame number
		outputString = "Frame: " + std::to_string(globalInformation.frameNumber);
		outputStringList.push_back(outputString);

		// Lane status
		if (globalInformation.laneStatus == LANE_STATUS::MIDDLE_TRACK) {
			outputStringList.push_back("Driving on middle track");
		}
		else if (globalInformation.laneStatus == LANE_STATUS::RIGHT_TRACK) {
			outputStringList.push_back("Driving on right track");
		}
		else if (globalInformation.laneStatus == LANE_STATUS::LEFT_TRACK) {
			outputStringList.push_back("Driving on left track");
		}
		else if (globalInformation.laneStatus == LANE_STATUS::SINGLE_TRACK) {
			outputStringList.push_back("Driving on single track");
		}

		// Track view distance
		outputString = "Track view distance: ";
		if (globalInformation.maxTrackViewDistance > -1) {
			outputString += std::to_string(globalInformation.maxTrackViewDistance) + " m";
			if (globalInformation.minTrackViewDistance > -1) {
				outputString += " (" + std::to_string(globalInformation.minTrackViewDistance) + " m)";
			}
		}
		outputStringList.push_back(outputString);

		// Message of direction change at a switch
		outputString = "Switch: ";
		if (globalInformation.switchDetected) {
			outputString += "Detected";
		}
		else {
			if (globalInformation.switchedLaneStatus == SWITCH::CONTINUED_ON_TRACK) {
				outputString += "Continued on track";
			}
			else if (globalInformation.switchedLaneStatus == SWITCH::SWITCHED_LEFT) {
				outputString += "Switched left";
			}
			else if (globalInformation.switchedLaneStatus == SWITCH::SWITCHED_RIGHT) {
				outputString += "Switched right";
			}
			else if (globalInformation.incomingSwitchStatus == SWITCH::INCOMING_FROM_LEFT) {
				outputString += "Incoming track from left";
			}
			else if (globalInformation.incomingSwitchStatus == SWITCH::INCOMING_FROM_RIGHT) {
				outputString += "Incoming track from right";
			}
		}
		outputStringList.push_back(outputString);

		// Main signal text
		outputString = "Signal: " + MAIN_SIGNAL_MESSAGE::TEXT[globalInformation.currentMainSignalMessage];
		if (globalInformation.distanceToMainSignal > -1) {
			outputString += " (" + std::to_string(globalInformation.distanceToMainSignal) + " m)";
		}
		outputStringList.push_back(outputString);

		// Distant main signal
		outputString = "Next signal: " + MAIN_SIGNAL_MESSAGE::TEXT[globalInformation.nextMainSignalMessage];
		outputStringList.push_back(outputString);

		// Crossing signal text
		outputString = "Crossing";
		if (globalInformation.currentRoadCrossingMessage == ROAD_CROSSING_MESSAGE::UPCOMING &&
			globalInformation.nextRoadCrossingMessage != ROAD_CROSSING_MESSAGE::NONE) {
			outputString = "Upcoming Crossing: Expect " + ROAD_CROSSING_MESSAGE::TEXT[globalInformation.nextRoadCrossingMessage];
		}
		else {
			if (globalInformation.roadCrossingBarriersDetected)
				outputString += " (Barriers): ";
			else
				outputString += ": ";
			outputString += ROAD_CROSSING_MESSAGE::TEXT[globalInformation.currentRoadCrossingMessage];
			if (globalInformation.roadCrossingDetected && globalInformation.distanceToRoadCrossing > -1)
				outputString += " (" + std::to_string(globalInformation.distanceToRoadCrossing) + " m)";
		}
		outputStringList.push_back(outputString);

		// Draw speed limit sign
		if (globalInformation.maxSpeedLimit > 0) {
			int circleSize = 50;
			string text = std::to_string(globalInformation.maxSpeedLimit);
			cv::Size const text_size = getTextSize(text, cv::FONT_HERSHEY_COMPLEX_SMALL, 2, 3, 0);
			cv::circle(image, cv::Point(OUTPUT_IMAGE_WIDTH - circleSize - 3, circleSize + 3), circleSize, cv::Scalar(0, 255, 255), -1);
			putText(image, text, cv::Point2f(OUTPUT_IMAGE_WIDTH - circleSize - 3 - text_size.width / 2, circleSize + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0, 0, 0), 3);
		}

		// Speed limit text
		outputString = "Speed limit: ";
		if (globalInformation.maxSpeedLimit > 0) {
			outputString += std::to_string(globalInformation.maxSpeedLimit) + " km/h";
			if (globalInformation.distanceToSpeedSign > -1) {
				outputString += " (" + std::to_string(globalInformation.distanceToSpeedSign) + " m)";
			}

			if (globalInformation.ATCSpeedUpSignDetected)
				outputString += " (ATC: Speed up)";
			else if (globalInformation.ATCSpeedDownSignDetected)
				outputString += " (ATC: Speed down)";
		}
		outputStringList.push_back(outputString);

		// Various objects detected
		outputString = "Other info: ";
		string originalText = outputString;
		if (globalInformation.distantSignalSignDetected && globalInformation.warningSignDetected)
			outputString += "Upcoming main signal, ";
		else if (globalInformation.warningSignDetected)
			outputString += "Warning sign, ";

		if (globalInformation.signVDetected)
			outputString += "Sign V, ";

		if (outputString.compare(originalText)) {
			outputString = outputString.substr(0, outputString.size() - 2);
		}
		outputStringList.push_back(outputString);

		outputString = "Objects in warning zone: ";
		if (globalInformation.objectList.size() != 0) {
			outputString += std::to_string(globalInformation.objectList.size());
		}
		else {
			outputString += "0";
		}
		outputStringList.push_back(outputString);

		// Pole counters
		outputString = "Left poles: " + std::to_string(globalInformation.leftPolesCounter) + ", Right poles: " + std::to_string(globalInformation.rightPolesCounter);
		outputStringList.push_back(outputString);

		// Text background
		memset(infoBoxOverlayImage.data, 0, outputImageArea * 3 * sizeof(*infoBoxOverlayImage.data));
		cv::Rect outerMask(0, 0, 353, 23 + outputStringList.size() * textSize * 25);
		cv::rectangle(infoBoxOverlayImage, outerMask, cv::Scalar(200, 110, 0), -1);
		cv::Rect innerMask(0, 0, 350, 20 + outputStringList.size() * textSize * 25);
		cv::rectangle(infoBoxOverlayImage, innerMask, cv::Scalar(160, 80, 0), -1);
		cv::addWeighted(image(outerMask), 0.5, infoBoxOverlayImage(outerMask), 0.5, 0, image(outerMask));

		// Actual print of above messages
		for (int i = 0; i < outputStringList.size(); i++) {
			int y = 20 + i * textSize * 25;
			cv::putText(image, outputStringList[i], cv::Point(textHorizontalShift, y), cv::FONT_HERSHEY_COMPLEX_SMALL,
				textSize, textColor, textThickness, lineType);
		}
		return;
	}

#pragma endregion Draw Functions


	// ######## ######## ######## ######## Load Frame ######## ######## ######## ########
#pragma region Load Frame

	/*
	* Load the next video frame.
	* Can be run in a separate thread.
	*/
	void LoadVideoFrame() {
		cap >> nextFrameImage;

		if (!nextFrameImage.empty()) {
			if (bottomPixelsToCrop > 0) {
				cv::Rect mask(0, 0, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT - bottomPixelsToCrop);
				nextFrameImage = nextFrameImage(mask);
			}

			if (nextFrameImage.size() != outputImageSize) {
				cv::resize(nextFrameImage, nextOutputImage, outputImageSize);
			}
			else {
				nextFrameImage.copyTo(nextOutputImage);
			}
		}
	}

	/*
	* Load the next camera frame.
	* Cannot be run in a separate thread.
	*/
	void LoadCameraFrame() {
		cap >> frameImage;
		
		if (!frameImage.empty()) {
			if (bottomPixelsToCrop > 0) {
				cv::Rect mask(0, 0, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT - bottomPixelsToCrop);
				frameImage = frameImage(mask);
			}

			if (frameImage.size() != outputImageSize) {
				cv::resize(frameImage, outputImage, outputImageSize);
			}
			else {
				frameImage.copyTo(outputImage);
			}
		}
	}

#pragma endregion Load Frame


	// ######## ######## ######## ######## Vision Detect ######## ######## ######## ########
#pragma region Vision Detect

	/*
	* Main function for computer vision, combining track detection and object detection. 
	* Exits and returns false if next frame is empty.
	* Shows a window with the vision output image by default. Use windowLess = true to suppress this.
	*/
	bool VisionDetect(bool windowLess = false, cv::Mat nextFrame_ = cv::Mat()){
		//Start time
		auto tstart = std::chrono::high_resolution_clock::now();

		// Camera: load frame directly
		// Video: load the next frame in separate thread and use the already loaded frame now.
		if (useCameraInput && !useROSCameras) {
			LoadCameraFrame();
		}
		else if (useROSCameras) { // get frame from ROS
			nextFrame_.copyTo(frameImage);
			if (!frameImage.empty()) {
				if (bottomPixelsToCrop > 0) {
					cv::Rect mask(0, 0, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT - bottomPixelsToCrop);
					frameImage = frameImage(mask);
				}

				if (frameImage.size() != outputImageSize) {
					cv::resize(frameImage, outputImage, outputImageSize);
				}
				else {
					frameImage.copyTo(outputImage);
				}
			}
		}
		else { // last frame loaded from video
			nextFrameImage.copyTo(frameImage);
			nextOutputImage.copyTo(outputImage);
		}

		// Exit if no more frames (video)
		if (frameImage.empty()) {
			workDone.store(true);
			return false;
		}

		// Reset global variables
		ClearGlobalInformation();

		// Detect track and objects
		std::thread trackDetectionThread(track_detection::DetectTracks, frameImage, drawVisionOutput);
		std::thread objectDetectionThread(object_detection::DetectObjects, outputImage);
		if (!useCameraInput) {
			// Load next video frame
			std::thread loadImageThread(LoadVideoFrame);
			loadImageThread.join();
		}
		objectDetectionThread.join();
		trackDetectionThread.join();

		// After processing
		object_detection::ProcessObjectsAfterTrackDetection();

		// Output global information
		object_detection::OutputGlobalInformation();
		track_detection::OutputGlobalInformation();

		//Draw pretty things
		if (drawVisionOutput) {
			track_detection::DrawOverlayImages(outputImage, false, drawWarningZoneLines);
			object_detection::DrawBoundingBoxes(outputImage);
			DrawInfoBoxVision(outputImage);
			if (drawCalibrationAid)
				track_detection::CalibrationAid(outputImage); // Draw lines to show the track width and a point for the horizon
		}

		// Show image
		if (!windowLess) {
			cv::imshow(windowName, outputImage);
			cv::waitKey(1);
		}

		// Set output parameters
		globalInformation.fps = framesPerSecond;
		globalInformation.frameNumber = frameNumber;
		globalInformation.image = outputImage;

		// Save frame in new thread
		if (saveVideo) {
			outputImage.copyTo(saveOutputImage);
			if (saveVideoWorkerThread.joinable())
				saveVideoWorkerThread.join();
			saveVideoWorkerThread = std::thread(SaveVideoFrame);
		}

		if (pause.load()) {
			cv::waitKey(0);
			pause.store(false);
		}

		frameNumber++;

		// End time
		auto tend = std::chrono::high_resolution_clock::now();
		auto duration = tend - tstart;
		framesPerSecond = (int)1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
		
		return true;
	}

#pragma endregion Vision Detect
	

	// ######## ######## ######## ######## Setup ######## ######## ######## ########
#pragma region Setup

	/*
	* Load parameters, set up the camera/video capture, setup object detection and track detection etc.
	* Shows a window with the vision output image by default. Use windowLess = true to suppress this.
	*/
	void SetupVision(bool windowLess = false, bool useROSCameras_ = false) {
		useROSCameras = useROSCameras_;
		if (useROSCameras) {
			useCameraInput = true;
		}

		LoadParameters();

		if (useCameraInput)
			frameNumber = 1;
		else {
			bottomPixelsToCrop = cropBottomFraction * OUTPUT_IMAGE_HEIGHT;
		}

		//Allocate memory for various objects
		outputImageSize = cv::Size(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT);
		track_detection::InizializeArrays();
		infoBoxOverlayImage = cv::Mat(outputImageSize, CV_8UC3);

		// Setup neural network
		object_detection::SetupNeuralNetworks();

		// Setup window
		if (!windowLess) {
			windowName = "Video Window";
			cv::namedWindow(windowName);
		}

		// Set up video saver
		if (saveVideo) {
			string savePathName = saveFolderPath + saveName;
			video = cv::VideoWriter(savePathName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsSavedVideo, outputImageSize);
			saveOutputImage = cv::Mat(outputImageSize, CV_8UC3);
		}

		// Setup camera/video capture
		if (useCameraInput && !useROSCameras) {// Use camera as input
			cout << "[INFO] Using camera input" << endl;
			std::string pipeline = gstreamer_pipeline(0, captureWidth, captureHeight, displayWidth, displayHeight, framerate, flipMethod);
			capForwardCamera = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
			capForwardCamera.set(cv::CAP_PROP_BUFFERSIZE, 1);

			pipeline = gstreamer_pipeline(1, captureWidth, captureHeight, displayWidth, displayHeight, framerate, flipMethod);
			capBackwardCamera = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
			capBackwardCamera.set(cv::CAP_PROP_BUFFERSIZE, 1);

			cap = capForwardCamera;
		}
		else if (!useCameraInput) {// Use video as input
			cout << "[INFO] Using video input" << endl;
			string videoPath = videoFolderPath + videoName;
			cap = cv::VideoCapture(videoPath);
		}

		// Load and set start frame
		if (!useCameraInput) {
			cap.set(cv::CAP_PROP_POS_FRAMES, frameNumber);
		}
		if (!useROSCameras) {
			cap >> nextFrameImage;
			if (nextFrameImage.empty()) {
				cout << "[ERROR] Empty frame at start. Exiting." << endl;
				exit(-1);
			}

			if (bottomPixelsToCrop > 0) {
				cv::Rect mask(0, 0, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT - bottomPixelsToCrop);
				nextFrameImage = nextFrameImage(mask);
			}

			if (nextFrameImage.size() != outputImageSize) {
				cv::resize(nextFrameImage, nextOutputImage, outputImageSize);
			}
			else {
				nextFrameImage.copyTo(nextOutputImage);
			}
		}

		//commandInputWorkerThread = std::thread(CommandInputsWorker, ref(pause), ref(workDone)); // uncomment to run a thread to handle user inputs from command line
	}

	// Call before exit to release data
	void VisionDestroy() {
		workDone.store(true);
		if (commandInputWorkerThread.joinable())
			commandInputWorkerThread.join();
		if (saveVideoWorkerThread.joinable())
			saveVideoWorkerThread.join();
		cout << "[OK] All Treads Joined" << endl;

		cap.release();
		video.release();
		object_detection::Cleanup();
		cout << "[SUCCESS] Program exiting." << endl;
	}

#pragma endregion Setup

}