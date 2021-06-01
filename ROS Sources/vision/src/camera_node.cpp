#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <atomic>

#include "vision/vision_input_message.h"
#include "GlobalEnumerationConstants.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

// Camera settings
int captureWidth = 1920; // TODO: same as output sizes?
int captureHeight = 1080;
int displayWidth = 1920;
int displayHeight = 1080;
int framerate = 30;
int flipMethod = 0; // 2 for Raspberry Pi camera, 0 for Arducam

cv::VideoCapture cap;
cv::VideoCapture capForwardCamera;
cv::VideoCapture capBackwardCamera;
cv::Mat frameImage;
cv::Mat forwardFrameImage;
cv::Mat backwardFrameImage;

std::atomic<bool> workDone(false);

using std::cout;
using std::endl;

// Create argument for initializing camera with wanted setup
std::string gstreamer_pipeline(int sensor_id, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
	return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
		std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
		"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false";
}

// Read input from vision_input_topic to choose camera or shut down
void VisionInputCallback(const vision::vision_input_message::ConstPtr& msg)
{
    switch ((*msg).input) {
    case SET_PARAMETER::FORWARD_DRIVING_MODE:
		cap = capForwardCamera;
        break;
	case SET_PARAMETER::BACKWARD_DRIVING_MODE:
		cap = capBackwardCamera;
        break;
    case SET_PARAMETER::EXIT:
		workDone.store(true);
        break;
    default:
        break;
    }
}

// Connect to two cameras and send out images via ROS. Requires two cameras!
int main(int argc, char** argv) {
	// Initialize node
    ros::init(argc, argv, "camera_node");
    ros::NodeHandle n;
	ros::Rate loop_rate(15);
    ros::Publisher imagePublisher = n.advertise<sensor_msgs::Image>("camera_topic", 1);
	ros::Subscriber visionInputSubscriber = n.subscribe("vision_input_topic", 10, VisionInputCallback);

	cout << "[INFO] Setting up cameras for ROS output" << endl;

	// Initialize forward camera
	std::string pipeline = gstreamer_pipeline(0, captureWidth, captureHeight, displayWidth, displayHeight, framerate, flipMethod);
	capForwardCamera = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
	capForwardCamera.set(cv::CAP_PROP_BUFFERSIZE, 1);

	// Initialize backward camera
	pipeline = gstreamer_pipeline(1, captureWidth, captureHeight, displayWidth, displayHeight, framerate, flipMethod);
	capBackwardCamera = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
	capBackwardCamera.set(cv::CAP_PROP_BUFFERSIZE, 1);

	// Set forward camera as default
	cap = capForwardCamera;

	while (!workDone.load()) {
		// Loop through possible buffered images
		for (int i = 0; i < 5; i++) {
			cap >> frameImage;
		}

		if (frameImage.empty()) {
			cout << "[ERROR] Empty image" << endl;
			exit(-1);
		}

        cv::resize(frameImage, frameImage, cv::Size(displayWidth, displayHeight));

		// Convert image from cv::Mat to ROS format
        cv_bridge::CvImage imageBridge;
        sensor_msgs::Image imageMessage;
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        imageBridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frameImage);
        imageBridge.toImageMsg(imageMessage);
        imagePublisher.publish(imageMessage);

		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}