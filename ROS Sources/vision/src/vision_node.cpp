#include "ros/ros.h"
#include "std_msgs/String.h"
#include "vision/vision_message.h"
#include "vision/vision_input_message.h"
#include "vision/vision_object.h"
#include "VisionSystem.h"
#include <sstream>
#include "GlobalInformation.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "GlobalEnumerationConstants.h"

std::string visionNodeErrors = "";
cv::Mat loadedImage(cv::Size(1920, 1080), CV_8UC3);

// Receive image
void CameraCallback(const sensor_msgs::ImageConstPtr& msg) {
  loadedImage = cv_bridge::toCvShare(msg, "bgr8")->image;
  std::cout << "Image received" << std::endl;
}

// Receive command input
void VisionInputCallback(const vision::vision_input_message::ConstPtr& msg)
{
  switch((*msg).input){
    case SET_PARAMETER::DRAW_VISION_OUTPUT_ON:
      vision::SetDrawVisionOutput(true);
      break;
    case SET_PARAMETER::DRAW_VISION_OUTPUT_OFF:
      vision::SetDrawVisionOutput(false);
      break;
    case SET_PARAMETER::FORWARD_DRIVING_MODE:
      // vision::SetDrivingMode(DRIVING_MODE::FORWARD); // for cameras directly connected to vision system (not via ROS)
      // handled by camera_node
      break;
    case SET_PARAMETER::BACKWARD_DRIVING_MODE:
      // vision::SetDrivingMode(DRIVING_MODE::BACKWARD); // for cameras directly connected to vision system (not via ROS)
      // handled by camera_node
      break;
    case SET_PARAMETER::EXIT:
      vision::VisionDestroy();
      break;
    default:
        visionNodeErrors += "[ERROR] Unsupported parameter in vision input: ";
        visionNodeErrors += (*msg).input;
        visionNodeErrors += "\n";
      break;
  }
}

// Run vision node to receive image to detect on and output detection results.
int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_node");
  ros::NodeHandle n;
  ros::Publisher publisher = n.advertise<vision::vision_message>("vision_topic", 100);
  ros::Publisher imagePublisher = n.advertise<sensor_msgs::Image>("vision_image_topic", 1);
  ros::Subscriber visionInputSubscriber = n.subscribe("vision_input_topic", 10, VisionInputCallback);
  ros::Subscriber imageSubscriber = n.subscribe("camera_topic", 1, CameraCallback);

  vision::SetupVision(true, true);

  ROS_INFO("%s", "VISION SETUP DONE");

  while (ros::ok())
  {
      if (!loadedImage.empty()) {
          // Detect on image
          vision::VisionDetect(true, loadedImage);

          // Create new output message with Global information
          GlobalInformation globalInformation = vision::GetGlobalInformation();

          vision::vision_message msg;

          msg.frameNumber = globalInformation.frameNumber;

          msg.laneStatus = globalInformation.laneStatus;

          msg.maxSpeedLimit = globalInformation.maxSpeedLimit;

          msg.currentMainSignalMessage = globalInformation.currentMainSignalMessage;
          msg.nextMainSignalMessage = globalInformation.nextMainSignalMessage;
          msg.currentRoadCrossingMessage = globalInformation.currentRoadCrossingMessage;
          msg.nextRoadCrossingMessage = globalInformation.nextRoadCrossingMessage;

          msg.maxTrackViewDistance = globalInformation.maxTrackViewDistance;
          msg.minTrackViewDistance = globalInformation.minTrackViewDistance;

          msg.distanceToRoadCrossing = globalInformation.distanceToRoadCrossing;
          msg.distanceToMainSignal = globalInformation.distanceToMainSignal;
          msg.distanceToRoadCrossingSignal = globalInformation.distanceToRoadCrossingSignal;
          msg.distanceToSpeedSign = globalInformation.distanceToSpeedSign;

          msg.switchDetected = globalInformation.switchDetected;
          msg.switchedLaneStatus = globalInformation.switchedLaneStatus;
          msg.incomingSwitchStatus = globalInformation.incomingSwitchStatus;

          msg.roadCrossingDetected = globalInformation.roadCrossingDetected;
          msg.roadCrossingBarriersDetected = globalInformation.roadCrossingBarriersDetected;

          msg.ATCSpeedUpSignDetected = globalInformation.ATCSpeedUpSignDetected;
          msg.ATCSpeedDownSignDetected = globalInformation.ATCSpeedDownSignDetected;
          msg.warningSignDetected = globalInformation.warningSignDetected;
          msg.distantSignalSignDetected = globalInformation.distantSignalSignDetected;
          msg.signVDetected = globalInformation.signVDetected;

          msg.leftPolesCounter = globalInformation.leftPolesCounter;
          msg.rightPolesCounter = globalInformation.rightPolesCounter;

          msg.errors = globalInformation.errors + visionNodeErrors;
          visionNodeErrors = "";

          // Add objects
          for (int iObject = 0; iObject < globalInformation.objectList.size(); iObject++) {
              vision::vision_object ROSObject;
              Object object = globalInformation.objectList[iObject];
              ROSObject.x = object.x;
              ROSObject.y = object.y;
              ROSObject.width = object.width;
              ROSObject.height = object.height;
              ROSObject.objectType = object.objectType;
              ROSObject.insideWarningZone = object.insideWarningZone;
              ROSObject.distance = object.distance;
              msg.objectList.push_back(ROSObject);
          }
          
          // Convert output image from OpenCV to ROS bridge
          cv::Mat image = globalInformation.image;
          cv_bridge::CvImage imageBridge;
          sensor_msgs::Image imageMessage;
          std_msgs::Header header;
          header.seq = globalInformation.frameNumber;
          header.stamp = ros::Time::now();
          imageBridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image);
          imageBridge.toImageMsg(imageMessage);

          // Publish message
          imagePublisher.publish(imageMessage);
          publisher.publish(msg);
      }

    ros::spinOnce();
  }

  vision::VisionDestroy();

  return 0;
}
