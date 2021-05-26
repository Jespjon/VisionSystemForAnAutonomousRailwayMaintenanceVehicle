#include "ros/ros.h"
#include "std_msgs/String.h"
#include "gps/gps_message.h"
#include "vision/vision_message.h"
#include "vision/vision_object.h"
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;

std::ofstream file("/home/jetson/localization.txt");

int heardVisionCounter = 0;
int heardGPSCounter = 0;

struct GPS {
    float latitude;
    float longitude;
    bool connected;
    int timeStamp;
    int numberOfSatellites;
} gpsStruct;

struct DetectedObjects {
    int laneStatus;
    bool switchDetected;
    int leftPolesCounter;
    int rightPolesCounter;
} detectedObjects;
DetectedObjects lastDetectedObjects;

// Add localization data to file
void ProcessDetectedObjects() {
    if (file.is_open()) {
        
        // Store time
        file << gpsStruct.timeStamp << endl;

        // Store position
        file << std::fixed << std::setprecision(5) << gpsStruct.latitude << endl;
        file << std::fixed << std::setprecision(5) << gpsStruct.longitude << endl;

        // Store track status
        file << detectedObjects.laneStatus << endl;

        // Store detected object information
        if (detectedObjects.switchDetected && !lastDetectedObjects.switchDetected) {
            file << "S" << endl;  // S = switch detected
        }
        if (detectedObjects.leftPolesCounter > lastDetectedObjects.leftPolesCounter) {
            file << "L" << endl;  // L = left pole detected
        }
        if (detectedObjects.rightPolesCounter > lastDetectedObjects.rightPolesCounter) {
            file << "R" << endl;  // R = right pole detected
        }
        file << endl;
    }
}

// Load and convert the neccesary GPS information to usable format
void gpsCallback(const gps::gps_message::ConstPtr& msg)
{
  // Print if GPS data was received
  heardGPSCounter++;
  if ((heardGPSCounter % 100) == 0) {
    ROS_INFO("I heard: GPS");
  }

  gpsStruct.latitude = (*msg).latitude;
  gpsStruct.longitude = (*msg).longitude;
  gpsStruct.connected = (*msg).connected;
  gpsStruct.timeStamp = (*msg).timeStamp;
  gpsStruct.numberOfSatellites = (*msg).numberOfSatellites;
}

// Load vision information to usable format
void visionCallback(const vision::vision_message::ConstPtr& msg)
{
  // Print if vision data was received
  heardVisionCounter++;
  if ((heardVisionCounter % 100) == 0){
    ROS_INFO("I heard: VISION");
  }

  // Store the old vision information
  lastDetectedObjects.switchDetected = detectedObjects.switchDetected;
  lastDetectedObjects.leftPolesCounter = detectedObjects.leftPolesCounter;
  lastDetectedObjects.rightPolesCounter = detectedObjects.rightPolesCounter;

  // Update the vision information
  detectedObjects.laneStatus = (*msg).laneStatus;
  detectedObjects.switchDetected = (*msg).switchDetected;
  detectedObjects.leftPolesCounter = (*msg).leftPolesCounter;
  detectedObjects.rightPolesCounter = (*msg).rightPolesCounter;

  // Store localization data only if enough satellites were detected
  if (gpsStruct.connected && gpsStruct.numberOfSatellites > 2){
    ProcessDetectedObjects();
  }
}

// Receive GPS and vision output and save localization data to file
int main(int argc, char **argv)
{
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle n;
  ros::Subscriber vision_subsriber = n.subscribe("vision_topic", 10, visionCallback);
  ros::Subscriber gps_subscriber = n.subscribe("gps_topic", 10, gpsCallback);
  ros::spin();

  file.close();

  return 0;
}