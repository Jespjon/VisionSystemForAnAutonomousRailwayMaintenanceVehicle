// About
/*
* ObjectDetection.cpp
* 
* Detects objects with a neural network, YOLOv4-tiny. 
* The objects are different vehicles (car, truck, bus, motorbike, bicycle, train), pedestrians, 
* different railway signs (speed limit, ATC speed up, ATC speed down, sign V, warning sign), 
* different railway signals (main signals, distant signals, road crossing signals, distant road crossing signals),
* road crossings, road crossing barriers and poles for the catenary system. 
* 
* The speed signs and signals are further classified by CNNs to find the specific message.
* 
* The detections are improved with a temporal robustness implementation. 
* 
* The object detection must be run combined with the track detection, as in VisionSystem.cpp.
* 
* First, initialize variables with SetupNeuralNetworks().
* For each frame:
*   -Detect objects with DetectObjects(). 
*      This function is independent of the track detection and can thus be run in parallel.
*   -ProcessObjectsAfterTrackDetection(). 
*      This function handles all detected objects, classifies, detects robustly, creates the logics etc.
*      This function is however dependent of the track detection, 
*      and must run after the track detection is complete.
*   -Call OutputGlobalInformation() to output the global variables in GlobalInformation.h 
*      with the logics from the object detection. 
*   -Call DrawBoundingBoxes() to draw all detected bounding boxes.
* 
*/

// Structure
/*
* Parameters, Constants and Variables
* 
* Utilities Functions
* 
* Detect Objects
* 
* Process
*   -Process Signs and Signals
*   -Process Poles, Vehicles and Pedestrians
*   -Preprocess Objects
*   -Process Objects
* 
* Output Global Information
* 
* Draw Bounding Boxes
* 
* Setup
*/


#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

#define OPENCV
#include "yolo_v2_class.hpp"    // imported functions from darknet DLL

#include "SharedParameters.h"
#include "TrackDetection.h"
#include "GlobalEnumerationConstants.h"
#include "GlobalInformation.h"
#include "YOLO-TensorRT.h"
#include "ObjectDetection.h"

using std::cout;
using std::endl;
using std::string;

namespace object_detection {

    // ######## ######## ######## ######## Parameters, Constants and Variables ######## ######## ######## ########
#pragma region Parameters, Constants and Variables

    // Global variables
    cv::Mat frameImage;

    // Scale
    const float OUTPUT_TO_DEVELOPMENT_SCALE = OUTPUT_IMAGE_HEIGHT / 720; // constants are optimized at 720 resolution; need to scale if other resolutions are used

    // ######## ######## Paths ######## ########

    std::string pathToYOLONetwork = pathToVisionSystem + "yolo-network/";
    std::string pathToClassificationNetworks = pathToVisionSystem + "classification-networks/";

    // CV_FILLED not defined in some OpenCV versions
#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

    // ######## ######## YOLO Network ######## ########

    // Paths to files
    const std::string  yoloNamesFile = pathToYOLONetwork + "obj.names";
    const std::string  yoloCfgFile = pathToYOLONetwork + "yolov4-tiny.cfg";
    const std::string  yoloWeightsFile = pathToYOLONetwork + "yolov4-tiny_best.weights";
    const std::string yoloEnginePath = pathToYOLONetwork + "yolov4-tiny-608.trt";

    // YOLO objects
    Detector* detector;
    std::vector<std::string> ObjectNames;
    std::vector<BoundingBox> resultVector;
    const float DETECTED_OBJECT_THRESHOLD = 0.3;

    // ######## ######## Current Statuses ######## ########

    // Current lane status
    int laneStatus = LANE_STATUS::LEFT_TRACK; // of type LANE_STATUS
    int mainTrackMiddlePosition = OUTPUT_IMAGE_WIDTH / 2;

    // Current speed limit
    int maxSpeedLimit = -1;
    int maxSpeedLimitSpeedSign = -1;
    int maxSpeedLimitSignal = -1;

    // Current signal messages
    int currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::NONE; // of type MAIN_SIGNAL_MESSAGE
    int nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::NONE; // of type MAIN_SIGNAL_MESSAGE
    int currentRoadCrossingMessage = ROAD_CROSSING_MESSAGE::NONE; // of type ROAD_CROSSING_MESSAGE
    int nextRoadCrossingMessage = ROAD_CROSSING_MESSAGE::NONE; // of type ROAD_CROSSING_MESSAGE

    // ######## ######## Distance ######## ########

    // Distance parameters
    const int C_4 = 735;
    const int MAX_DISTANCE = 200;

    // Object widths (in meters) (approximated values)
    const float SIGN_V_WIDTH = 0.3;
    const float SIGN_SPEED_UP_WIDTH = 0.4;
    const float SIGN_SPEED_DOWN_WIDTH = 0.4;
    const float SIGN_TRIANGLE_WARNING_WIDTH = 0.4;
    const float SIGN_SPEED_WIDTH = 0.4;
    const float MAIN_SIGNAL_WIDTH = 0.4;
    const float ROAD_CROSSING_SIGNAL_WIDTH = 0.4;
    const float DISTANT_SIGNAL_WIDTH = 0.6;
    const float DISTANT_ROAD_CROSSING_SIGNAL_WIDTH = 0.6;

    // ######## ######## Detection of Signs and Signals ######## ########

    const int INCREASED_BOX_WIDTH = 3 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int INCREASED_BOX_HEIGHT = 3 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int MAX_OBJECT_BOX_SIZE = 1000;

    // Sign detection constants
    const int MIN_ATC_SPEED_SIZE = 20 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int MIN_TRIANGLE_SIGN_SIZE = 20 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int SIGN_OCCURENCE_THRESHOLD = 10;
    const int MIN_SIGN_V_SIZE = 10 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int SIGN_V_ENDING_THRESHOLD = 10;
    const int SIGN_TRIANGLE_WARNING_ENDING_THRESHOLD = 10;

    // Signal detection constants
    const int SIGNAL_DETECTION_ENDING_THRESHOLD = 40;
    const int MIN_SIGNAL_OCCURENCE = 20;

    // Sign and signal struct
    struct Signal {
        int signalMessage = -1;
        int sideOfTrack = SIDE::LEFT;
        int xPosition = 0;
        int yPosition = 0;
        int w = 0;
        int h = 0;
        int distance = -1;
        Signal() {}
        Signal(int signalMessage, int sideOfTrack, int xPosition, int yPosition, int w = 0, int h = 0, int distance = -1) {
            this->signalMessage = signalMessage;
            this->sideOfTrack = sideOfTrack;
            this->xPosition = xPosition;
            this->yPosition = yPosition;
            this->w = w;
            this->h = h;
            this->distance = distance;
        }
    };
    std::vector<Signal> detectedMainSignals;
    std::vector<Signal> detectedRoadCrossingSignals;
    std::vector<Signal> detectedSpeedSigns;

    // ######## Classification ########

    // Speed sign classification
    string speedSignClassificationSavedModelPathPyTorchOnnx = pathToClassificationNetworks + "speed_sign_pytorch_model.onnx";
    std::string speedSignClassificationNamesFile = pathToClassificationNetworks + "used_classes_speed_sign_pytorch.names";
    std::vector<std::string> speedSignClassificationObjNames;
    cv::dnn::Net speedSignClassificationModel;
    std::vector<int> detectedSpeedSignClassesCounter;
    std::vector<int> lastFrameWithSpeedSign;
    const int MIN_SPEED_SIGN_OCCURENCE = 10;
    const int MIN_SPEED_SIGN_BOX_WIDTH = 15;
    const int MIN_SPEED_SIGN_BOX_HEIGHT = 15;
    int distanceToSpeedSign;

    // Main signal classification
    string mainSignalClassificationSavedModelPathPyTorchOnnx = pathToClassificationNetworks + "main_signal_pytorch_model.onnx";
    std::string mainSignalClassificationNamesFile = pathToClassificationNetworks + "used_classes_main_signal_pytorch.names";
    std::vector<std::string> detectedMainSignalClassNames;
    cv::dnn::Net mainSignalClassificationModel;
    const int MIN_MAIN_SIGNAL_BOX_WIDTH = 7;
    const int MIN_MAIN_SIGNAL_BOX_HEIGHT = 12;
    const int MAX_MAIN_SIGNAL_BOX_WIDTH = 40 * OUTPUT_TO_DEVELOPMENT_SCALE;
    int distanceToMainSignal;

    // Distant signal classification
    const int MIN_DISTANT_SIGNAL_BOX_WIDTH = 10;
    const int MIN_DISTANT_SIGNAL_BOX_HEIGHT = 10;
    string distantSignalClassificationSavedModelPathPyTorchOnnx = pathToClassificationNetworks + "distant_signal_pytorch_model.onnx";
    std::string  distantSignalClassificationNamesFile = pathToClassificationNetworks + "used_classes_distant_signal_pytorch.names";
    std::vector<std::string> distantSignalClassificationObjNames;
    cv::dnn::Net distantSignalClassificationModel;

    // Road crossing signal classification
    const int MIN_ROAD_CROSSING_SIGNAL_BOX_WIDTH = 10;
    const int MIN_ROAD_CROSSING_SIGNAL_BOX_HEIGHT = 10;
    const int MAX_ROAD_CROSSING_SIGNAL_BOX_SIZE = 35 * OUTPUT_TO_DEVELOPMENT_SCALE;
    string roadCrossingSignalClassificationSavedModelPathPyTorchOnnx = pathToClassificationNetworks + "road_crossing_signal_pytorch_model.onnx";
    std::string roadCrossingSignalClassificationNamesFile = pathToClassificationNetworks + "used_classes_road_crossing_signal_pytorch.names";
    std::vector<std::string> detectedRoadCrossingSignalClassNames;
    cv::dnn::Net roadCrossingSignalClassificationModel;
    int distanceToRoadCrossingSignal;

    // Distant road crossing signal classification
    const int MIN_DISTANT_ROAD_CROSSING_SIGNAL_BOX_WIDTH = 20 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int MIN_DISTANT_ROAD_CROSSING_SIGNAL_BOX_HEIGHT = 20 * OUTPUT_TO_DEVELOPMENT_SCALE;
    string distantRoadCrossingSignalClassificationSavedModelPathPyTorchOnnx = pathToClassificationNetworks + "distant_road_crossing_signal_pytorch_model.onnx";
    std::string  distantRoadCrossingSignalClassificationNamesFile = pathToClassificationNetworks + "used_classes_distant_road_crossing_signal_pytorch.names";
    std::vector<std::string> distantRoadCrossingSignalClassificationObjNames;
    cv::dnn::Net distantRoadCrossingSignalClassificationModel;

    // ######## Temporal Recognition ########

    // Main Signal temporal recognition
    std::vector<int> lastFrameWithMainSignal;
    std::vector<int> detectedMainSignalClassesCounter;
    std::vector<std::string> compositeMainSignalClasseNames({ "0", "1", "13", "1-13", "1-14", "135", "1-135", "2" });
    std::vector<int> compositeMainSignalSpeeds({ -1, 80, 40, 80, 80, 40, 80, 0 });
    std::vector<int> compositeMainSignalClassesCounter;

    // Distant signal temporal recognition
    std::vector<int> detectedDistantSignalClassesCounter;
    std::vector<int> compositeDistantSignalClassesCounter;
    std::vector<std::string> compositeDistantSignalClasseNames({ "0", "0-1", "0-2", "0-13" });
    std::vector<int> lastFrameWithDistantSignal;
    std::vector<Signal> detectedDistantSignals;
    bool distantSignalWarningSignDetected = false;

    // Road crossing signal temporal recognition
    std::vector<int> lastFrameWithRoadCrossingSignal;
    std::vector<int> detectedRoadCrossingSignalClassesCounter;
    std::vector<std::string> compositeRoadCrossingSignalClasseNames({ "-1", "0", "2", "3" });
    std::vector<int> compositeRoadCrossingSignalClassesCounter;

    // Distant road crossing signal temporal recognition
    std::vector<int> detectedDistantRoadCrossingSignalClassesCounter;
    std::vector<int> compositeDistantRoadCrossingSignalClassesCounter;
    std::vector<std::string> compositeDistantRoadCrossingSignalClasseNames({ "0", "1", "0-1" });
    std::vector<int> lastFrameWithDistantRoadCrossingSignal;
    std::vector<Signal> detectedDistantRoadCrossingSignals;
    bool detectedDistantRoadCrossingSignal = false;
    bool distantRoadCrossingSignalStop = false;
    int distantRoadCrossingSignalLastFrameDetected = 0;

    // ######## Enumeration Constants for Signal Classes ######## 

    namespace DETECTED_MAIN_SIGNAL_CLASSES {
        enum DETECTED_MAIN_SIGNAL_CLASSES {
            FALSE,
            NO_SIGNAL,
            SIGNAL_1,
            SIGNAL_13,
            SIGNAL_14,
            SIGNAL_135,
            SIGNAL_2,
        };
    }

    namespace COMPOSITE_MAIN_SIGNAL_CLASSES {
        enum COMPOSITE_MAIN_SIGNAL_CLASSES {
            NO_SIGNAL,
            SIGNAL_1,
            SIGNAL_13,
            SIGNAL_1_13,
            SIGNAL_1_14,
            SIGNAL_135,
            SIGNAL_1_135,
            SIGNAL_2,
        };
        const std::vector<std::string> TEXT({ "No signal", "Signal 1", "Signal 13", "Signal 1-13", "Signal 1-14", "Signal 135", "Signal 1-135", "Signal_2" });
    }

    namespace DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES {
        enum DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES {
            FALSE,
            NO_SIGNAL,
            SIGNAL,
        };
    }

    namespace COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES {
        enum COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES {
            NO_SIGNAL,
            SIGNAL,
            BLINKING,
        };
        const std::vector<std::string> TEXT({ "No signal", "Solid", "Blinking" });
    }

    namespace DETECTED_DISTANT_SIGNAL_CLASSES {
        enum DETECTED_DISTANT_SIGNAL_CLASSES {
            FALSE,
            NO_SIGNAL,
            SIGNAL_1,
            SIGNAL_2,
            SIGNAL_13,
        };
    }

    namespace COMPOSITE_DISTANT_SIGNAL_CLASSES {
        enum COMPOSITE_DISTANT_SIGNAL_CLASSES {
            NO_SIGNAL,
            SIGNAL_0_1,
            SIGNAL_0_2,
            SIGNAL_0_13,
        };
        const std::vector<std::string> TEXT({ "No signal", "Signal 1", "Signal 2", "Signal 1-3" });
    }

    namespace ROAD_CROSSING_SIGNAL_CLASSES {
        enum ROAD_CROSSING_SIGNAL_CLASSES {
            FALSE,
            NO_SIGNAL,
            STOP,
            GO,
        };
        const std::vector<std::string> TEXT({ "False", "No signal", "Signal 2", "Signal 3" });
    }

    // ######## ######## Detection of Objects ######## ########

    //Various object detection constants
    const int DETECTED_OBJECT_OCCURENCE_THRESHOLD = 5;
    const int OBJECT_DETECTION_ENDING_THRESHOLD = 20;

    // Train detection constants
    const int MIN_TRAIN_SIZE = 15 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int MAX_TRAIN_DETECTION_DISTANCE = 100; // only detect trains within this distance to decrease false positives
    const int MAX_TRAIN_LANE_STATUS_DISTANCE = 60; // set the lane status if train is inside this distance 

    // Object struct
    struct DetectedObject {
        bool detected = false;
        int lastFrameDetected = 0;
        int counter = 0;
    };

    // Signs
    DetectedObject signV;
    DetectedObject signSpeedUp;
    DetectedObject signSpeedDown;
    DetectedObject possiblePreSignalSign;
    DetectedObject signTriangleWarning;

    // Pedestrians and vehicles
    DetectedObject bicycle;
    DetectedObject bus;
    DetectedObject car;
    DetectedObject motorbike;
    DetectedObject person;
    DetectedObject train;
    DetectedObject truck;

    // Train
    Train detectedTrain;

    // ######## Road Crossing ########

    // Road crossings constants
    const int ROAD_CROSSING_DETECTION_MIN_OCCURENCE_THRESHOLD = 20;
    const int ROAD_CROSSING_DETECTION_ENDING_THRESHOLD = 30;
    const int MAX_ROAD_CROSSING_DISTANCE = 80;

    DetectedObject roadCrossing;
    DetectedObject roadCrossingBarrier;

    bool roadCrossingSignalStop = false;
    bool roadCrossingSignalDetected = false;
    bool upcomingRoadCrossing = false;
    int distanceToRoadCrossing;

    // ######## Poles ########

    // Pole counter parameters
    int poleDistanceThreshold = 200 * OUTPUT_TO_DEVELOPMENT_SCALE;
    const int POLE_DETECTED_ENDING_THRESHOLD = 7;
    const int POLE_DETECTED_COUNTER_THRESHOLD = 7;

    // Poles
    struct Pole {
        int sideOfTrack = SIDE::LEFT; // of type SIDE
        int xPosition = 0;
        int yPosition = 0;

        Pole() {}
        Pole(int sideOfTrack, int xPosition, int yPosition) {
            this->sideOfTrack = sideOfTrack;
            this->xPosition = xPosition;
            this->yPosition = yPosition;
        }
    };
    std::vector<Pole> detectedPoles;
    Pole oldLeftSidePole;
    Pole oldRightSidePole;
    int lastFrameOldLeftPoleDetected = 0;
    int lastFrameOldRightPoleDetected = 0;
    int leftPoleDetectedCounter = 0;
    int rightPoleDetectedCounter = 0;
    int leftPolesCounter = 0;
    int rightPolesCounter = 0;

#pragma endregion Parameters, Constants and Variables


    // ######## ######## ######## ######## Utilities Functions ######## ######## ######## ########
#pragma region Utilities Functions

    /*
    * Calculate the side of track an object is standing on.
    * Matters for signals and poles.
    */
    int CalculateSideOfTrack(BoundingBox boundingBox) {
        int trackMiddle;
        int trackWidth;
        int sideOfTrack;
        float scaleX = track_detection::GetExportDetectionWidthRatio();
        float scaleY = track_detection::GetExportDetectionHeightRatio();
        track_detection::GetTrackPosition((boundingBox.y + boundingBox.h) / scaleY, trackWidth, trackMiddle);
        trackMiddle = (int)(trackMiddle * scaleX);
        int boundingBoxMiddle = (int)(boundingBox.x + boundingBox.w / 2);

        if (boundingBoxMiddle < trackMiddle) {
            sideOfTrack = SIDE::LEFT;
        }
        else {
            sideOfTrack = SIDE::RIGHT;
        }
        return sideOfTrack;
    }

    /*
    * Returns the train object. 
    * Are used in TrackDetection.cpp
    */
    Train GetDetectedTrain() {
        return detectedTrain;
    }

#pragma endregion Utilities Functions


    // ######## ######## ######## ######## Detect Objects ######## ######## ######## ########
#pragma region Detect Objects

    /*
    * Detect objects from an image with a YOLO network, running on either Darknet or TensorRT.
    * The detected objects are in form of bounding boxes.
    */
    void DetectObjects(cv::Mat inputFrameImage) {
        frameImage = inputFrameImage;
#ifdef TENSORRT
        // Detect by YOLO in TensorRT
        resultVector = yolo_tensorrt::Detect(frameImage);
#else
        // Detection by Yolo in Darknet
        std::shared_ptr<image_t> det_image = detector->mat_to_image_resize(frameImage); // resize
        std::vector<bbox_t> resultVectorTmp = detector->detect_resized(*det_image, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT, DETECTED_OBJECT_THRESHOLD, true);

        // Convert from bbox_t to BoundingBox
        resultVector.clear();
        for (bbox_t boundingBoxTmp : resultVectorTmp){
            int newX = std::min(std::max(0, (int)boundingBoxTmp.x), OUTPUT_IMAGE_WIDTH - 1);
            int newY = std::min(std::max(0, (int)boundingBoxTmp.y), OUTPUT_IMAGE_HEIGHT - 1);

            int diffX = std::abs(newX - (int)boundingBoxTmp.x);
            int diffY = std::abs(newY - (int)boundingBoxTmp.y);

            int newW = std::min((int)boundingBoxTmp.w - diffX, OUTPUT_IMAGE_WIDTH - newX - 1);
            int newH = std::min((int)boundingBoxTmp.h - diffY, OUTPUT_IMAGE_HEIGHT - newY - 1);

            BoundingBox boundingBox(newX, newY, newW, newH, boundingBoxTmp.prob, boundingBoxTmp.obj_id, 1);
            resultVector.push_back(boundingBox);
        }
#endif
    }

#pragma endregion Detect Objects


    // ######## ######## ######## ######## Process Signs and Signals ######## ######## ######## ########
#pragma region Process Signs and Signals
    
    /*
    * Find what sign or signal should be used depending on lane status and the distance to the main track.
    * Use the closest sign/signal if several is found.
    */
    Signal GetSignalToUse(std::vector<Signal> detectedMainSignals){
        float scaleX = track_detection::GetExportDetectionWidthRatio();
        float scaleY = track_detection::GetExportDetectionHeightRatio();

        // Choose the signal depending on lane status
        Signal signalToUse;
        if (laneStatus == LANE_STATUS::MIDDLE_TRACK || laneStatus == LANE_STATUS::LEFT_TRACK) {
            int mostRightX = 0;

            for (int iSignal = 0; iSignal < detectedMainSignals.size(); iSignal++) {
                Signal signal = detectedMainSignals[iSignal];
                if (signal.signalMessage != -1 && signal.sideOfTrack == SIDE::LEFT && signal.xPosition > mostRightX) {
                    mostRightX = signal.xPosition;
                    signalToUse = signal;
                }
            }
        }
        else { // RIGHT_TRACK or SINGLE_TRACK
            if (!detectedMainSignals.empty()){
                // If more than one signal found on single track/right track, use the most right one (assuming placed to the left of track)
                int closestDistance = OUTPUT_IMAGE_WIDTH;
                for (int iSignal = 0; iSignal < detectedMainSignals.size(); iSignal++) {
                    Signal signal = detectedMainSignals[iSignal];
                    int trackMiddle = track_detection::GetTrackMiddle(signal.yPosition / scaleY) * scaleX;
                    int distance = std::abs(signal.xPosition + signal.w / 2 - trackMiddle);
                    if (signal.signalMessage != -1 && distance < closestDistance) {
                        closestDistance = distance;
                        signalToUse = signal;
                    }
                }
            }
        }
        return signalToUse;
    }

    /*
    * Find which sign to use and robustly detect it's message.
    * Each possible message is counted for each frame. A message is robust if it has occurred enough frames.
    */
    void RecogniseSpeedSigns() {
        // Reset vector
        for (int i = 0; i < lastFrameWithSpeedSign.size(); i++)
        {
            if (frameNumber - lastFrameWithSpeedSign[i] == SIGNAL_DETECTION_ENDING_THRESHOLD) {
                detectedSpeedSignClassesCounter[i] = 0;
            }
        }

        Signal signToUse = GetSignalToUse(detectedSpeedSigns);

        if (signToUse.signalMessage == -1) {
            return;
        }

        // Increment the current detected signal
        detectedSpeedSignClassesCounter[signToUse.signalMessage]++;
        lastFrameWithSpeedSign[signToUse.signalMessage] = frameNumber;

        if (detectedSpeedSignClassesCounter.size() != 0) {
            // Find most occurring sign
            int maxSpeedSignClassCounterIndex = 0;
            int maxSpeedSignClassCounter = 0;
            for (int i = 1; i < detectedSpeedSignClassesCounter.size(); i++) {
                if (detectedSpeedSignClassesCounter[i] > MIN_SPEED_SIGN_OCCURENCE && detectedSpeedSignClassesCounter[i] > maxSpeedSignClassCounter) {
                    maxSpeedSignClassCounter = detectedSpeedSignClassesCounter[i];
                    maxSpeedSignClassCounterIndex = i;
                }
            }

            if (maxSpeedSignClassCounter > 0) {
                // Set speed limit
                if (std::stoi(speedSignClassificationObjNames[maxSpeedSignClassCounterIndex]) > 0) {
                    maxSpeedLimitSpeedSign = std::stoi(speedSignClassificationObjNames[maxSpeedSignClassCounterIndex]);
                    distanceToSpeedSign = signToUse.distance;
                }
            }
        }
    }

    /*
    * Find which signal to use and robustly detect it's message.
    * Each possible message is counted for each frame. A message is robust if it has occurred enough frames.
    */
    void RecogniseMainSignals() {
        // Reset vector
        for (int i = 0; i < lastFrameWithMainSignal.size(); i++)
        {
            if (frameNumber - lastFrameWithMainSignal[i] == SIGNAL_DETECTION_ENDING_THRESHOLD) {
                detectedMainSignalClassesCounter[i] = 0;
            }
        }

        Signal signalToUse = GetSignalToUse(detectedMainSignals);
        if (signalToUse.signalMessage != -1) {
            // Increment the current detected signal
            detectedMainSignalClassesCounter[signalToUse.signalMessage]++;
            lastFrameWithMainSignal[signalToUse.signalMessage] = frameNumber;

            // Reset vector
            for (int i = 1; i < compositeMainSignalClassesCounter.size(); i++) {
                compositeMainSignalClassesCounter[i] = 0;
            }

            // Combine detected messages to composite messages (blinking), and find if robustly detected.
            for (int i = 1; i < detectedMainSignalClassesCounter.size(); i++) { // Skips false signal
                if (i == DETECTED_MAIN_SIGNAL_CLASSES::NO_SIGNAL && detectedMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE) {
                    compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::NO_SIGNAL] = detectedMainSignalClassesCounter[i];
                }
                else if (i == DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_1 && detectedMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE)
                    compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1] = detectedMainSignalClassesCounter[i];
                else if (i == DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_13 && detectedMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE / 3) {
                    if (detectedMainSignalClassesCounter[DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_1] > MIN_SIGNAL_OCCURENCE / 2) {
                        // Blinking
                        compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1_13] = detectedMainSignalClassesCounter[DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_1] + detectedMainSignalClassesCounter[i];
                    }
                    else {
                        // Solid
                        compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_13] = detectedMainSignalClassesCounter[i];
                    }
                }
                else if (i == DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_14 && detectedMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE / 3) {
                    compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1_14] = detectedMainSignalClassesCounter[DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_1] + detectedMainSignalClassesCounter[i];
                }
                else if (i == DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_135 && detectedMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE / 3) {
                    if (detectedMainSignalClassesCounter[DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_1] > MIN_SIGNAL_OCCURENCE / 2) {
                        // Blinking
                        compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1_135] = detectedMainSignalClassesCounter[DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_1] + detectedMainSignalClassesCounter[i];
                    }
                    else {
                        // Solid
                        compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_135] = detectedMainSignalClassesCounter[i];
                    }
                }
                else if (i == DETECTED_MAIN_SIGNAL_CLASSES::SIGNAL_2 && detectedMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE) {
                    compositeMainSignalClassesCounter[COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_2] = detectedMainSignalClassesCounter[i];
                }
            }

            // Find most occurring signal
            int maxMainSignalClassCounterIndex = 0;
            int maxMainSignalClassCounter = 0;
            for (int i = 1; i < compositeMainSignalClassesCounter.size(); i++) { // Skips no signal, skips signal1
                if (compositeMainSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE && compositeMainSignalClassesCounter[i] > maxMainSignalClassCounter) {
                    maxMainSignalClassCounter = compositeMainSignalClassesCounter[i];
                    maxMainSignalClassCounterIndex = i;
                }
            }

            if (maxMainSignalClassCounterIndex > 0) {
                // Signal robustly detected, set main signal message
                if (compositeMainSignalSpeeds[maxMainSignalClassCounterIndex] > 0) // do not set speed limit from stop sign
                    maxSpeedLimitSignal = compositeMainSignalSpeeds[maxMainSignalClassCounterIndex];
                switch (maxMainSignalClassCounterIndex) // Set signal message
                {
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_80;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::NONE;
                    break;
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_13:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_40_GENTLY;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::NONE;
                    break;
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1_13:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_80;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::STOP;
                    break;
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1_14:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_80;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_80;
                    break;
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_135:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_40;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::SHORT_TO_STOP;
                    break;
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_1_135:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_80;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_40;
                    break;
                case COMPOSITE_MAIN_SIGNAL_CLASSES::SIGNAL_2:
                    currentMainSignalMessage = MAIN_SIGNAL_MESSAGE::STOP;
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::NONE;
                    break;
                default: break;
                }
                distanceToMainSignal = signalToUse.distance;
            }
        }
    }

    /*
    * Find which signal to use and robustly detect it's message.
    * Each possible message is counted for each frame. A message is robust if it has occurred enough frames.
    */
    void RecogniseDistantSignals(){
        // Reset vector
        for (int i = 0; i < lastFrameWithDistantSignal.size(); i++)
        {
            if (frameNumber - lastFrameWithDistantSignal[i] == SIGNAL_DETECTION_ENDING_THRESHOLD) {
                detectedDistantSignalClassesCounter[i] = 0;
            }
        }

        Signal signalToUse = GetSignalToUse(detectedDistantSignals);

        if (signalToUse.signalMessage == -1)
            return;
        
        // Increment the current detected signal
        detectedDistantSignalClassesCounter[signalToUse.signalMessage]++;
        lastFrameWithDistantSignal[signalToUse.signalMessage] = frameNumber;

        if (detectedDistantSignals.size() != 0) {

            // Reset vector
            for (int i = 0; i < compositeDistantSignalClassesCounter.size(); i++) {
                compositeDistantSignalClassesCounter[i] = 0;
            }

            // Combine detected messages to composite messages, and find if robustly detected.
            if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL] > MIN_SIGNAL_OCCURENCE) {
                compositeDistantSignalClassesCounter[COMPOSITE_DISTANT_SIGNAL_CLASSES::NO_SIGNAL] = detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL];
            }
            if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_1] > MIN_SIGNAL_OCCURENCE / 3) {
                if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL] > MIN_SIGNAL_OCCURENCE / 3 && detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_1] > MIN_SIGNAL_OCCURENCE / 3) {
                    compositeDistantSignalClassesCounter[COMPOSITE_DISTANT_SIGNAL_CLASSES::SIGNAL_0_1] = detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_1] + detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL];
                }
            }
            if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_2] > MIN_SIGNAL_OCCURENCE / 3) {
                if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL] > MIN_SIGNAL_OCCURENCE / 3 && detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_2] > MIN_SIGNAL_OCCURENCE / 3) {
                    compositeDistantSignalClassesCounter[COMPOSITE_DISTANT_SIGNAL_CLASSES::SIGNAL_0_2] = detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_2] + detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL];
                }
            }
            if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_13] > MIN_SIGNAL_OCCURENCE / 3) {
                if (detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL] > MIN_SIGNAL_OCCURENCE / 3 && detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_13] > MIN_SIGNAL_OCCURENCE / 3) {
                    compositeDistantSignalClassesCounter[COMPOSITE_DISTANT_SIGNAL_CLASSES::SIGNAL_0_13] = detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::SIGNAL_13] + detectedDistantSignalClassesCounter[DETECTED_DISTANT_SIGNAL_CLASSES::NO_SIGNAL];
                }
            }

            // Find most occurring signal
            int maxDistantSignalClassCounterIndex = 0;
            int maxDistantSignalClassCounter = 0;
            for (int i = 0; i < compositeDistantSignalClassesCounter.size(); i++) {
                if (compositeDistantSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE && compositeDistantSignalClassesCounter[i] > maxDistantSignalClassCounter) {
                    maxDistantSignalClassCounter = compositeDistantSignalClassesCounter[i];
                    maxDistantSignalClassCounterIndex = i;
                }
            }

            if (maxDistantSignalClassCounter > 0) {
                // Signal robustly detected, set next main signal message
                if (maxDistantSignalClassCounterIndex == COMPOSITE_DISTANT_SIGNAL_CLASSES::SIGNAL_0_1) {
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::STOP;
                }
                else if (maxDistantSignalClassCounterIndex == COMPOSITE_DISTANT_SIGNAL_CLASSES::SIGNAL_0_2) {
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_80;
                }
                else if (maxDistantSignalClassCounterIndex == COMPOSITE_DISTANT_SIGNAL_CLASSES::SIGNAL_0_13) {
                    nextMainSignalMessage = MAIN_SIGNAL_MESSAGE::GO_40;
                }
                else if (maxDistantSignalClassCounterIndex == COMPOSITE_DISTANT_SIGNAL_CLASSES::NO_SIGNAL) {
                    possiblePreSignalSign.detected = true;
                    possiblePreSignalSign.lastFrameDetected = frameNumber;
                }
            }
            
        }
    }

    /*
    * Find which signal to use and robustly detect it's message.
    * Each possible message is counted for each frame. A message is robust if it has occurred enough frames.
    */
    void RecogniseRoadCrossingSignal(){
        // Reset vector
        for (int i = 0; i < lastFrameWithRoadCrossingSignal.size(); i++)
        {
            if (frameNumber - lastFrameWithRoadCrossingSignal[i] == SIGNAL_DETECTION_ENDING_THRESHOLD) {
                detectedRoadCrossingSignalClassesCounter[i] = 0;
            }
        }

        Signal signalToUse = GetSignalToUse(detectedRoadCrossingSignals);
        if (signalToUse.signalMessage != -1) {
            // Increment the current detected signal
            detectedRoadCrossingSignalClassesCounter[signalToUse.signalMessage]++;
            lastFrameWithRoadCrossingSignal[signalToUse.signalMessage] = frameNumber;
            
            // Reset vector
            for (int i = 0; i < compositeRoadCrossingSignalClassesCounter.size(); i++) {
                compositeRoadCrossingSignalClassesCounter[i] = 0;
            }

            // Find if robustly detected.
            for (int i = 1; i < detectedRoadCrossingSignalClassesCounter.size(); i++) { // skips false signal
                if (i == ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL && detectedRoadCrossingSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE) {
                    compositeRoadCrossingSignalClassesCounter[ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL] = detectedRoadCrossingSignalClassesCounter[i];
                }
                else if (i == ROAD_CROSSING_SIGNAL_CLASSES::STOP && detectedRoadCrossingSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE) {
                    compositeRoadCrossingSignalClassesCounter[ROAD_CROSSING_SIGNAL_CLASSES::STOP] = detectedRoadCrossingSignalClassesCounter[i];
                }
                else if (i == ROAD_CROSSING_SIGNAL_CLASSES::GO && detectedRoadCrossingSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE) {
                    compositeRoadCrossingSignalClassesCounter[ROAD_CROSSING_SIGNAL_CLASSES::GO] = detectedRoadCrossingSignalClassesCounter[i];
                }
            }

            // Find most occurring road signal.
            int maxRoadCrossingSignalClassCounterIndex = 0;
            int maxRoadCrossingSignalClassCounter = 0;
            for (int i = 1; i < compositeRoadCrossingSignalClassesCounter.size(); i++) { // only signal1
                if (compositeRoadCrossingSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE && compositeRoadCrossingSignalClassesCounter[i] > maxRoadCrossingSignalClassCounter) {
                    maxRoadCrossingSignalClassCounter = compositeRoadCrossingSignalClassesCounter[i];
                    maxRoadCrossingSignalClassCounterIndex = i;
                }
            }
            
            if (maxRoadCrossingSignalClassCounter > 0) {
                // Signal robustly detected
                roadCrossingSignalDetected = true;
                distanceToRoadCrossingSignal = signalToUse.distance;
                if (maxRoadCrossingSignalClassCounterIndex == ROAD_CROSSING_SIGNAL_CLASSES::GO) {
                    roadCrossingSignalStop = false;
                }
                else {
                    roadCrossingSignalStop = true;
                }
            }
        }
    }

    /*
    * Find which signal to use and robustly detect it's message.
    * Each possible message is counted for each frame. A message is robust if it has occurred enough frames.
    */
    void RecogniseDistantRoadCrossingSignals() {
        // Reset vector
        for (int i = 0; i < lastFrameWithDistantRoadCrossingSignal.size(); i++)
        {
            if (frameNumber - lastFrameWithDistantRoadCrossingSignal[i] == SIGNAL_DETECTION_ENDING_THRESHOLD) {
                detectedDistantRoadCrossingSignalClassesCounter[i] = 0;
            }
        }

        Signal signalToUse = GetSignalToUse(detectedDistantRoadCrossingSignals);

        if (signalToUse.signalMessage == -1)
            return;

        // Increment the current detected signal
        detectedDistantRoadCrossingSignalClassesCounter[signalToUse.signalMessage]++;
        lastFrameWithDistantRoadCrossingSignal[signalToUse.signalMessage] = frameNumber;

        if (detectedDistantRoadCrossingSignals.size() != 0) {

            // Reset vector
            for (int i = 0; i < compositeDistantRoadCrossingSignalClassesCounter.size(); i++) {
                compositeDistantRoadCrossingSignalClassesCounter[i] = 0;
            }

            // Combine detected messages to composite messages, and find if robustly detected.
            if (detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL] > MIN_SIGNAL_OCCURENCE) {
                compositeDistantRoadCrossingSignalClassesCounter[COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL] = detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL];
            }
            if (detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL] > MIN_SIGNAL_OCCURENCE / 2) {
                if (detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL] > MIN_SIGNAL_OCCURENCE / 2 && detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL] > MIN_SIGNAL_OCCURENCE / 2) {
                    compositeDistantRoadCrossingSignalClassesCounter[COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::BLINKING] = detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL] + detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::NO_SIGNAL];
                }
                else if (detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL] > MIN_SIGNAL_OCCURENCE) {
                    compositeDistantRoadCrossingSignalClassesCounter[COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL] = detectedDistantRoadCrossingSignalClassesCounter[DETECTED_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL];
                }
            }

            // Find most occurring signal
            int maxDistantRoadCrossingSignalClassCounterIndex = 0;
            int maxDistantRoadCrossingSignalClassCounter = 0;
            for (int i = 0; i < compositeDistantRoadCrossingSignalClassesCounter.size(); i++) {
                if (compositeDistantRoadCrossingSignalClassesCounter[i] > MIN_SIGNAL_OCCURENCE && compositeDistantRoadCrossingSignalClassesCounter[i] > maxDistantRoadCrossingSignalClassCounter) {
                    maxDistantRoadCrossingSignalClassCounter = compositeDistantRoadCrossingSignalClassesCounter[i];
                    maxDistantRoadCrossingSignalClassCounterIndex = i;
                }
            }

            if (maxDistantRoadCrossingSignalClassCounter > 0) {
                // Signal robustly detected
                detectedDistantRoadCrossingSignal = true;
                distantRoadCrossingSignalLastFrameDetected = frameNumber;
                if (maxDistantRoadCrossingSignalClassCounterIndex == COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::SIGNAL) { distantRoadCrossingSignalStop = false; }
                else if (maxDistantRoadCrossingSignalClassCounterIndex == COMPOSITE_DISTANT_ROAD_CROSSING_SIGNAL_CLASSES::BLINKING) { distantRoadCrossingSignalStop = true; }
            }
        }

        // Set the current message
        if (detectedDistantRoadCrossingSignal) {
            upcomingRoadCrossing = true;
            if (distantRoadCrossingSignalStop) {
                nextRoadCrossingMessage = ROAD_CROSSING_MESSAGE::STOP;
            }
            else {
                nextRoadCrossingMessage = ROAD_CROSSING_MESSAGE::GO;
            }
        }
    }
    
#pragma endregion Process Signs and Signals


    // ######## ######## ######## ######## Process Poles, Vehicles and Pedestrians ######## ######## ######## ########
#pragma region Process Poles, Vehicles and Pedestrians

    /*
    * Detects and tracks the closest poles to the left and right side. 
    * A pole is counted if it hasn't been seen for some frames.
    */
    void ProcessPoles(){
        Pole leftSidePole;
        Pole rightSidePole;
        // Find closest poles
        for (Pole pole : detectedPoles){
            if (pole.sideOfTrack == SIDE::LEFT && pole.yPosition > leftSidePole.yPosition){
                leftSidePole = pole;
            }
            else if (pole.sideOfTrack == SIDE::RIGHT && pole.yPosition > rightSidePole.yPosition){
                rightSidePole = pole;
            }
        }

        // Set the oldLeftSidePole for the first time
        if (oldLeftSidePole.yPosition == 0){
            oldLeftSidePole = leftSidePole;
        }
        if (oldRightSidePole.yPosition == 0){
            oldRightSidePole = rightSidePole;
        }

        if (leftSidePole.yPosition != 0) {
            // Find if the distance to the last detected pole is close enough to be the same pole
            if (oldLeftSidePole.xPosition - leftSidePole.xPosition > -10){
                int distanceToOldLeftPole = std::abs(leftSidePole.xPosition - oldLeftSidePole.xPosition) + std::abs(leftSidePole.yPosition - oldLeftSidePole.yPosition);
                if (distanceToOldLeftPole < poleDistanceThreshold){
                    oldLeftSidePole = leftSidePole;
                    leftPoleDetectedCounter++;
                    lastFrameOldLeftPoleDetected = frameNumber;
                }
            }
        }
        
        if (rightSidePole.yPosition != 0) {
            // Find if the distance to the last detected pole is close enough to be the same pole
            if (rightSidePole.xPosition - oldRightSidePole.xPosition > -10) {
                int distanceToOldRightPole = std::abs(rightSidePole.xPosition - oldRightSidePole.xPosition) + std::abs(rightSidePole.yPosition - oldRightSidePole.yPosition);
                if (distanceToOldRightPole < poleDistanceThreshold) {
                    oldRightSidePole = rightSidePole;
                    rightPoleDetectedCounter++;
                    lastFrameOldRightPoleDetected = frameNumber;
                }
            }
        }

        // If the old pole has not be seen for some frames, count the pole
        if (frameNumber - lastFrameOldLeftPoleDetected > POLE_DETECTED_ENDING_THRESHOLD){
            if (leftPoleDetectedCounter > POLE_DETECTED_COUNTER_THRESHOLD){
                //Found left pole!
                leftPolesCounter++;
            }
            leftPoleDetectedCounter = 0;
            oldLeftSidePole.yPosition = 0;
        }
        if (frameNumber - lastFrameOldRightPoleDetected > POLE_DETECTED_ENDING_THRESHOLD){
            if (rightPoleDetectedCounter > POLE_DETECTED_COUNTER_THRESHOLD){
                //Found right pole
                rightPolesCounter++;
            }
            rightPoleDetectedCounter = 0;
            oldRightSidePole.yPosition = 0;
        }
    }

    /*
    * Find if detected vehicle or pedestrian is inside the warning zone and calculate the distance to the object.
    */
    void ProcessVehiclesAndPedestrians(DetectedObject tempObject, BoundingBox i) {
        if (tempObject.detected) {
            cv::Point bottomLeftCorner = cv::Point(i.x, i.y + i.h);
            cv::Point bottomRightCorner = cv::Point(i.x + i.w, i.y + i.h);

            int insideWarningZone = track_detection::FindIfObjectIsInsideWarningZone(bottomLeftCorner, bottomRightCorner, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT);
            
            if (insideWarningZone != WARNING_ZONE::OUTSIDE_ZONE) {
                float scaleY = track_detection::GetExportDetectionHeightRatio();
                int distanceToObject = track_detection::GetDistance((i.y + i.h) / scaleY);

                Object object;
                object.x = i.x;
                object.y = i.y;
                object.width = i.w;
                object.height = i.h;
                object.distance = distanceToObject;
                object.objectType = i.classId;
                object.insideWarningZone = insideWarningZone;
                globalInformation.objectList.push_back(object);
            }
        }
    }

#pragma endregion Process Poles, Vehicles and Pedestrians


    // ######## ######## ######## ######## Preprocess Objects ######## ######## ######## ########
#pragma region Preprocess Objects
     
    // ######## ######## Distance ######## ########

    /*
    * Calculate the distance to an object with a known set object width.
    * distance = C_4 * realObjectWidthMeters / imageObjectWidthPixels,  where C_4 = metersToPixels * focalLenght.
    */
    int GetObjectDistance(BoundingBox boundingBox, float objectRealWidth) {
        int distance = (C_4 * objectRealWidth) / boundingBox.w;
        return distance;
    }

    /*
    * Sets the distance to the object.
    * The distance to objects with ground contact (vehicles, road crossings etc.) will be calculated using the track width.
    * The distance to objects with a set width (signs and signals etc.) will be calculated using the object width.
    */
    void SetObjectDistance(BoundingBox& boundingBox, float objectRealWidth = -1.0) {
        int distance;
        if (objectRealWidth != -1) { // Calculate the distance by the object width (only works or objects with a set width)
            distance = GetObjectDistance(boundingBox, objectRealWidth);
        }
        else { // Calculate distance by using the track width (only works for objects on the ground)
            distance = track_detection::GetDistance((boundingBox.y + boundingBox.h) / track_detection::GetExportDetectionHeightRatio());
        }
        distance = std::min(distance, MAX_DISTANCE);
        boundingBox.distance = distance;
    }
    
    // ######## ######## Object Robustness ######## ########

    void CheckObjectRobustness(DetectedObject& object, int robustnessThreshold = DETECTED_OBJECT_OCCURENCE_THRESHOLD) {
        object.counter++;
        object.lastFrameDetected = frameNumber;
        if (object.counter > robustnessThreshold) {
            object.detected = true;
        }
    }

    void CheckIfClearObject(DetectedObject& object, int objectDetectionEndingThreshold = OBJECT_DETECTION_ENDING_THRESHOLD) {
        if ((frameNumber - object.lastFrameDetected) == objectDetectionEndingThreshold) {
            object.detected = false;
            object.counter = 0;
        }
    }

    // ######## ######## Preprocess Signs and Signals ######## ########

    /*
    * Preprocess and classify speed signs and signals with CNN from PyTorch. 
    */
    int SignSignalClassification(cv::Mat image, cv::dnn::Net classificationModel, bool convertToGrayscale) {
        cv::resize(image, image, cv::Size(32, 32));
        if (convertToGrayscale) {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            image.convertTo(image, CV_32FC1);
        }
        else {
            image.convertTo(image, CV_32FC3);
        }
        image /= 255.0;

        cv::Mat blob = cv::dnn::blobFromImage(image);
        classificationModel.setInput(blob);
        cv::Mat output = classificationModel.forward();

        cv::Point classIdPoint;
        double confidence;
        minMaxLoc(output.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classIndex = classIdPoint.x;
        return classIndex;
    }

    /*
    * Preprocess and classify signs and signals. 
    * The bounding boxes must not be too large or to small. 
    */
    void PreprocessAndClassifySignal(BoundingBox &i, int increasedBoxWidth, int increasedBoxHeight, int minBoxWidth, int minBoxHeight, int maxBoxWidth, int maxBoxHeight, std::vector<Signal> &detectedSignalVector, cv::dnn::Net classificationModel, bool convertToGrayscale){
        i.w = std::min(i.w + increasedBoxWidth, OUTPUT_IMAGE_WIDTH - i.x);
        i.h = std::min(i.h + increasedBoxHeight, OUTPUT_IMAGE_HEIGHT - i.y);
        int classIndex;
        bool outsideView = i.x == 0 || (i.x + i.w) == OUTPUT_IMAGE_WIDTH - 1 || i.y == 0 || i.y + i.h == OUTPUT_IMAGE_HEIGHT - 1;
        bool boxTooSmall = i.w < minBoxWidth + increasedBoxWidth || i.h < minBoxHeight + increasedBoxHeight;
        bool boxTooLarge = i.w > maxBoxWidth + increasedBoxWidth || i.h > maxBoxHeight + increasedBoxHeight;
        if (!boxTooSmall && !boxTooLarge && !outsideView) {
            cv::Rect mask = cv::Rect(i.x, i.y, i.w, i.h);
            cv::Mat croppedImage = frameImage(mask);
            classIndex = SignSignalClassification(croppedImage, classificationModel, convertToGrayscale);
            int sideOfTrack = CalculateSideOfTrack(i);
            Signal detectedSignal(classIndex, sideOfTrack, i.x, i.y + i.h, i.w, i.h, i.distance);
            detectedSignalVector.push_back(detectedSignal);
        }
        else {
            classIndex = -1; // Too small image to classify signal correctly
            Signal detectedSignal;
            detectedSignalVector.push_back(detectedSignal);
        }
    }

#pragma endregion Preprocess Objects


    // ######## ######## ######## ######## Process Objects ######## ######## ######## ########
#pragma region Process Objects

    /*
    * Process the detected objects by processing temporal robustness variables and classifying signs and signals.
    */
    void ProcessObjects() {
        // Reset variables
        detectedMainSignals.clear();
        detectedRoadCrossingSignals.clear();
        detectedDistantRoadCrossingSignals.clear();
        detectedDistantSignals.clear();
        detectedPoles.clear();
        detectedSpeedSigns.clear();

        distanceToMainSignal = -1;
        distanceToRoadCrossingSignal = -1;
        distanceToSpeedSign = -1;
        distanceToRoadCrossing = -1;

        // Clear robustness variables if too long time since objects seen, special for barriers and crossings
        if (frameNumber - roadCrossing.lastFrameDetected == ROAD_CROSSING_DETECTION_ENDING_THRESHOLD) {
            if (roadCrossing.detected && frameNumber - distantRoadCrossingSignalLastFrameDetected >= ROAD_CROSSING_DETECTION_ENDING_THRESHOLD) {
                detectedDistantRoadCrossingSignal = false;
                distantRoadCrossingSignalStop = false;
            }
            roadCrossing.detected = false;
            roadCrossingSignalStop = false;
            roadCrossingSignalDetected = false;
            roadCrossing.counter = 0;
            distanceToRoadCrossing = -1;
            roadCrossingBarrier.detected = false;
            roadCrossingBarrier.counter = 0;
            if (!upcomingRoadCrossing)
                currentRoadCrossingMessage = ROAD_CROSSING_MESSAGE::NONE;
            if (!detectedDistantRoadCrossingSignal){
                nextRoadCrossingMessage = ROAD_CROSSING_MESSAGE::NONE;
            }
        }

        // Clear robustness variables if too long time since objects seen
        CheckIfClearObject(signV, SIGN_V_ENDING_THRESHOLD);
        CheckIfClearObject(signSpeedUp);
        CheckIfClearObject(signSpeedDown);
        CheckIfClearObject(signTriangleWarning, SIGN_TRIANGLE_WARNING_ENDING_THRESHOLD);

        // Clear robustness variables if too long time since objects seen, special for distant signal
        //CheckIfClearObject(possiblePreSignalSign);
        if (frameNumber - possiblePreSignalSign.lastFrameDetected > OBJECT_DETECTION_ENDING_THRESHOLD) {
            distantSignalWarningSignDetected = false;
            possiblePreSignalSign.detected = false;
        }

        // Clear robustness variables if too long time since objects seen, pedestrians and vehicles
        CheckIfClearObject(bicycle);
        CheckIfClearObject(bus);
        CheckIfClearObject(car);
        CheckIfClearObject(motorbike);
        CheckIfClearObject(person);
        CheckIfClearObject(train);
        CheckIfClearObject(truck);

        detectedTrain.detected = false;

        // Find and preprocess (and classify) present objects
        int closestCrossingDistance = 1000;
        for (auto& i : resultVector) {
            if (i.classId == OBJECTS::CROSSING) {
				SetObjectDistance(i);
                if (i.distance < MAX_ROAD_CROSSING_DISTANCE) {
                    CheckObjectRobustness(roadCrossing, ROAD_CROSSING_DETECTION_MIN_OCCURENCE_THRESHOLD);

                    if (i.distance < closestCrossingDistance) {
                        distanceToRoadCrossing = i.distance;
                        closestCrossingDistance = i.distance;
                    }
                }
			}
            else if (i.classId == OBJECTS::BARRIER) {
                if (roadCrossing.detected)
                    CheckObjectRobustness(roadCrossingBarrier);
            }
            else if (i.classId == OBJECTS::SIGN_V) {
                if (i.w > MIN_SIGN_V_SIZE && i.h > MIN_SIGN_V_SIZE){
                    SetObjectDistance(i, SIGN_V_WIDTH);
                    CheckObjectRobustness(signV, SIGN_OCCURENCE_THRESHOLD);
                }
            }
            else if (i.classId == OBJECTS::SIGN_SPEED_UP) {
                if (i.w > MIN_ATC_SPEED_SIZE && i.h > MIN_ATC_SPEED_SIZE){
                    SetObjectDistance(i, SIGN_SPEED_UP_WIDTH);
                    CheckObjectRobustness(signSpeedUp, SIGN_OCCURENCE_THRESHOLD);
                }
            }
            else if (i.classId == OBJECTS::SIGN_SPEED_DOWN) {
                if (i.w > MIN_ATC_SPEED_SIZE && i.h > MIN_ATC_SPEED_SIZE){
                    SetObjectDistance(i, SIGN_SPEED_DOWN_WIDTH);
                    CheckObjectRobustness(signSpeedDown, SIGN_OCCURENCE_THRESHOLD);
                }
            }
            else if (i.classId == OBJECTS::SIGN_TRIANGLE_WARNING) {
                if (i.w > MIN_TRIANGLE_SIGN_SIZE && i.h > MIN_TRIANGLE_SIGN_SIZE){
                    SetObjectDistance(i, SIGN_TRIANGLE_WARNING_WIDTH);
                    CheckObjectRobustness(signTriangleWarning, SIGN_OCCURENCE_THRESHOLD);
                }
            }
            else if (i.classId == OBJECTS::BICYCLE) {
                SetObjectDistance(i);
                CheckObjectRobustness(bicycle);
                ProcessVehiclesAndPedestrians(bicycle, i);
            }
            else if (i.classId == OBJECTS::BUS) {
                SetObjectDistance(i);
                CheckObjectRobustness(bus);
                ProcessVehiclesAndPedestrians(bus, i);
            }
            else if (i.classId == OBJECTS::CAR) {
                SetObjectDistance(i);
                CheckObjectRobustness(car);
                ProcessVehiclesAndPedestrians(car, i);
            }
            else if (i.classId == OBJECTS::MOTORCYCLE) {
                SetObjectDistance(i);
                CheckObjectRobustness(motorbike);
                ProcessVehiclesAndPedestrians(motorbike, i);
            }
            else if (i.classId == OBJECTS::PERSON) {
                SetObjectDistance(i);
                CheckObjectRobustness(person);
                ProcessVehiclesAndPedestrians(person, i);
            }
            else if (i.classId == OBJECTS::TRAIN) {
                if (i.w > MIN_TRAIN_SIZE && i.h > MIN_TRAIN_SIZE){
                    SetObjectDistance(i);
                    if (i.distance < MAX_TRAIN_DETECTION_DISTANCE){
                        CheckObjectRobustness(train);
                        ProcessVehiclesAndPedestrians(train, i);

                        if (train.detected && i.distance < MAX_TRAIN_LANE_STATUS_DISTANCE) {
                            detectedTrain.detected = true;
                            detectedTrain.sideOfTrack = CalculateSideOfTrack(i);
                            cv::Point bottomLeftCorner = cv::Point(i.x, i.y + i.h);
                            cv::Point bottomRightCorner = cv::Point(i.x + i.w, i.y + i.h);
                            detectedTrain.insideWarningZone = track_detection::FindIfObjectIsInsideWarningZone(bottomLeftCorner, bottomRightCorner, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT) != WARNING_ZONE::OUTSIDE_ZONE;
                        }
                    }
                }
            }
            else if (i.classId == OBJECTS::TRUCK) {
                SetObjectDistance(i);
                CheckObjectRobustness(truck);
                ProcessVehiclesAndPedestrians(truck, i);
            }
            else if (i.classId == OBJECTS::SIGN_SPEED) {
                SetObjectDistance(i, SIGN_SPEED_WIDTH);
                PreprocessAndClassifySignal(i, 0, 0, MIN_SPEED_SIGN_BOX_WIDTH, MIN_SPEED_SIGN_BOX_HEIGHT, MAX_OBJECT_BOX_SIZE, MAX_OBJECT_BOX_SIZE, detectedSpeedSigns, speedSignClassificationModel, true);
            }
            else if (i.classId == OBJECTS::MAIN_SIGNAL) {
                SetObjectDistance(i, MAIN_SIGNAL_WIDTH);
                PreprocessAndClassifySignal(i, INCREASED_BOX_WIDTH, INCREASED_BOX_HEIGHT, MIN_MAIN_SIGNAL_BOX_WIDTH, MIN_MAIN_SIGNAL_BOX_HEIGHT, MAX_MAIN_SIGNAL_BOX_WIDTH, MAX_OBJECT_BOX_SIZE, detectedMainSignals, mainSignalClassificationModel, false);
            }
            else if (i.classId == OBJECTS::ROAD_CROSSING_SIGNAL) {
                SetObjectDistance(i, ROAD_CROSSING_SIGNAL_WIDTH);
                PreprocessAndClassifySignal(i, 0, 0, MIN_ROAD_CROSSING_SIGNAL_BOX_WIDTH, MIN_ROAD_CROSSING_SIGNAL_BOX_HEIGHT, MAX_ROAD_CROSSING_SIGNAL_BOX_SIZE, MAX_ROAD_CROSSING_SIGNAL_BOX_SIZE, detectedRoadCrossingSignals, roadCrossingSignalClassificationModel, false);
            }
            else if (i.classId == OBJECTS::DISTANT_SIGNAL) {
                SetObjectDistance(i, DISTANT_SIGNAL_WIDTH);
                PreprocessAndClassifySignal(i, INCREASED_BOX_WIDTH, INCREASED_BOX_HEIGHT, MIN_DISTANT_SIGNAL_BOX_WIDTH, MIN_DISTANT_SIGNAL_BOX_HEIGHT, MAX_OBJECT_BOX_SIZE, MAX_OBJECT_BOX_SIZE, detectedDistantSignals, distantSignalClassificationModel, false);
            }
            else if (i.classId == OBJECTS::DISTANT_ROAD_CROSSING_SIGNAL) {
                SetObjectDistance(i, DISTANT_ROAD_CROSSING_SIGNAL_WIDTH);
                PreprocessAndClassifySignal(i, INCREASED_BOX_WIDTH, INCREASED_BOX_HEIGHT, MIN_DISTANT_ROAD_CROSSING_SIGNAL_BOX_WIDTH, MIN_DISTANT_ROAD_CROSSING_SIGNAL_BOX_HEIGHT, MAX_OBJECT_BOX_SIZE, MAX_OBJECT_BOX_SIZE, detectedDistantRoadCrossingSignals, distantRoadCrossingSignalClassificationModel, true);
            }
            else if (i.classId == OBJECTS::POLE) {
                SetObjectDistance(i);
                float scaleY = track_detection::GetExportDetectionHeightRatio();
                int trackHighestRow = track_detection::GetTrackHighestRow() * scaleY;
                if (i.y + i.h >= trackHighestRow) {
                    int sideOfTrack = CalculateSideOfTrack(i);
                    Pole detectedPole(sideOfTrack, i.x, i.y + i.h);
                    detectedPoles.push_back(detectedPole);
                }
            }
        }

        if (roadCrossing.detected)
            upcomingRoadCrossing = false;
        if (signTriangleWarning.detected && signV.detected) {
            upcomingRoadCrossing = true;
        }

        if (possiblePreSignalSign.detected && (signTriangleWarning.detected || !detectedMainSignals.empty())) {
            distantSignalWarningSignDetected = true;
        }
    }

    /*
    * Sets the messages of the speed limit and the road crossing message.
    */
    void ProcessMessages(){
        // Set speed limit
        if (maxSpeedLimitSpeedSign > -1){
            maxSpeedLimit = maxSpeedLimitSpeedSign;
        }
        else if (maxSpeedLimitSignal > -1){
            maxSpeedLimit = maxSpeedLimitSignal;
        }
        maxSpeedLimitSpeedSign = -1;
        maxSpeedLimitSignal = -1;

        // Set road crossing message
        if (roadCrossing.detected) {
            if (roadCrossingSignalDetected) {
                if (roadCrossingSignalStop)
                    currentRoadCrossingMessage = ROAD_CROSSING_MESSAGE::STOP;
                else
                    currentRoadCrossingMessage = ROAD_CROSSING_MESSAGE::GO;
            }
            else {
                currentRoadCrossingMessage = ROAD_CROSSING_MESSAGE::DETECTED;
            }
        } 
        else if (upcomingRoadCrossing) {
            currentRoadCrossingMessage = ROAD_CROSSING_MESSAGE::UPCOMING;
        }
    }

    /*
    * Processes all detected objects.
    * Requires that the track has been detected.
    */
    void ProcessObjectsAfterTrackDetection() {
        laneStatus = track_detection::GetLaneStatus();

        ProcessObjects();

        RecogniseSpeedSigns();
        RecogniseDistantRoadCrossingSignals();
        RecogniseDistantSignals();
        RecogniseMainSignals();
        RecogniseRoadCrossingSignal();

        ProcessPoles();

        ProcessMessages();
    }

#pragma endregion Process Objects


    // ######## ######## ######## ######## Output Global Information ######## ######## ######## ########
#pragma region Output Global Information

    /*
    * Sets global variables relating to object detection to be used by the Vision System.
    */
    void OutputGlobalInformation() {
        globalInformation.currentMainSignalMessage = currentMainSignalMessage;
        globalInformation.nextMainSignalMessage = nextMainSignalMessage;

        globalInformation.currentRoadCrossingMessage = currentRoadCrossingMessage;
        globalInformation.nextRoadCrossingMessage = nextRoadCrossingMessage;
        globalInformation.roadCrossingDetected = roadCrossing.detected;
        globalInformation.roadCrossingBarriersDetected = roadCrossingBarrier.detected;
        if (roadCrossing.detected)
            globalInformation.distanceToRoadCrossing = distanceToRoadCrossing;

        globalInformation.maxSpeedLimit = maxSpeedLimit;

        globalInformation.ATCSpeedUpSignDetected = signSpeedUp.detected;
        globalInformation.ATCSpeedDownSignDetected = signSpeedDown.detected;

        globalInformation.warningSignDetected = signTriangleWarning.detected;
        globalInformation.distantSignalSignDetected = distantSignalWarningSignDetected;
        globalInformation.signVDetected = signV.detected;

        globalInformation.leftPolesCounter = leftPolesCounter;
        globalInformation.rightPolesCounter = rightPolesCounter;

        globalInformation.distanceToMainSignal = distanceToMainSignal;
        globalInformation.distanceToRoadCrossingSignal = distanceToRoadCrossingSignal;
        globalInformation.distanceToSpeedSign = distanceToSpeedSign;
    }

#pragma endregion Output Global Information


    // ######## ######## ######## ######## Draw Bounding Boxes ######## ######## ######## ########
#pragma region Draw Bounding Boxes

    /*
    * Add specific class information to the bounding box.
    */
    void AddExtraTextToBoxes(std::string& printName, std::vector<Signal> detectedObjects, int& detectedObjectIndex, std::vector<std::string> objectClassificationNames) {
        if (detectedObjects.size() > 0 && detectedObjects[detectedObjectIndex].signalMessage > 0) {
            printName += " " + objectClassificationNames[detectedObjects[detectedObjectIndex].signalMessage];
            detectedObjectIndex++;
        }
    }

    void DrawBoundingBoxes(cv::Mat image) {
        //Red, Orange, Yellow, Yellow-orange, Teal, Spring Green, Lime Green, Lighter Blue, Darker Blue, Black
        float const colors[10][3] = { {0.1,0.1,1}, {0,0.3,1}, {0,1,1}, {0,0.7,1}, {0.5,0.5,0}, {0.7,1,0}, {0.2,0.9,0.2}, {1,0.6,0.1}, {0.9,0.4,0}, {0,0,0} };

        int classColorIndices[19] = { 0, 1, 3, 2, 2, 2, 2, 6, 5, 6, 4, 5, 0, 8, 0, 0, 7, 0, 9 };
        string customClassNames[19] = { "car", "person", "speed sign", "V-sign", "warning sign", "ATC speed down", "ATC speed up", "distant signal", "crossing signal", "main signal", "train", "distant crossing", "bus", "barrier", "truck", "motorcycle", "crossing", "bicycle", "pole" };

        // Reset the counters for number of individual signs and signals
        int detectedSpeedSignIndex = 0;
        int detectedSignalIndex = 0;
        int detectedSignal1Index = 0;
        int detectedPreSignalIndex = 0;
        int detectedTriangleSignalIndex = 0;

        cv::Mat overlayImage;
        image.copyTo(overlayImage);

        for (auto& i : resultVector) {
            cv::Scalar color(colors[classColorIndices[i.classId]][0], colors[classColorIndices[i.classId]][1], colors[classColorIndices[i.classId]][2]);
            color *= 255;
            cv::rectangle(overlayImage, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
            if (ObjectNames.size() > i.classId && i.classId != OBJECTS::POLE) {
                string printName = customClassNames[i.classId];

                // Add extra information to bounding boxes for signals and speed signs
                if (i.classId == OBJECTS::SIGN_SPEED) {
                    AddExtraTextToBoxes(printName, detectedSpeedSigns, detectedSpeedSignIndex, speedSignClassificationObjNames);
                }
                else if (i.classId == OBJECTS::MAIN_SIGNAL) {
                    AddExtraTextToBoxes(printName, detectedMainSignals, detectedSignalIndex, detectedMainSignalClassNames);
                }
                else if (i.classId == OBJECTS::ROAD_CROSSING_SIGNAL) {
                    AddExtraTextToBoxes(printName, detectedRoadCrossingSignals, detectedSignal1Index, detectedRoadCrossingSignalClassNames);
                }
                else if (i.classId == OBJECTS::DISTANT_SIGNAL) {
                    AddExtraTextToBoxes(printName, detectedDistantSignals, detectedPreSignalIndex, distantSignalClassificationObjNames);
                }
                else if (i.classId == OBJECTS::DISTANT_ROAD_CROSSING_SIGNAL) {
                    AddExtraTextToBoxes(printName, detectedDistantRoadCrossingSignals, detectedTriangleSignalIndex, distantRoadCrossingSignalClassificationObjNames);
                }

                if (i.distance > -1) {
                    printName += " (" + std::to_string(i.distance) + " m)";
                }

                // Draw text box and object name
                float printTextSize = 0.6;
                int textThickness = 2;

                cv::Size const text_size = getTextSize(printName, cv::FONT_HERSHEY_SIMPLEX, printTextSize, textThickness, 0);
                int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
                max_width = std::max(int(max_width), (int)i.w + 2);

                cv::rectangle(overlayImage, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 10 - (int)text_size.height, 0)),
                    cv::Point2f(std::min((int)i.x + max_width, overlayImage.cols - 1), std::min((int)i.y, overlayImage.rows - 1)),
                    color, CV_FILLED, 8, 0);
                putText(overlayImage, printName, cv::Point2f(i.x, i.y - 5), cv::FONT_HERSHEY_SIMPLEX, printTextSize, cv::Scalar(0, 0, 0), textThickness, cv::LINE_AA);
            }
        }

        cv::addWeighted(image, 0.55, overlayImage, 0.45, 0, image);
    }

#pragma endregion Draw Bounding Boxes


    //  ######## ######## ######## ######## Setup ######## ######## ######## ########
#pragma region Setup

    std::vector<std::string> GetObjectsNamesFromFile(std::string const filename) {
        std::ifstream file(filename);
        std::vector<std::string> file_lines;
        if (!file.is_open()) {
            cout << "Error loading names file: File not found: " << filename << endl;
            exit(-1);
        }
#ifdef _WIN32
        for (std::string line; getline(file, line);) file_lines.push_back(line);
#else
        for (std::string line; getline(file, line);) file_lines.push_back(line.erase(line.size() - 1));
#endif
        return file_lines;
    }

    void SetupYoloNetwork() {
        ObjectNames = GetObjectsNamesFromFile(yoloNamesFile);
#ifdef TENSORRT
        cout << "Building YOLO network on GPU with TensorRT" << endl;
        yolo_tensorrt::BuildNetwork(yoloEnginePath, ObjectNames.size(), DETECTED_OBJECT_THRESHOLD, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT);
#else
        cout << "Building YOLO network on GPU with Darknet" << endl;
        detector = new Detector(yoloCfgFile, yoloWeightsFile);
#endif
    }

    void SetupSpeedSignClassification() {
        speedSignClassificationObjNames = GetObjectsNamesFromFile(speedSignClassificationNamesFile);
        speedSignClassificationModel = cv::dnn::readNetFromONNX(speedSignClassificationSavedModelPathPyTorchOnnx);
        for (int i = 0; i < speedSignClassificationObjNames.size(); i++) {
            detectedSpeedSignClassesCounter.push_back(0);
        }
        for (int i = 0; i < speedSignClassificationObjNames.size(); i++) {
            lastFrameWithSpeedSign.push_back(0);
        }
    }

    void SetupClassification(std::vector<std::string> &detectedClassNames, std::string classificationNamesFile, std::vector<std::string> &compositeClasseNames, std::vector<int> &classesCounter, std::vector<int> &compositeClassesCounter, std::vector<int> &lastFrame, std::string savedModelPath, cv::dnn::Net &model){
        detectedClassNames = GetObjectsNamesFromFile(classificationNamesFile);
        for (int i = 0; i < detectedClassNames.size(); i++) {
            classesCounter.push_back(0);
        }
        for (int i = 0; i < compositeClasseNames.size(); i++) {
            compositeClassesCounter.push_back(0);
        }
        for (int i = 0; i < detectedClassNames.size(); i++) {
            lastFrame.push_back(0);
        }
        model = cv::dnn::readNetFromONNX(savedModelPath);
    }

    void SetupNeuralNetworks() {
        SetupYoloNetwork();
        SetupSpeedSignClassification();
        SetupClassification(detectedMainSignalClassNames, mainSignalClassificationNamesFile, compositeMainSignalClasseNames, detectedMainSignalClassesCounter, compositeMainSignalClassesCounter, lastFrameWithMainSignal, mainSignalClassificationSavedModelPathPyTorchOnnx, mainSignalClassificationModel);
        SetupClassification(detectedRoadCrossingSignalClassNames, roadCrossingSignalClassificationNamesFile, compositeRoadCrossingSignalClasseNames, detectedRoadCrossingSignalClassesCounter, compositeRoadCrossingSignalClassesCounter, lastFrameWithRoadCrossingSignal, roadCrossingSignalClassificationSavedModelPathPyTorchOnnx, roadCrossingSignalClassificationModel);
        SetupClassification(distantRoadCrossingSignalClassificationObjNames, distantRoadCrossingSignalClassificationNamesFile, compositeDistantRoadCrossingSignalClasseNames, detectedDistantRoadCrossingSignalClassesCounter, compositeDistantRoadCrossingSignalClassesCounter, lastFrameWithDistantRoadCrossingSignal, distantRoadCrossingSignalClassificationSavedModelPathPyTorchOnnx, distantRoadCrossingSignalClassificationModel);
        SetupClassification(distantSignalClassificationObjNames, distantSignalClassificationNamesFile, compositeDistantSignalClasseNames, detectedDistantSignalClassesCounter, compositeDistantSignalClassesCounter, lastFrameWithDistantSignal, distantSignalClassificationSavedModelPathPyTorchOnnx, distantSignalClassificationModel);
    }

    void Cleanup() {
#ifdef TENSORRT
        yolo_tensorrt::Destroy();
#endif
    }

#pragma endregion Setup

}