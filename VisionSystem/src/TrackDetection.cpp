
// About
/*
* TrackDetection.cpp
* 
* Detects tracks and switches with a region growing algorithm based on the magnitude of the derivatives. 
* 
* Can run independently (RunTrackDetection.cpp) or together with object detection (VisionSystem.cpp).
* 
* The function DetectTracks() is the main function in this file to detect tracks. 
* First initialize the data with InizializeArrays(), before detecting with DetectTracks() for the first time.
* Use OutputGlobalInformation() to get the global variables in GlobalInformation.h for the vision system.
* Use DrawOverlayImages() to draw the tracks, switches and the warning zone on an output image.
*/

// Structure
/*
* Variables
*   -Parameters and Constants
*   -Global Variables
* 
* Utilities Functions
* 
* Track Detection
*   -Region Growing Algorithm
*   -Pair Rails to Tracks
*   -Lane Status
*   -Find Branches in Track
*   -Switch Detection
*   -Warning Zone
*   -Track Detection Algorithm <-- the main section
* 
* Output Global Information
* 
* Drawing Functions
* 
* Initialize Functions
* 
* Stand Alone Track Detection Program
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>
#include <thread> 
#include <string>
#include <fstream>

#include "SharedParameters.h"
#include "GlobalEnumerationConstants.h"
#include "GlobalInformation.h"
#include "ObjectDetection.h"

using namespace std;
using namespace cv;

namespace track_detection {

    // ######## ######## ######## ######## Parameters and Constants ######## ######## ######## ######## 
#pragma region Parameters and Constants
    
    // Image size
    const int DETECTION_IMAGE_WIDTH = 1280; // Used image width for the track detection
    const int DETECTION_IMAGE_HEIGHT = 720; // Used image height for the track detection
    const int DETECTION_IMAGE_AREA = DETECTION_IMAGE_WIDTH * DETECTION_IMAGE_HEIGHT;
    const int OUTPUT_IMAGE_AREA = OUTPUT_IMAGE_WIDTH * OUTPUT_IMAGE_HEIGHT;
    const float OUTPUT_TO_DETECTION_WIDTH_RATIO = OUTPUT_IMAGE_WIDTH / (float)DETECTION_IMAGE_WIDTH;
    const float OUTPUT_TO_DETECTION_HEIGHT_RATIO = OUTPUT_IMAGE_HEIGHT / (float)DETECTION_IMAGE_HEIGHT;

    // Preprocessing
    const int GAUSSIAN_BLUR_KERNEL_SIZE = 9; // Blur kernel size, used on bottom half of image
    const int SOBEL_KERNEL_SIZE = 3; // Derivative kernel size

    // Growing algorithm
    const int SEED_SKIP_NUMBER = 5; // The i_th starting point on the bottom/sides
    const int MAX_NEIGHBOUR_DISTANCE = 3; // Largest distance to neighbor to the right and left

    float magnitudeThreshold = 80; // Adaptive

    int horizonRow; // horizon pixel height
    int distanceBetweenTrackPairs;   // Estimation of MAIN track width at bottom row
    int bottomPixelsToCrop; // Pixels to crop at the bottom

    const int MIN_SIZE_OF_GROW_LIST = 100; // Smallest size of growList
    int minDistanceBetweenObjects = 60; // Two tracks with starting points within this distance is set to the same track
    const int GROW_LIST_ALLOCATED_ARRAY_SIZE = 30000;
    int minHeightOfRailInGrowList; // The minimum height above the image bottom for a object to be detected

    // Find adaptive threshold for grow list
    float adaptiveThresholdStartWidth;
    float adaptiveThresholdStartHeight = 0.98;

    // Track processing
    const int TRACK_PAIR_FOUND_COUNTER_THRESHOLD = 20; // Counter value at which certain of left/right track pair
    const int TRACK_PAIR_MAX_COUNTER = 60; // Maximum value of side tracks counter

    int scanImageForBranchHeight; 
    const float BRANCHING_THRESHOLD_MARGIN = 1.3; // When distance between two branches is large enough to be a branch
    const float LANE_THRESHOLD_MARGIN = 0.6; // When width between the branches is wide enough so there is a intersection point with another track
    const int N_ROWS_BRANCHED_THRESHOLD = 20 * 720.0 / DETECTION_IMAGE_HEIGHT; // Min number of branched rows for branch detection

    int distanceBetweenMainTracksMargin = 50; // Error margin for distance between MAIN rails
    const int MAIN_TRACK_DISTANCE_TO_IMAGE_MIDDLE_MARGIN = 200 * DETECTION_IMAGE_WIDTH / 1280.0; // Error margin for MAIN rail middle from image middle 

    int distanceBetweenSideTracksMargin = distanceBetweenMainTracksMargin * 1.5; // Error margin for distance between SIDE rails

    float distanceBetweenTracksNarrowingConstant; // Used to estimate track width at certain row
    int estimatedRailWidth = 25; 
    float railWidthNarrowingConstant; 

    // Switch
    const int SWITCH_DETECTION_MAX_EVALUATION_TIME = 100; // Maximum time (in frames) after possible switch for branching to be found
    const int INCOMING_SWITCH_DETECTION_TIME = 50; // Number of frames that switch text is displayed, 
    const int SWITCH_DETECTION_MIN_TIME_TO_DETECT_NEXT_SWITCH_MERGE_POINT = 15; // Time from branch found after switch until that branch used to evaluate switch
    const int SWITCH_DIRECTION_THRESHOLD = 7;
    const int SWITCH_DIRECTION_DELAY = 7;
    const int INCOMING_SWITCH_THRESHOLD = 7;

    // Distance
    const int MAX_DISTANCE = 200; // Maximum distance returned (meters)
    const int C1 = 900 * 1.435; // C1 = metersToPixels * focalLength * trackWidthInMeters, normal track width = 1.435 m

    // WarningZone
    const float WARNING_ZOONE_MARGIN = 2;
    const bool ALWAYS_DRAW_WARNING_ZONE = false; // only for running track detection independently

#pragma endregion Parameters and Constants


    // ######## ######## ######## ######## Global Variables ######## ######## ######## ######## 
#pragma region Global Variables

    //int framesPerSecond = 0;
    string windowName;
    //int nStartPoints;
    int minTrackViewDistance;
    int maxTrackViewDistance;
    bool trackDetectionStandAlone = false; // Sets automatically to true if run from RunTrackDetection
    bool drawWarningZone = false;
    
    cv::Size imageSize;
    cv::Size outputImageSize;
    cv::Mat resizedImage;
    cv::Mat grayScaleImage;
    cv::Mat xDerivative;
    cv::Mat yDerivative;
    cv::Mat magnitudeMat;
    cv::Mat classifiedPixelsMat;
    cv::Mat outputImage;

    cv::Mat detectionOverlayImage;
    cv::Mat outputOverlayImage;
    cv::Mat tmpOutputOverlayImage;
    cv::Mat infoBoxOverlayImage;
    cv::Mat warningZoneMat;

    cv::Rect blurMask;
    cv::Rect sobelMask;
    cv::Mat verticalBlurMaskMat;
    bool verticalMaskExists = false;

    std::vector<cv::Point> growList;
    std::vector<cv::Point> startPoints;
    cv::VideoWriter video;
    std::vector<cv::Point> trackStartPoints;
    std::vector<int> leftAndRightTrackIndex;

    // Track drawing
    cv::Scalar mainTrackColor = cv::Scalar(0, 0, 255);      // Red
    cv::Scalar sideTrackColor = cv::Scalar(0, 255, 0);      // Green

    // Switch drawing
    cv::Scalar switchDrawColor = cv::Scalar(0, 215, 255);   // Gold
    int switchDotSize = 5;

    // Constants for lane status
    int laneStatus = LANE_STATUS::SINGLE_TRACK;

    // Track width, track middle etc
    std::vector<int> trackWidths;
    std::vector<int> trackMiddles;
    int trackHighestRow = 0;

    // Struct used for pairing of tracks
    struct TrackPairs {
        int leftMainTrackIndex;
        int rightMainTrackIndex;

        bool mainTrackPairFound;

        int leftTrackPairFoundCounter = 0;
        int rightTrackPairFoundCounter = 0;
    };
    TrackPairs trackPairs;

    // Variables connecting to if a switch on the main track is found
    struct TrackSwitch {
        cv::Point leftTrackBranch;
        cv::Point leftTrackIntersection;
        cv::Point leftTrackIntersectionMirror;
        cv::Point rightTrackBranch;
        cv::Point rightTrackIntersection;
        cv::Point rightTrackIntersectionMirror;
    };
    TrackSwitch mainTrackSwitch;

    // Variables for switch detection. Can be used outside this program
    struct SwitchDetection {
        bool switchDetected = false;
        bool switchDirectionDetected = false;
        bool switchedLeft = false;
        bool switchedRight = false;
        bool branchFound = false;
        int lastFrameDetected = -10000;
        int lastFrameBranchFound = -10000;
        int beforeSwitchLaneStatus = -1;
        bool incomingSwitchLeft = false;
        bool incomingSwitchRight = false;
        int lastFrameIncomingSwitchDetected = -10000;
        int incomingSwitchCounterLeft = 0;
        int incomingSwitchCounterRight = 0;
        std::vector<int> sideBranchingCounters;
        SwitchDetection() {
            sideBranchingCounters = std::vector<int>({ 0, 0 });
        }
    };
    SwitchDetection switchDetection;

    // Struct for a track pair used when finding all possible main track pairs
    struct TrackPair {
        cv::Point leftStartPoint;
        cv::Point rightStartPoint;
        int leftTrackIndex;
        int rightTrackIndex;
    };
    std::vector<TrackPair> possibleMainTracks;

#pragma endregion Global Variables


    // ######## ######## ######## ######## Utilities Functions ######## ######## ######## ######## 
#pragma region Utilities Functions

    float GetExportDetectionWidthRatio() {
        return OUTPUT_TO_DETECTION_WIDTH_RATIO;
    }

    float GetExportDetectionHeightRatio() {
        return OUTPUT_TO_DETECTION_HEIGHT_RATIO;
    }

    int GetTrackHighestRow() {
        return trackHighestRow;
    }

    // Find the most left and right pixels in a specific row that belongs to specified track
    inline void FindFirstAndLastNonzero(int rowIndex, uchar*& matrix, int* minCol, int* maxCol, int trackIndex, int startCol = 0, int endCol = DETECTION_IMAGE_WIDTH) {
        *minCol = DETECTION_IMAGE_WIDTH;
        *maxCol = 0;
        for (int col = startCol; col < endCol; col++) {
            if (matrix[col + rowIndex * DETECTION_IMAGE_WIDTH] == trackIndex) {
                if (col < *minCol)
                    *minCol = col;
                if (col > *maxCol)
                    *maxCol = col;
            }
        }
        return;
    }

    // ######## ######## Resizing images ######## ######## 

    /*
    * Scale an image to track detection size
    */
    void ScaleImage(Mat& image)
    {
        int inputImageWidth = image.size[1];
        int inputImageHeight = image.size[0];

        if (DETECTION_IMAGE_WIDTH != inputImageWidth || DETECTION_IMAGE_HEIGHT != inputImageHeight) {
            cv::resize(image, resizedImage, imageSize);
        }
        else {
            image.copyTo(resizedImage);
        }
    }

    /*
    * Resizes an image with track detection size to and image with output size.
    */
    void ResizeToOutput(cv::Mat inputImage, cv::Mat outputImage) {
        if (imageSize != outputImageSize) {
            cv::resize(inputImage, outputImage, outputImageSize);
        }
        else {
            inputImage.copyTo(outputImage);
        }
    }

    // ######## ######## Track Width ######## ######## 

    // Estimate the rail width at a certain row in the image
    inline float CalculateRailWidth(int row) {
        float railWidth;
        if (row > DETECTION_IMAGE_HEIGHT - horizonRow)
            railWidth = -railWidthNarrowingConstant * (DETECTION_IMAGE_HEIGHT - row) + estimatedRailWidth * DETECTION_IMAGE_WIDTH / 1280;
        else
            railWidth = 0;
        return railWidth;
    }

    // Estimate the track width at a certain row in the image
    inline int CalculateTrackWidth(int row) {
        float trackWidth;
        if (row >= DETECTION_IMAGE_HEIGHT - horizonRow) {
            trackWidth = -distanceBetweenTracksNarrowingConstant * (DETECTION_IMAGE_HEIGHT - row) + distanceBetweenTrackPairs * DETECTION_IMAGE_WIDTH / 1280;
            trackWidth = std::roundf(trackWidth);
            trackWidth = std::max((float)1, trackWidth);
        }
        else
            trackWidth = 0;
        return trackWidth;
    }

    // Compute the track widths for all rows in the image
    void CalculateAllTrackWidths(){
        for (int row = 0; row < DETECTION_IMAGE_HEIGHT; row++){
            trackWidths.push_back(CalculateTrackWidth(row));
        }
    }

    /*
    Returns the track width for a certain row.
    Returns the latest track width if row is above the horizon.
    */
    int GetTrackWidth(int row) {
        if (row < DETECTION_IMAGE_HEIGHT - horizonRow) {
            return 0;
        }
        return trackWidths[row];
    }

    // ######## ######## Track Middle ######## ######## 

    int CalculateTrackMiddle(int row, uchar*& matrix, int trackIndex, int startCol = 0, int endCol = DETECTION_IMAGE_WIDTH) {
        int average = 0;
        int numberOfPoints = 0;
        for (int col = startCol; col < endCol; col++) {
            if (matrix[col + row * DETECTION_IMAGE_WIDTH] == trackIndex) {
                numberOfPoints++;
                average += col;
            }
        }
        return average / (float)numberOfPoints;
    }

    void CalculateAllTrackMiddles() {
        int leftCol;
        int rightCol;
        int tmp;
        int lastFoundMiddle = DETECTION_IMAGE_WIDTH / 2;
        uchar* classifiedPixels = (uchar*)classifiedPixelsMat.data;

        for (int row = DETECTION_IMAGE_HEIGHT - 1; row >= DETECTION_IMAGE_HEIGHT - horizonRow; row--) {
            FindFirstAndLastNonzero(row, classifiedPixels, &leftCol, &tmp, trackPairs.leftMainTrackIndex);
            FindFirstAndLastNonzero(row, classifiedPixels, &tmp, &rightCol, trackPairs.rightMainTrackIndex);
            int trackWidth = trackWidths[row];
            int middle = DETECTION_IMAGE_WIDTH / 2.0;
            if (leftCol != DETECTION_IMAGE_WIDTH && rightCol != 0) {
                middle = (leftCol + rightCol) / 2.0;
                trackHighestRow = row;
            }
            else if (leftCol != DETECTION_IMAGE_WIDTH) {
                middle = leftCol + trackWidth / 2.0;
                trackHighestRow = row;
            }
            else if (rightCol != 0) {
                middle = rightCol - trackWidth / 2.0;
                trackHighestRow = row;
            }
            else {
                middle = lastFoundMiddle;
            }
            lastFoundMiddle = middle;
            trackMiddles[row] = middle;
        }
    }

    /*
    Returns the track middle for a certain row.
    Returns the latest track middle if row is above the horizon.
    */
    int GetTrackMiddle(int row) {
        if (row < DETECTION_IMAGE_HEIGHT - horizonRow) {
            return trackMiddles[trackHighestRow];
        }
        return trackMiddles[row];
    }

    void GetTrackPosition(int row, int& trackWidth, int& trackMiddle) {
        trackWidth = GetTrackWidth(row);
        trackMiddle = GetTrackMiddle(row);
    }

    // ######## ######## Distance ######## ######## 

    /* 
    Estimate the actual distance in units of meters to a row in the image by
    using the track width for that row.
    distance = C2 / trackwidth,     where C2 = metersToPixels * focalLength * trackWidthInMeters
    */
    int GetDistance(int row) {
        if (row == 0) {
            return 0;
        }
        else if (row < DETECTION_IMAGE_HEIGHT - horizonRow) {
            return MAX_DISTANCE;
        }
        int distance = C1 / GetTrackWidth(row);
        distance = std::max(distance, 0);
        distance = std::min(distance, MAX_DISTANCE);
        return distance;
    }

    // ######## ######## Blur Mask ######## ######## 

    // Compute the vertical blur mask based on the already detected main track
    void CreateVerticalBlurMask() {
        float margin = 2;
        int upperRow = scanImageForBranchHeight;
        int trackWidthUpper = GetTrackWidth(upperRow);
        int trackWidthBottom = GetTrackWidth(DETECTION_IMAGE_HEIGHT - 1);
        int trackMiddleUpper = GetTrackMiddle(DETECTION_IMAGE_HEIGHT - 1);
        int trackMiddleBottom = GetTrackMiddle(upperRow);
        memset(verticalBlurMaskMat.data, 0, DETECTION_IMAGE_AREA * sizeof(*verticalBlurMaskMat.data));

        cv::Point leftBottom = cv::Point(trackMiddleUpper - trackWidthBottom / 2.0 * margin, DETECTION_IMAGE_HEIGHT - 1);
        cv::Point rightBottom = cv::Point(trackMiddleUpper + trackWidthBottom / 2.0 * margin, DETECTION_IMAGE_HEIGHT - 1);
        cv::Point upperRight = cv::Point(trackMiddleBottom + trackWidthUpper / 2.0 * margin, upperRow);
        cv::Point upperLeft = cv::Point(trackMiddleBottom - trackWidthUpper / 2.0 * margin, upperRow);
        std::vector<cv::Point> points = { leftBottom, rightBottom, upperRight, upperLeft, leftBottom };
        cv::fillPoly(verticalBlurMaskMat, points, cv::Scalar(255));
        verticalMaskExists = true;
    }

#pragma endregion Utilities Functions
    

    // ######## ######## ######## ######## Region Growing Algorithm ######## ######## ######## ######## 
#pragma region Region Growing Algorithm

    /*
    * Adapts the magnitude threshold for the region growing algorithm to the current environment.
    * The magnitude threshold is set by the max and mean of the magnitude within a box at the bottom of the image.
    */
    float CalculateAdaptiveMagnitudeThresholdWithinRegionMax(float* magnitudes) { 
        float maxMagnitude = 0;
        int meanMagnitude = 0;
        int nPixels = 0;

        // Compute max and mean gradient magnitude in centered rectangle in bottom om image
        for (int row = DETECTION_IMAGE_HEIGHT * adaptiveThresholdStartHeight; row < DETECTION_IMAGE_HEIGHT; row++) {
            for (int col = DETECTION_IMAGE_WIDTH * adaptiveThresholdStartWidth; col < DETECTION_IMAGE_WIDTH * (1 - adaptiveThresholdStartWidth); col++) {
                float magnitude = magnitudes[col + row * DETECTION_IMAGE_WIDTH];
                maxMagnitude = std::max(maxMagnitude, magnitude);

                meanMagnitude += magnitude;
                nPixels++;
            }
        }
        meanMagnitude /= nPixels;

        // The basic magnitude factor
        float magnitudeFactor = 0.25;

        //The basic lowest magnitude threshold
        float minMagnitudeThreshold = 55;

        // Increase threshold if large mean
        if (meanMagnitude > 20){
            magnitudeFactor = 0.35;
        }

        // Reduce the lowest magnitude if too low mean
        if (meanMagnitude < 13) {
            minMagnitudeThreshold = 40;
        }
        
        // Compute and limit the magnitude threshold
        magnitudeThreshold = maxMagnitude * magnitudeFactor;
        magnitudeThreshold = std::max(std::min(magnitudeThreshold, (float)200.0), minMagnitudeThreshold);
        return magnitudeThreshold;
    }

    /*
    * Growing algorithm to find rails in the image. 
    * Starts with seeds at the bottom or sides of the image, grows upwards by adding neighbours.
    * Only neighbours with a large magnitude threshold is added.
    * Seeds close to each others combines into a single rail. 
    */
    int RegionGrowingAlgorithm(float* magnitude, uchar* classifiedPixels) {
        int nTracks = 0;
        cv::Point currentPixel;
        cv::Point neighbourPixel;
        cv::Point lastFoundSeed = cv::Point(0, 0);

        // Convert vertical blur mask to usable form
        uchar* verticalBlurMask = (uchar*)verticalBlurMaskMat.data;

        // Reset the pixel classifications from earlier frames
        memset(classifiedPixels, 0, DETECTION_IMAGE_AREA * sizeof(*classifiedPixels));

        // Compute the adaptive threshold
        magnitudeThreshold = CalculateAdaptiveMagnitudeThresholdWithinRegionMax(magnitude);

        // Loop over all start seed points
        for (int seedIndex = 0; seedIndex < startPoints.size(); seedIndex++) {
            int start = 0;
            int end = 0;
            growList.clear();

            // Set and classify the seed point
            cv::Point startSeedPoint = startPoints[seedIndex];
            growList.push_back(startSeedPoint);
            classifiedPixels[startSeedPoint.y + startSeedPoint.x * DETECTION_IMAGE_WIDTH] = nTracks + 1;

            // Loop over all pixels in the growth list
            while (start <= end) {
                // Load current pixel
                currentPixel = growList[start];
                float magnitudeValue = magnitude[currentPixel.y + currentPixel.x * DETECTION_IMAGE_WIDTH];

                // Do not evaluate pixels above horizon
                if (currentPixel.x + 1 < DETECTION_IMAGE_HEIGHT - horizonRow) {
                    start++;
                    continue;
                }            

                // Reduce the maximum neighborhood distance if close to horizon
                int neighbourDistance = MAX_NEIGHBOUR_DISTANCE;
                if (currentPixel.x < DETECTION_IMAGE_HEIGHT - horizonRow + 20){
                    neighbourDistance = 1;
                }
                
                // Change the magnitude threshold depending on height in image and if within vertical blur region.
                // Should be easier to detect tracks within the vertical blur region and at the bottom.
                float currentMagnitudeThreshold = magnitudeThreshold;
                bool insideVerticalBlurMask = false;
                if (verticalMaskExists && verticalBlurMask[currentPixel.y + currentPixel.x * DETECTION_IMAGE_WIDTH] == 0) {
                    insideVerticalBlurMask = true;
                }
                if (!insideVerticalBlurMask){
                    currentMagnitudeThreshold = currentMagnitudeThreshold * 0.5;
                    if (currentPixel.x < (DETECTION_IMAGE_HEIGHT - 0.6 * horizonRow)) {
                        currentMagnitudeThreshold = currentMagnitudeThreshold * 4.0;
                    }
                }
                else {
                    if (currentPixel.x > (DETECTION_IMAGE_HEIGHT - 0.2 * horizonRow)) {
                        currentMagnitudeThreshold = currentMagnitudeThreshold * 0.7;
                    }
                }

                // Loop over all neighbors of the current pixel
                for (int neighbourDifference = -neighbourDistance; neighbourDifference < 1 + neighbourDistance; neighbourDifference++) {
                    neighbourPixel = cv::Point(currentPixel.x - 1, currentPixel.y + neighbourDifference);

                    // Do not evaluate if outside of image or above horizon
                    if (neighbourPixel.y < 0 || neighbourPixel.y >= DETECTION_IMAGE_WIDTH || neighbourPixel.x < DETECTION_IMAGE_HEIGHT - horizonRow) {
                        continue;
                    }

                    // Check if neighbor already have been evaluated
                    if (classifiedPixels[neighbourPixel.y + neighbourPixel.x * DETECTION_IMAGE_WIDTH] == 0) {
                        float neighbourMagnitudeValue = magnitude[neighbourPixel.y + neighbourPixel.x * DETECTION_IMAGE_WIDTH];

                        // Do not add neighbor to growth list if its magnitude too low
                        if (neighbourMagnitudeValue < currentMagnitudeThreshold) {
                            continue;
                        }

                        // Else classify and add neighbour
                        growList.push_back(neighbourPixel);
                        classifiedPixels[neighbourPixel.y + neighbourPixel.x * DETECTION_IMAGE_WIDTH] = nTracks + 1;
                        end++;
                    }
                }
                start++;
            }

            // Check requirements on the found possible rail
            int sizeOfGrowList = growList.size();
            bool railTooCloseToPreviousRail = sqrt(pow(startSeedPoint.x - lastFoundSeed.x, 2) + pow(startSeedPoint.y - lastFoundSeed.y, 2)) < minDistanceBetweenObjects;
            if (sizeOfGrowList > MIN_SIZE_OF_GROW_LIST && railTooCloseToPreviousRail) {// Join valid rail with earlier rail since too close
                // Seed point close to another rail, combine those. 
                for (int i = 0; i < sizeOfGrowList; i++) {
                    classifiedPixels[growList[i].y + growList[i].x * DETECTION_IMAGE_WIDTH] = nTracks;
                }
            }
            else if (sizeOfGrowList < MIN_SIZE_OF_GROW_LIST || growList[sizeOfGrowList - 1].x > minHeightOfRailInGrowList) {// Delete rail if too small or too short
                // Rail too small or too short, reset.
                for (int i = 0; i < sizeOfGrowList; i++) {
                    classifiedPixels[growList[i].y + growList[i].x * DETECTION_IMAGE_WIDTH] = 0;
                }
            }
            else {
                // Add the rail as a new rail
                nTracks++;
                lastFoundSeed = startSeedPoint;
            }
        }
        return nTracks;
    }

#pragma endregion Region Growing Algorithm


    // ######## ######## ######## ######## Pair Rails to Tracks ######## ######## ######## ######## 
#pragma region Pair Rails to Tracks

    /*
    * Finds tracks by paring rails that have a distance between them equal to the track width.
    * First pairs the main track by finding the par of rails closest to the middle.
    * Then finds the side tracks, first by checking the starting points on the bottom of the image,
    * secondly by checking the start points on the side of the image.
    */
    void PairRailsToTracks(uchar* classifiedPixels) {
        trackPairs.leftTrackPairFoundCounter = std::max(0, trackPairs.leftTrackPairFoundCounter - 1);
        trackPairs.rightTrackPairFoundCounter = std::max(0, trackPairs.rightTrackPairFoundCounter - 1);

        // Find starting points of all detected rails
        int currentClassFound = 0;
        trackPairs.leftMainTrackIndex = -1;
        trackPairs.rightMainTrackIndex = -1;
        trackStartPoints.clear();
        for (int iStartPoint = 0; iStartPoint < startPoints.size(); iStartPoint++) {
            cv::Point startPoint = startPoints[iStartPoint];
            int classIndex = (int)classifiedPixels[startPoint.y + startPoint.x * DETECTION_IMAGE_WIDTH];
            if (classIndex > currentClassFound) {
                currentClassFound = classIndex;
                trackStartPoints.push_back(startPoint);
            }
        }

        // Loop over all found tracks to find pair that might be main track
        possibleMainTracks.clear();
        for (int iFirstTrackStartPoint = 0; iFirstTrackStartPoint < trackStartPoints.size(); iFirstTrackStartPoint++) {
            cv::Point firstTrackStartPoint = trackStartPoints[iFirstTrackStartPoint];

            for (int iOtherTrackStartPoint = iFirstTrackStartPoint + 1; iOtherTrackStartPoint < trackStartPoints.size(); iOtherTrackStartPoint++) {
                cv::Point otherTrackStartPoint = trackStartPoints[iOtherTrackStartPoint];

                if (iOtherTrackStartPoint == iFirstTrackStartPoint)
                    continue;

                // Check if possible main track
                if (firstTrackStartPoint.x == DETECTION_IMAGE_HEIGHT - 1 && otherTrackStartPoint.x == DETECTION_IMAGE_HEIGHT - 1) {
                    int distanceBetweenTracks = std::abs(firstTrackStartPoint.y - otherTrackStartPoint.y);

                    if (std::abs(distanceBetweenTracks - distanceBetweenTrackPairs) < distanceBetweenMainTracksMargin) {
                        int distanceToMiddle = std::abs((firstTrackStartPoint.y + otherTrackStartPoint.y) / 2 - DETECTION_IMAGE_WIDTH / 2.0);

                        if (distanceToMiddle < MAIN_TRACK_DISTANCE_TO_IMAGE_MIDDLE_MARGIN) {
                            //Store info about potential main track
                            TrackPair newPossibleMainTrackPair = TrackPair();
                            if (firstTrackStartPoint.y < otherTrackStartPoint.y) {
                                newPossibleMainTrackPair.leftStartPoint = firstTrackStartPoint;
                                newPossibleMainTrackPair.rightStartPoint = otherTrackStartPoint;
                                newPossibleMainTrackPair.leftTrackIndex = iFirstTrackStartPoint + 1;
                                newPossibleMainTrackPair.rightTrackIndex = iOtherTrackStartPoint + 1;
                            }
                            else {
                                newPossibleMainTrackPair.leftStartPoint = otherTrackStartPoint;
                                newPossibleMainTrackPair.rightStartPoint = firstTrackStartPoint;
                                newPossibleMainTrackPair.leftTrackIndex = iOtherTrackStartPoint + 1;
                                newPossibleMainTrackPair.rightTrackIndex = iFirstTrackStartPoint + 1;
                            }
                            possibleMainTracks.push_back(newPossibleMainTrackPair);
                        }
                    }
                }
            }
        }

        // Find the track from above that is the most probable (most centered) to be main track
        int smallestDistanceToMiddle = DETECTION_IMAGE_WIDTH;
        int smallestPossibleMainTrackIndex = -1;
        for (int iPossibleMainTrack = 0; iPossibleMainTrack < possibleMainTracks.size(); iPossibleMainTrack++) {
            int centerOfTrackPair = (int)((possibleMainTracks[iPossibleMainTrack].leftStartPoint.y + possibleMainTracks[iPossibleMainTrack].rightStartPoint.y) / 2.0);
            int distanceToCenter = std::abs(centerOfTrackPair - DETECTION_IMAGE_WIDTH / 2);

            if (distanceToCenter < smallestDistanceToMiddle) {
                smallestDistanceToMiddle = distanceToCenter;
                smallestPossibleMainTrackIndex = iPossibleMainTrack;
            }
        }

        // Set the found track as main track
        if (smallestPossibleMainTrackIndex >= 0) {
            TrackPair mainTrackPair = possibleMainTracks[smallestPossibleMainTrackIndex];
            trackPairs.leftMainTrackIndex = mainTrackPair.leftTrackIndex;
            trackPairs.rightMainTrackIndex = mainTrackPair.rightTrackIndex;
        }

        leftAndRightTrackIndex.clear();

        // Loop over all found tracks to find pair that might be left/right track
        for (int iFirstTrackStartPoint = 0; iFirstTrackStartPoint < trackStartPoints.size(); iFirstTrackStartPoint++) {
            cv::Point firstTrackStartPoint = trackStartPoints[iFirstTrackStartPoint];

            for (int iOtherTrackStartPoint = iFirstTrackStartPoint + 1; iOtherTrackStartPoint < trackStartPoints.size(); iOtherTrackStartPoint++) {
                cv::Point otherTrackStartPoint = trackStartPoints[iOtherTrackStartPoint];

                //Check if track already done
                if (iOtherTrackStartPoint == iFirstTrackStartPoint)
                    continue;
                else if (iFirstTrackStartPoint + 1 == trackPairs.leftMainTrackIndex || iFirstTrackStartPoint + 1 == trackPairs.rightMainTrackIndex)
                    continue;
                else if (iOtherTrackStartPoint + 1 == trackPairs.leftMainTrackIndex || iOtherTrackStartPoint + 1 == trackPairs.rightMainTrackIndex)
                    continue;

                bool trackAlreadyPaired = false;
                for (int i = 0; i < leftAndRightTrackIndex.size(); i++) {
                    if (leftAndRightTrackIndex[i] == iFirstTrackStartPoint + 1 || leftAndRightTrackIndex[i] == iOtherTrackStartPoint + 1) {
                        trackAlreadyPaired = true;
                        break;
                    }
                }
                if (trackAlreadyPaired)
                    continue;

                // Check all track pairs with both rails starting in bottom of image
                if (firstTrackStartPoint.x == DETECTION_IMAGE_HEIGHT - 1 && otherTrackStartPoint.x == DETECTION_IMAGE_HEIGHT - 1) {
                    int distanceBetweenTracks = std::abs(firstTrackStartPoint.y - otherTrackStartPoint.y);

                    if (std::abs(distanceBetweenTracks - distanceBetweenTrackPairs) < distanceBetweenSideTracksMargin) {
                        int middleCol = (int)((firstTrackStartPoint.y + otherTrackStartPoint.y) / 2);
                        if (middleCol < DETECTION_IMAGE_WIDTH / 2.0) {
                            trackPairs.leftTrackPairFoundCounter = std::min(TRACK_PAIR_MAX_COUNTER, trackPairs.leftTrackPairFoundCounter + 2);
                        }
                        else {
                            trackPairs.rightTrackPairFoundCounter = std::min(TRACK_PAIR_MAX_COUNTER, trackPairs.rightTrackPairFoundCounter + 2);
                        }

                        leftAndRightTrackIndex.push_back(iFirstTrackStartPoint + 1);
                        leftAndRightTrackIndex.push_back(iOtherTrackStartPoint + 1);
                    }
                }
                else {// Check all other track pairs that may start at the side of the image
                    // Get row/col of the track starting the highest up in the image
                    int highestStartRow;
                    int trackNumber;
                    int colHighest;
                    if (firstTrackStartPoint.x < otherTrackStartPoint.x) {
                        highestStartRow = firstTrackStartPoint.x;
                        trackNumber = iOtherTrackStartPoint + 1;
                        colHighest = firstTrackStartPoint.y;
                    }
                    else {
                        highestStartRow = otherTrackStartPoint.x;
                        trackNumber = iFirstTrackStartPoint + 1;
                        colHighest = otherTrackStartPoint.y;
                    }

                    int minCol = DETECTION_IMAGE_WIDTH;
                    int maxCol = 0;
                    for (int col = 0; col < DETECTION_IMAGE_WIDTH; col++) {
                        if (classifiedPixels[col + highestStartRow * DETECTION_IMAGE_WIDTH] == trackNumber) {
                            if (col < minCol)
                                minCol = col;
                            if (col > maxCol)
                                maxCol = col;
                        }
                    }

                    // Pair the valid pairs
                    if (maxCol >= 0 || minCol < DETECTION_IMAGE_WIDTH) {
                        int middleCol = int(minCol + (maxCol - minCol) / 2.0);
                        int distanceBetweenTracks = std::abs(middleCol - colHighest);
                        float trackWidth = GetTrackWidth(highestStartRow);

                        if (std::abs(distanceBetweenTracks - trackWidth) < distanceBetweenSideTracksMargin) {
                            if (middleCol < DETECTION_IMAGE_WIDTH / 2.0) {
                                trackPairs.leftTrackPairFoundCounter = std::min(TRACK_PAIR_MAX_COUNTER, trackPairs.leftTrackPairFoundCounter + 2);
                            }
                            else {
                                trackPairs.rightTrackPairFoundCounter = std::min(TRACK_PAIR_MAX_COUNTER, trackPairs.rightTrackPairFoundCounter + 2);
                            }

                            leftAndRightTrackIndex.push_back(iFirstTrackStartPoint + 1);
                            leftAndRightTrackIndex.push_back(iOtherTrackStartPoint + 1);
                        }
                    }
                }
            }
        }
        return;
    }

#pragma endregion Pair Rails to Tracks


    // ######## ######## ######## ######## Lane Status ######## ######## ######## ######## 
#pragma region Lane Status

    int GetLaneStatus() {
        return laneStatus;
    }

    /*
    * Sets if the current track has neighbouring tracks (left/right/middle) or not (single track)
    */
    void UpdateLaneStatus() {
        bool leftTrackFound = trackPairs.leftTrackPairFoundCounter > TRACK_PAIR_FOUND_COUNTER_THRESHOLD;
        bool rightTrackFound = trackPairs.rightTrackPairFoundCounter > TRACK_PAIR_FOUND_COUNTER_THRESHOLD;

        // Include possible info about detected trains when determining track status
        if (!trackDetectionStandAlone) {
            object_detection::Train detectedTrain = object_detection::GetDetectedTrain();

            if (detectedTrain.detected && detectedTrain.insideWarningZone) {
                if (detectedTrain.sideOfTrack == SIDE::RIGHT) {
                    rightTrackFound = true;
                    trackPairs.rightTrackPairFoundCounter = TRACK_PAIR_MAX_COUNTER;
                }
                if (detectedTrain.sideOfTrack == SIDE::LEFT) {
                    leftTrackFound = true;
                    trackPairs.leftTrackPairFoundCounter = TRACK_PAIR_MAX_COUNTER;
                }
            }
        }

        // Determine the current track status
        if (leftTrackFound && rightTrackFound) {
            laneStatus = LANE_STATUS::MIDDLE_TRACK;
        }
        else if (leftTrackFound) {
            laneStatus = LANE_STATUS::RIGHT_TRACK;
        }
        else if (rightTrackFound) {
            laneStatus = LANE_STATUS::LEFT_TRACK;
        }
        else {
            laneStatus = LANE_STATUS::SINGLE_TRACK;
        }
    }

#pragma endregion Lane Status


    // ######## ######## ######## ######## Find Branches in Track ######## ######## ######## ######## 
#pragma region Find Branches in Track

    /*
    * Find if there is any empty middle in a rail.
    */
    inline bool FindEmptyMiddle(int mainMinCol, int mainMaxCol, int row, uchar* classifiedPixels, bool* emptyMiddleRows) {
        bool emptyMiddle = false;
        int nEmptyCols = 0;

        for (int col = mainMinCol; col < mainMaxCol; col++) {
            if (classifiedPixels[col + row * DETECTION_IMAGE_WIDTH] == 0) {
                nEmptyCols += 1;
            }
            else{
                nEmptyCols = 0;
            }

            if (nEmptyCols == 10){
                emptyMiddle = true;
                emptyMiddleRows[row] = true;
                break;
            }
        }
        return emptyMiddle;
    }

    /*
    * Find if there is any non-empty middle in a rail.
    */
    inline bool FindAnyNonEmptyMiddle(int row, bool* emptyMiddleRows) {
        bool anyNonEmptyMiddle = false;
        for (int subRow = row; subRow < DETECTION_IMAGE_HEIGHT - 1; subRow++) {
            if (!emptyMiddleRows[subRow]) {
                anyNonEmptyMiddle = true;
                break;
            }
        }
        return anyNonEmptyMiddle;
    }

    /*
    * Find branches in a track that may indicate a switch is approaching.
    * Loops from the bottom row to limited height.
    * Finds empty middles in the rail robustly -> branch
    */
    void FindBranchesFromTrack(int trackIndex, int* distance) {
        bool branchFound = false;
        bool emptyMiddle = false;
        bool intersectionFound = false;

        int lastColLeft = 0;
        int lastColRight = 0;
        int nRowsBranched = 0;

        int mainMinCol;
        int mainMaxCol;

        int middleLeftBranchCol;
        int middleRightBranchCol;

        int leftBranchMinCol;
        int leftBranchMaxCol;
        int rightBranchMinCol;
        int rightBranchMaxCol;

        uchar* classifiedPixels = (uchar*)classifiedPixelsMat.data;

        int trackHeight = 0;

        bool* emptyMiddleRows = new bool[DETECTION_IMAGE_HEIGHT]();

        for (int row = DETECTION_IMAGE_HEIGHT - 1; row > scanImageForBranchHeight; row--) {
            float trackWidth = GetTrackWidth(row);
            float railWidth = CalculateRailWidth(row);

            FindFirstAndLastNonzero(row, classifiedPixels, &mainMinCol, &mainMaxCol, trackIndex);

            // Check if any pixel in between mainMin and mainMax is zero
            emptyMiddle = FindEmptyMiddle(mainMinCol, mainMaxCol, row, classifiedPixels, emptyMiddleRows);

            // Set the highest rail height
            bool anyTrackPixel = mainMaxCol > 0 || mainMinCol < DETECTION_IMAGE_WIDTH;
            if (anyTrackPixel) {
                trackHeight = row;
            }

            // Check if there is possible branching and increment branching counter
            int middleCol = int(mainMinCol + (mainMaxCol - mainMinCol) / 2.0);
            if (emptyMiddle && (mainMaxCol - mainMinCol) > railWidth * BRANCHING_THRESHOLD_MARGIN) {
                bool anyNonEmptyMiddle = FindAnyNonEmptyMiddle(row, emptyMiddleRows);
                if (anyNonEmptyMiddle) {
                    nRowsBranched += 1;
                    branchFound = true;
                }
            }

            // Possible branch
            if (branchFound && emptyMiddle) {
                // Find if branching point has occurred enough rows to be robust
                if (nRowsBranched == N_ROWS_BRANCHED_THRESHOLD) {
                    int baseBranchMinCol;
                    int baseBranchMaxCol;
                    FindFirstAndLastNonzero(row + nRowsBranched + 1, classifiedPixels, &baseBranchMinCol, &baseBranchMaxCol, trackIndex);
                    int dotCenterColumn = int(baseBranchMinCol + (baseBranchMaxCol - baseBranchMinCol) / 2.0);

                    // Set the position of the robust branch point
                    if (trackIndex == trackPairs.leftMainTrackIndex) {
                        mainTrackSwitch.leftTrackBranch.x = dotCenterColumn;
                        mainTrackSwitch.leftTrackBranch.y = row + nRowsBranched + 1;
                    }
                    else {
                        mainTrackSwitch.rightTrackBranch.x = dotCenterColumn;
                        mainTrackSwitch.rightTrackBranch.y = row + nRowsBranched + 1;
                    }
                }

                //Find left branch middle
                FindFirstAndLastNonzero(row, classifiedPixels, &leftBranchMinCol, &leftBranchMaxCol, trackIndex, 0, middleCol);
                middleLeftBranchCol = int(leftBranchMinCol + (leftBranchMaxCol - leftBranchMinCol) / 2.0);

                //Find right branch middle
                FindFirstAndLastNonzero(row, classifiedPixels, &rightBranchMinCol, &rightBranchMaxCol, trackIndex, middleCol, DETECTION_IMAGE_WIDTH);
                middleRightBranchCol = int(rightBranchMinCol + (rightBranchMaxCol - rightBranchMinCol) / 2.0);

                // Find if the branch is wide enough that the right branch from the left rail may intersect with the left branch from the right rail, and vice versa.
                // A second track has then arisen. 
                if (!intersectionFound && std::abs(middleLeftBranchCol - middleRightBranchCol) > trackWidth * LANE_THRESHOLD_MARGIN) {
                    intersectionFound = true;
                    if (trackIndex == trackPairs.leftMainTrackIndex) {
                        mainTrackSwitch.leftTrackIntersection.x = middleLeftBranchCol + trackWidth;
                        mainTrackSwitch.leftTrackIntersection.y = row;
                        mainTrackSwitch.leftTrackIntersectionMirror.x = middleLeftBranchCol;
                        mainTrackSwitch.leftTrackIntersectionMirror.y = row;
                    }
                    else {
                        mainTrackSwitch.rightTrackIntersection.x = middleRightBranchCol - trackWidth;
                        mainTrackSwitch.rightTrackIntersection.y = row;
                        mainTrackSwitch.rightTrackIntersectionMirror.x = middleRightBranchCol;
                        mainTrackSwitch.rightTrackIntersectionMirror.y = row;
                    }
                }
            }
            else { // No branch or end of the branching
                // If branching was too short to be a true branching, reset
                if (nRowsBranched < N_ROWS_BRANCHED_THRESHOLD) {
                    branchFound = false;
                    nRowsBranched = 0;
                }
            }
        }

        // Find the highest row in the image where the track is present.
        // Calculate the distance to that row.
        // This distance is the view distance for the rail, where the way is free and no obstacle are present. 
        if (trackHeight == scanImageForBranchHeight + 1) {
            for (int row = scanImageForBranchHeight; row > DETECTION_IMAGE_HEIGHT - horizonRow; row--) {
                FindFirstAndLastNonzero(row, classifiedPixels, &mainMinCol, &mainMaxCol, trackIndex);
                bool anyTrackPixel = (mainMaxCol > 0 || mainMinCol < DETECTION_IMAGE_WIDTH);
                if (anyTrackPixel) {
                    trackHeight = row;
                }
            }
        }
        (*distance) = GetDistance(trackHeight);
    }

#pragma endregion Find Branches in Track


    // ######## ######## ######## ######## Switch Detection ######## ######## ######## ######## 
#pragma region Switch Detection

    /*
    * Find if a switch is approaching by the branches found.
    * Find if an incoming track is approaching by the branches found. 
    * Draw the branching points and the switch.
    */
    void ProcessAndDrawSwitch(bool draw) {
        switchDetection.branchFound = false;

        //Draw left track branching points
        if (mainTrackSwitch.leftTrackBranch != cv::Point(0, 0)) {
            if (draw)
                cv::circle(detectionOverlayImage, mainTrackSwitch.leftTrackBranch, switchDotSize, switchDrawColor, -1);

            switchDetection.branchFound = true;

            if (!switchDetection.switchDirectionDetected) {
                switchDetection.lastFrameBranchFound = frameNumber;
            }
            
            // Possible intersection point to find the switch direction
            if (switchDetection.switchDetected && frameNumber - switchDetection.lastFrameDetected > SWITCH_DIRECTION_DELAY && mainTrackSwitch.rightTrackBranch == cv::Point(0, 0)) {
                switchDetection.sideBranchingCounters[SIDE::LEFT]++;
            }
        }

        //Draw right track branching points
        if (mainTrackSwitch.rightTrackBranch != cv::Point(0, 0)) {
            if (draw)
                cv::circle(detectionOverlayImage, mainTrackSwitch.rightTrackBranch, switchDotSize, switchDrawColor, -1);

            switchDetection.branchFound = true;

            if (!switchDetection.switchDirectionDetected) {
                switchDetection.lastFrameBranchFound = frameNumber;
            }

            // Possible intersection point to find the switch direction
            if (switchDetection.switchDetected && frameNumber - switchDetection.lastFrameDetected > SWITCH_DIRECTION_DELAY && mainTrackSwitch.leftTrackBranch == cv::Point(0, 0)) {
                switchDetection.sideBranchingCounters[SIDE::RIGHT]++;
            }
        }

        // Find if Switch is Found, or incoming tracks.
        int minSwitchHeight = 25;
        bool switchPointsExist = mainTrackSwitch.leftTrackIntersection != cv::Point(0, 0) && mainTrackSwitch.rightTrackIntersection != cv::Point(0, 0) && mainTrackSwitch.rightTrackBranch != cv::Point(0, 0) && mainTrackSwitch.leftTrackBranch != cv::Point(0, 0);
        bool rightPositionsOk = (mainTrackSwitch.rightTrackBranch.y - mainTrackSwitch.rightTrackIntersection.y) > minSwitchHeight && (mainTrackSwitch.leftTrackBranch.y - mainTrackSwitch.rightTrackIntersection.y) > minSwitchHeight;
        bool leftPositionsOk = (mainTrackSwitch.rightTrackBranch.y - mainTrackSwitch.leftTrackIntersection.y) > minSwitchHeight && (mainTrackSwitch.leftTrackBranch.y - mainTrackSwitch.leftTrackIntersection.y) > minSwitchHeight;
        if (switchPointsExist && rightPositionsOk && leftPositionsOk) {
            // Switch found
            if (draw) {
                // Draw mirrored point
                cv::circle(detectionOverlayImage, mainTrackSwitch.leftTrackIntersectionMirror, switchDotSize, switchDrawColor, -1);
                cv::circle(detectionOverlayImage, mainTrackSwitch.rightTrackIntersectionMirror, switchDotSize, switchDrawColor, -1);

                // Draw common intersection point
                cv::Point intersectionPoint = cv::Point((mainTrackSwitch.leftTrackIntersection.x + mainTrackSwitch.rightTrackIntersection.x) / 2, (mainTrackSwitch.leftTrackIntersection.y + mainTrackSwitch.rightTrackIntersection.y) / 2);
                cv::circle(detectionOverlayImage, intersectionPoint, switchDotSize, switchDrawColor, -1);
                
                // Draw polygon over switch
                cv::Point intersectionPointDown = cv::Point(intersectionPoint.x, (intersectionPoint.y + 10));
                std::vector<cv::Point> points = { mainTrackSwitch.leftTrackBranch, intersectionPointDown, mainTrackSwitch.rightTrackBranch, mainTrackSwitch.rightTrackIntersectionMirror, intersectionPoint, mainTrackSwitch.leftTrackIntersectionMirror };
                for (int i = 0; i < points.size(); i++) {
                    points[i].x *= OUTPUT_TO_DETECTION_WIDTH_RATIO;
                    points[i].y *= OUTPUT_TO_DETECTION_HEIGHT_RATIO;
                }
                cv::fillPoly(tmpOutputOverlayImage, points, switchDrawColor);
            }

            // Switch trigger
            switchDetection.switchDetected = true;
            switchDetection.lastFrameDetected = frameNumber;
            switchDetection.switchedLeft = false;
            switchDetection.switchedRight = false;
            switchDetection.switchDirectionDetected = false;

            switchDetection.incomingSwitchLeft = false;
            switchDetection.incomingSwitchRight = false;
            switchDetection.incomingSwitchCounterLeft = 0;
            switchDetection.incomingSwitchCounterRight = 0;
        }
        else if (!switchDetection.switchDetected && !switchDetection.switchDirectionDetected && mainTrackSwitch.leftTrackBranch != cv::Point(0, 0)) { // incoming switch from the left
            // Incoming track from the right
            switchDetection.incomingSwitchRight = false;
            switchDetection.incomingSwitchCounterLeft++;
            switchDetection.lastFrameIncomingSwitchDetected = frameNumber;
            if (switchDetection.incomingSwitchCounterLeft > INCOMING_SWITCH_THRESHOLD && trackPairs.leftTrackPairFoundCounter > TRACK_PAIR_FOUND_COUNTER_THRESHOLD / 2) {
                switchDetection.incomingSwitchLeft = true;
            }
        }
        else if (!switchDetection.switchDetected && !switchDetection.switchDirectionDetected && mainTrackSwitch.rightTrackBranch != cv::Point(0, 0)) { // incoming switch from the right
            // Incoming track from the left
            switchDetection.incomingSwitchLeft = false;
            switchDetection.incomingSwitchCounterRight++;
            switchDetection.lastFrameIncomingSwitchDetected = frameNumber;
            if (switchDetection.incomingSwitchCounterRight > INCOMING_SWITCH_THRESHOLD && trackPairs.rightTrackPairFoundCounter > TRACK_PAIR_FOUND_COUNTER_THRESHOLD / 2) {
                switchDetection.incomingSwitchRight = true;
            }
        }
        return;
    }

    /*
    * Finds the switch direction and handles the switch detection ending.
    */
    void CheckIfSwitchFound() {
        if (switchDetection.switchDetected) {
            if (frameNumber - switchDetection.lastFrameDetected < SWITCH_DETECTION_MAX_EVALUATION_TIME) {
                // Set the switch direction (left).
                if (switchDetection.sideBranchingCounters[SIDE::LEFT] > SWITCH_DIRECTION_THRESHOLD && !switchDetection.switchDirectionDetected) {
                    switchDetection.switchedRight = true;
                    switchDetection.switchedLeft = false;
                }
                // Set the switch direction (right).
                if (switchDetection.sideBranchingCounters[SIDE::RIGHT] > SWITCH_DIRECTION_THRESHOLD && !switchDetection.switchDirectionDetected) {
                    switchDetection.switchedLeft = true;
                    switchDetection.switchedRight = false;
                }
                // Switch direction found, switch detection is now over.
                if (switchDetection.switchedLeft || switchDetection.switchedRight) {
                    switchDetection.switchDetected = false;
                    switchDetection.switchDirectionDetected = true;
                    switchDetection.lastFrameDetected = frameNumber; // reset to be used as a timer for how long the switch info is displayed
                }
            }
            else { // No switch direction found within the time frame
                switchDetection.switchDetected = false;
                // cout << "[FAILURE] No switch direction found" << endl;
            }
        }
        else if (frameNumber - switchDetection.lastFrameDetected == INCOMING_SWITCH_DETECTION_TIME) {
            switchDetection.switchDirectionDetected = false;
            switchDetection.sideBranchingCounters[SIDE::LEFT] = 0;
            switchDetection.sideBranchingCounters[SIDE::RIGHT] = 0;
        }

        if (frameNumber - switchDetection.lastFrameIncomingSwitchDetected == INCOMING_SWITCH_DETECTION_TIME) {
            // Reset if no incoming switch was robustly detected
            switchDetection.incomingSwitchLeft = false;
            switchDetection.incomingSwitchRight = false;
            switchDetection.incomingSwitchCounterLeft = 0;
            switchDetection.incomingSwitchCounterRight = 0;
        }
        return;
    }

    void ResetSwitch() {
        mainTrackSwitch.leftTrackBranch.x = 0;
        mainTrackSwitch.leftTrackBranch.y = 0;
        mainTrackSwitch.leftTrackIntersection.x = 0;
        mainTrackSwitch.leftTrackIntersection.y = 0;
        mainTrackSwitch.leftTrackIntersectionMirror.x = 0;
        mainTrackSwitch.leftTrackIntersectionMirror.y = 0;
        mainTrackSwitch.rightTrackBranch.x = 0;
        mainTrackSwitch.rightTrackBranch.y = 0;
        mainTrackSwitch.rightTrackIntersection.x = 0;
        mainTrackSwitch.rightTrackIntersection.y = 0;
        mainTrackSwitch.rightTrackIntersectionMirror.x = 0;
        mainTrackSwitch.rightTrackIntersectionMirror.y = 0;
    }

#pragma endregion Switch Detection


    // ######## ######## ######## ######## Warning Zone ######## ######## ######## ######## 
#pragma region Warning Zone

    /*
    * Returns the warning zone index of an object from its lower left and right corner points. 
    * Warning zone index is of type WARNING_ZONE. 
    * Outside the warning zone will be 0, anything > 0 is inside the warning zone. 
    * The warning zone will later be drawn if an object is found within the warning zone.
    */
    int FindIfObjectIsInsideWarningZone(cv::Point leftPoint, cv::Point rightPoint, int outputImageWidth = DETECTION_IMAGE_WIDTH, int outputImageHeight = DETECTION_IMAGE_HEIGHT) {
        // Scale points to the track detection sizes
        float xScale = DETECTION_IMAGE_WIDTH / (float)outputImageWidth;
        float yScale = DETECTION_IMAGE_HEIGHT / (float)outputImageHeight;
        int row = leftPoint.y * yScale;
        
        int warningZoneRowLimit1 = DETECTION_IMAGE_HEIGHT * 0.5;
        int warningZoneRowLimit2 = DETECTION_IMAGE_HEIGHT * 0.4;

        int warningZoneIndex;

        int trackWidth = GetTrackWidth(row);
        int middle = GetTrackMiddle(row);

        int OUTER_WARNING_ZONE_MARGIN = 3 * WARNING_ZOONE_MARGIN;

        int warningZoneLeftCol = std::max((int)(middle - trackWidth * WARNING_ZOONE_MARGIN), 0);
        int warningZoneRightCol = std::min((int)(middle + trackWidth * WARNING_ZOONE_MARGIN), DETECTION_IMAGE_WIDTH);
        int outerWarningZoneLeftCol = std::max((int)(middle - trackWidth * OUTER_WARNING_ZONE_MARGIN), 0);
        int outerWarningZoneRightCol = std::min((int)(middle + trackWidth * OUTER_WARNING_ZONE_MARGIN), DETECTION_IMAGE_WIDTH);

        // Set the warning zone index
        if (rightPoint.x * xScale < outerWarningZoneLeftCol || leftPoint.x * xScale > outerWarningZoneRightCol){
            warningZoneIndex = WARNING_ZONE::OUTSIDE_ZONE;
        }
        else {
            int boxMiddle = (rightPoint.x + leftPoint.x) * xScale / 2.0;
            if (boxMiddle < warningZoneLeftCol){//Left
                warningZoneIndex = WARNING_ZONE::INSIDE_WARNING_ZONE + WARNING_ZONE::LEFT;
            }
            else if (boxMiddle > warningZoneRightCol){//Right
                warningZoneIndex = WARNING_ZONE::INSIDE_WARNING_ZONE + WARNING_ZONE::RIGHT;
            }
            else{//Middle
                warningZoneIndex = WARNING_ZONE::INSIDE_TRACK_ZONE;
            }
            
            if (row < warningZoneRowLimit2) {//Far
                warningZoneIndex += WARNING_ZONE::FAR;
            }
            else if (row < warningZoneRowLimit1) {//Mid
                warningZoneIndex += WARNING_ZONE::MID;
            }
            else {//Near
                warningZoneIndex += WARNING_ZONE::NEAR;
            }
        }

        if (warningZoneIndex != WARNING_ZONE::OUTSIDE_ZONE)
            drawWarningZone = true;
        return warningZoneIndex;
    }

    /*
    * Draw the warning zone.
    * Red is the track zone, left is the side zones.
    * @param contours Draw only the contours of the warning zone.
    */
    void DrawWarningZone(bool contours = false){
        int warningZoneRowLimit1 = DETECTION_IMAGE_HEIGHT * 0.5;
        int warningZoneRowLimit2 = DETECTION_IMAGE_HEIGHT * 0.4;

        uchar* warningZone = (uchar*)warningZoneMat.data;
        uchar* classifiedPixels = (uchar*)classifiedPixelsMat.data;

        // Reset the images to draw on.
        memset(warningZone, 0, DETECTION_IMAGE_AREA * sizeof(*warningZone));
        memset(detectionOverlayImage.data, 0, DETECTION_IMAGE_AREA * 3 * sizeof(*detectionOverlayImage.data));

        for (int row = trackHighestRow; row < DETECTION_IMAGE_HEIGHT; row++){
            int verticalZoneMultiplier = 0;
            if (row < warningZoneRowLimit2){
                verticalZoneMultiplier = 2;
            }
            else if (row < warningZoneRowLimit1)
                verticalZoneMultiplier = 1;

            int trackWidth = GetTrackWidth(row);
            int middle = GetTrackMiddle(row);

            float OUTER_WARNING_ZONE_MARGIN = 3 * WARNING_ZOONE_MARGIN;

            if (contours) {
                // Draw the contours of the warning zone.
                int left = std::max((int)(middle - trackWidth * OUTER_WARNING_ZONE_MARGIN), 0) + row * DETECTION_IMAGE_WIDTH;
                int right = std::min((int)(middle + trackWidth * OUTER_WARNING_ZONE_MARGIN), DETECTION_IMAGE_WIDTH) + row * DETECTION_IMAGE_WIDTH;
                int innerLeft = std::max((int)(middle - trackWidth * WARNING_ZOONE_MARGIN), 0) + row * DETECTION_IMAGE_WIDTH;
                int innerRight = std::min((int)(middle + trackWidth * WARNING_ZOONE_MARGIN), DETECTION_IMAGE_WIDTH) + row * DETECTION_IMAGE_WIDTH;

                int thickness = 4;
                for (int i = left; i < left + thickness; i++) {
                    warningZone[i] = 100;
                }
                for (int i = right - thickness; i < right; i++) {
                    warningZone[i] = 100;
                }
                for (int i = innerLeft; i < innerLeft + thickness; i++) {
                    warningZone[i] = 100;
                }
                for (int i = innerRight - thickness; i < innerRight; i++) {
                    warningZone[i] = 100;
                }
            }
            else {
                // Draw an filled warning zone.
                for (int col = std::max((int)(middle - trackWidth * OUTER_WARNING_ZONE_MARGIN), 0); col < std::min((int)(middle + trackWidth * OUTER_WARNING_ZONE_MARGIN), DETECTION_IMAGE_WIDTH); col++) {
                    // Sides
                    warningZone[col + row * DETECTION_IMAGE_WIDTH] = 2;
                }
                for (int col = std::max((int)(middle - trackWidth * WARNING_ZOONE_MARGIN), 0); col < std::min((int)(middle + trackWidth * WARNING_ZOONE_MARGIN), DETECTION_IMAGE_WIDTH); col++) {
                    // Middle
                    warningZone[col + row * DETECTION_IMAGE_WIDTH] = 1;
                }
            }
        }
        
        // Convert to color and resize to output size
        detectionOverlayImage.setTo(cv::Scalar(0, 75, 255), warningZoneMat == 1); // Red for the middle
        detectionOverlayImage.setTo(cv::Scalar(0, 150, 255), warningZoneMat == 2); // Yellow for the sides
        ResizeToOutput(detectionOverlayImage, tmpOutputOverlayImage);
        return;
    }

#pragma endregion Warning Zone


    // ######## ######## ######## ######## Track Detection Algorithm ######## ######## ######## ######## 
#pragma region Track Detection Algorithm

    /*
    * The main function that detects the tracks from an image. 
    * Preprocesses the image by scaling, gray scale, blurring and calculates the derivatives and their magnitude.
    * Finds possible rails with a region growing algorithm.
    * Pairs rails to tracks.
    * Find switches in the main track. 
    * Draw the tracks and switches. 
    */
    void DetectTracks(cv::Mat frameImage, bool draw) {
        drawWarningZone = false;
        trackHighestRow = DETECTION_IMAGE_HEIGHT - horizonRow;

        // Resize image
        ScaleImage(frameImage); 

        // Gray scale image
        cv::cvtColor(resizedImage, grayScaleImage, cv::COLOR_BGR2GRAY);

        // Blur bottom half of image (Gaussian)
        cv::GaussianBlur(grayScaleImage(blurMask), grayScaleImage(blurMask), cv::Size(3, 3), 0);

        // Create the vertical (uniform) blur kernel
        const int BLUR_SIZE = 5;
        cv::Mat kernel(cv::Size(1, BLUR_SIZE), CV_32FC1);
        for (int i = 0; i < BLUR_SIZE; i++){
            kernel.at<float>(i) = 1 / (float)BLUR_SIZE;
        }
        
        // Apply vertical blur on the region of the main track
        if (verticalMaskExists){
            cv::Mat imageToBlur;
            cv::filter2D(grayScaleImage, imageToBlur, -1, kernel);
            cv::bitwise_and(imageToBlur, verticalBlurMaskMat, imageToBlur);
            cv::bitwise_not(verticalBlurMaskMat, verticalBlurMaskMat);
            cv::bitwise_and(grayScaleImage, verticalBlurMaskMat, grayScaleImage);
            grayScaleImage = grayScaleImage + imageToBlur;
        }

        // Calculate vertical and horizontal derivatives and their total magnitude
        cv::Sobel(grayScaleImage(sobelMask), xDerivative(sobelMask), 5, 1, 0, SOBEL_KERNEL_SIZE);
        cv::Sobel(grayScaleImage(sobelMask), yDerivative(sobelMask), 5, 0, 1, SOBEL_KERNEL_SIZE);
        cv::magnitude(xDerivative, yDerivative, magnitudeMat);
        float* magnitude = (float*)magnitudeMat.data;

        // Perform region growing to find tracks    
        uchar* classifiedPixels = (uchar*)classifiedPixelsMat.data;
        int tmp = RegionGrowingAlgorithm(magnitude, classifiedPixels);

        // Find pairs of rails that form tracks
        PairRailsToTracks(classifiedPixels);

        // Decide on the current lane status
        UpdateLaneStatus();

        // Calculate all the track middles once per frame
        CalculateAllTrackMiddles();

        //Create the vertical blur mask applying to the next frame
        CreateVerticalBlurMask();

        // Reset the main track switch elements
        ResetSwitch();

        // Find branches in main track that can be a switch
        int leftTrackViewDistance;
        int rightTrackViewDistance;
        std::thread processLeftRailThread(FindBranchesFromTrack, trackPairs.leftMainTrackIndex, &leftTrackViewDistance);
        std::thread processRightRailThread(FindBranchesFromTrack, trackPairs.rightMainTrackIndex, &rightTrackViewDistance);
        processLeftRailThread.join();
        processRightRailThread.join();
        maxTrackViewDistance = std::max(leftTrackViewDistance, rightTrackViewDistance);
        minTrackViewDistance = std::min(leftTrackViewDistance, rightTrackViewDistance);

        if (draw){
            // Reset images to be drawn on
            memset(detectionOverlayImage.data, 0, DETECTION_IMAGE_AREA * 3 * sizeof(*detectionOverlayImage.data));
            memset(outputOverlayImage.data, 0, OUTPUT_IMAGE_AREA * 3 * sizeof(*outputOverlayImage.data));
            memset(tmpOutputOverlayImage.data, 0, OUTPUT_IMAGE_AREA * 3 * sizeof(*tmpOutputOverlayImage.data));

            // Draw side tracks
            //detectionOverlayImage.setTo(cv::Scalar(255, 0, 0), classifiedPixelsMat != 0); // Draw unpaired rails in blue
            for (int iLeftRight = 0; iLeftRight < leftAndRightTrackIndex.size(); iLeftRight++) { // Draw left and right tracks (not main track)
                detectionOverlayImage.setTo(sideTrackColor, classifiedPixelsMat == leftAndRightTrackIndex[iLeftRight]);
            }
            
            // Draw main track, left and right
            detectionOverlayImage.setTo(mainTrackColor, classifiedPixelsMat == trackPairs.leftMainTrackIndex);
            detectionOverlayImage.setTo(mainTrackColor, classifiedPixelsMat == trackPairs.rightMainTrackIndex);
        }

        // Find switch
        ProcessAndDrawSwitch(draw);
        if (draw) {
            ResizeToOutput(detectionOverlayImage, outputOverlayImage);
            cv::addWeighted(outputOverlayImage, 1, tmpOutputOverlayImage, 0.5, 0, outputOverlayImage); // Draw switch polygon
        }

        // Check if possible potential switch is actual switch
        CheckIfSwitchFound();
    }

#pragma endregion Track Detection Algorithm


    // ######## ######## ######## ######## Output Global Information ######## ######## ######## ######## 
#pragma region Output Global Information

    /*
    * Outputs information about the track, lane status and switches to a global struct. 
    * Used only by external files.
    */
    void OutputGlobalInformation() {
        // Track view distance
        globalInformation.maxTrackViewDistance = maxTrackViewDistance;
        globalInformation.minTrackViewDistance = minTrackViewDistance;

        if (!switchDetection.switchDetected && !switchDetection.switchDirectionDetected)
            switchDetection.beforeSwitchLaneStatus = globalInformation.laneStatus;
        globalInformation.laneStatus = laneStatus;

        // Message of direction change at a switch
        globalInformation.incomingSwitchStatus = SWITCH::NONE;
        if (switchDetection.switchDetected || switchDetection.switchDirectionDetected) {
            if (switchDetection.switchDirectionDetected) {
                if (switchDetection.switchedLeft) {
                    globalInformation.switchedLaneStatus = SWITCH::SWITCHED_LEFT;
                }

                if (switchDetection.switchedRight) {
                    globalInformation.switchedLaneStatus = SWITCH::SWITCHED_RIGHT;
                }
            }
            else {
                if (switchDetection.incomingSwitchLeft) {
                    globalInformation.incomingSwitchStatus = SWITCH::INCOMING_FROM_LEFT;
                }
                else if (switchDetection.incomingSwitchRight) {
                    globalInformation.incomingSwitchStatus = SWITCH::INCOMING_FROM_RIGHT;
                }
            }
        }
        else if (switchDetection.incomingSwitchLeft) {
            globalInformation.incomingSwitchStatus = SWITCH::INCOMING_FROM_LEFT;
        }
        else if (switchDetection.incomingSwitchRight) {
            globalInformation.incomingSwitchStatus = SWITCH::INCOMING_FROM_RIGHT;
        }
        globalInformation.switchDetected = switchDetection.switchDetected;
        return;
    }

#pragma endregion Output Global Information


    // ######## ######## ######## ######## Drawing Functions ######## ######## ######## ######## 
#pragma region Drawing Functions

    /*
    * Draw the warning zone, tracks and switches on an image.
    * The warning zone will automatically be drawn if an object is inside. 
    */
    void DrawOverlayImages(cv::Mat outputImage, bool alwaysDrawWarningZone = false, bool contours = false) {
        if (alwaysDrawWarningZone || drawWarningZone) {
            DrawWarningZone();
            cv::addWeighted(outputImage, 1, tmpOutputOverlayImage, 0.25, 0, outputImage); // Warning zone
        }
        else if (contours) {
            DrawWarningZone(contours);
            cv::addWeighted(outputImage, 1, tmpOutputOverlayImage, 1, 0, outputImage);
        }

        cv::addWeighted(outputImage, 1, outputOverlayImage, 1, 0, outputImage); // Draw tracks and switches
    }

    /*
    * Helper function to aid the calibration of the track width and horizon row.
    * Plots the track width with two lines.
    * Plots the horizon row with a point.
    */
    void CalibrationAid(cv::Mat image) {
        uchar* classifiedPixels = (uchar*)classifiedPixelsMat.data;
        int tmp;
        int leftCol;
        int rightCol;
        FindFirstAndLastNonzero(DETECTION_IMAGE_HEIGHT - 1, classifiedPixels, &leftCol, &tmp, trackPairs.leftMainTrackIndex);
        FindFirstAndLastNonzero(DETECTION_IMAGE_HEIGHT - 1, classifiedPixels, &tmp, &rightCol, trackPairs.rightMainTrackIndex);
        int trackWidth = GetTrackWidth(DETECTION_IMAGE_HEIGHT - 1);
        int middle;

        if (leftCol != DETECTION_IMAGE_WIDTH && rightCol != 0) {
            middle = (leftCol + rightCol) / 2.0;
        }
        else {
            middle = DETECTION_IMAGE_WIDTH / 2;
        }

        int leftX = middle - trackWidth / 2;
        int rightX = middle + trackWidth / 2;
        int height = 40 * OUTPUT_TO_DETECTION_WIDTH_RATIO;
        cv::Scalar color = cv::Scalar(0, 215, 255);
        cv::line(image, cv::Point(leftX * OUTPUT_TO_DETECTION_WIDTH_RATIO, DETECTION_IMAGE_HEIGHT * OUTPUT_TO_DETECTION_HEIGHT_RATIO), cv::Point(leftX * OUTPUT_TO_DETECTION_WIDTH_RATIO, (DETECTION_IMAGE_HEIGHT - height) * OUTPUT_TO_DETECTION_HEIGHT_RATIO), color, 2);
        cv::line(image, cv::Point(rightX * OUTPUT_TO_DETECTION_WIDTH_RATIO, DETECTION_IMAGE_HEIGHT * OUTPUT_TO_DETECTION_HEIGHT_RATIO), cv::Point(rightX * OUTPUT_TO_DETECTION_WIDTH_RATIO, (DETECTION_IMAGE_HEIGHT - height) * OUTPUT_TO_DETECTION_HEIGHT_RATIO), color, 2);

        cv::circle(image, cv::Point(middle * OUTPUT_TO_DETECTION_WIDTH_RATIO, (DETECTION_IMAGE_HEIGHT - horizonRow) * OUTPUT_TO_DETECTION_HEIGHT_RATIO), 3, color, -1);
    }

#pragma endregion Drawing Functions


    // ######## ######## ######## ######## Initialize Functions ######## ######## ######## ######## 
#pragma region Initialize Functions

    void InizializeArrays() {
        // Set and Scale Parameters
        horizonRow = horizonHeightFraction * DETECTION_IMAGE_HEIGHT;
        distanceBetweenTrackPairs = trackWidthFraction * DETECTION_IMAGE_WIDTH;
        bottomPixelsToCrop = cropBottomFraction * OUTPUT_IMAGE_HEIGHT;

        minDistanceBetweenObjects *= distanceBetweenTrackPairs / 310.0;
        distanceBetweenMainTracksMargin *= distanceBetweenTrackPairs / 310.0;
        distanceBetweenSideTracksMargin *= distanceBetweenTrackPairs / 310.0;
        estimatedRailWidth *= distanceBetweenTrackPairs / 310.0;

        distanceBetweenTracksNarrowingConstant = distanceBetweenTrackPairs / (float)horizonRow;
        minHeightOfRailInGrowList = DETECTION_IMAGE_HEIGHT - 0.3 * horizonRow;
        scanImageForBranchHeight = (DETECTION_IMAGE_HEIGHT - horizonRow) * 5.0 / 4.0;
        railWidthNarrowingConstant = estimatedRailWidth * 0.5 / horizonRow;

        adaptiveThresholdStartWidth = (1 - distanceBetweenTrackPairs / (float)DETECTION_IMAGE_WIDTH * 1.6) / 2;
        

        // Allocate memory for different cv::Mat's
        imageSize = cv::Size(DETECTION_IMAGE_WIDTH, DETECTION_IMAGE_HEIGHT);
        outputImageSize = cv::Size(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT);
        resizedImage = cv::Mat(imageSize, CV_8UC3);
        grayScaleImage = cv::Mat(imageSize, CV_8U);
        xDerivative = cv::Mat(imageSize, CV_32F);
        yDerivative = cv::Mat(imageSize, CV_32F);
        magnitudeMat = cv::Mat(grayScaleImage);
        classifiedPixelsMat = cv::Mat(imageSize, CV_8U);
        verticalBlurMaskMat = cv::Mat(imageSize, CV_8UC1);
        warningZoneMat = cv::Mat(imageSize, CV_8U);

        if (trackDetectionStandAlone) {
            outputImage = cv::Mat(outputImageSize, CV_8UC3);
            infoBoxOverlayImage = cv::Mat(imageSize, CV_8UC3);
        }

        detectionOverlayImage = cv::Mat(imageSize, CV_8UC3);
        outputOverlayImage = cv::Mat(outputImageSize, CV_8UC3);
        tmpOutputOverlayImage = cv::Mat(outputImageSize, CV_8UC3);

        blurMask = cv::Rect(0, int((DETECTION_IMAGE_HEIGHT - horizonRow) * 1.2), DETECTION_IMAGE_WIDTH, int(DETECTION_IMAGE_HEIGHT - (DETECTION_IMAGE_HEIGHT - horizonRow) * 1.2));
        sobelMask = cv::Rect(0, DETECTION_IMAGE_HEIGHT - horizonRow - 10, DETECTION_IMAGE_WIDTH, horizonRow + 10);

        // Initialize the main track switch elements
        mainTrackSwitch.leftTrackBranch = cv::Point(0, 0);
        mainTrackSwitch.leftTrackIntersection = cv::Point(0, 0);
        mainTrackSwitch.leftTrackIntersectionMirror = cv::Point(0, 0);
        mainTrackSwitch.rightTrackBranch = cv::Point(0, 0);
        mainTrackSwitch.rightTrackBranch = cv::Point(0, 0);
        mainTrackSwitch.rightTrackIntersectionMirror = cv::Point(0, 0);

        // Reserve space in the grow lists
        growList.reserve(GROW_LIST_ALLOCATED_ARRAY_SIZE);

        // Initialize the start seed points for region-growing algorithm
        for (int seedIndex = 0; seedIndex < DETECTION_IMAGE_WIDTH; seedIndex += SEED_SKIP_NUMBER) { // Bottom row seeds
            if (std::abs(std::abs(seedIndex - DETECTION_IMAGE_WIDTH / 2.0) - 155) < 60)
                startPoints.insert(startPoints.begin(), cv::Point(DETECTION_IMAGE_HEIGHT - 1, seedIndex));
            else
                startPoints.push_back(cv::Point(DETECTION_IMAGE_HEIGHT - 1, seedIndex));
        }
        for (int seedIndex = DETECTION_IMAGE_HEIGHT - SEED_SKIP_NUMBER - 1; seedIndex > int(DETECTION_IMAGE_HEIGHT / 3.0); seedIndex -= SEED_SKIP_NUMBER) { // Side column seeds
            startPoints.push_back(cv::Point(seedIndex, DETECTION_IMAGE_WIDTH - 1));
            startPoints.push_back(cv::Point(seedIndex, 0));
        }
        trackStartPoints.reserve(10);
        leftAndRightTrackIndex.reserve(8);
        
        for (int i = 0; i < DETECTION_IMAGE_HEIGHT; i++) {
            trackMiddles.push_back(0);
        }

        //Calculate the track widths for each image row, once and for all
        CalculateAllTrackWidths();
    }

#pragma endregion Initialize Functions


    // ######## ######## ######## ######## Stand Alone Track Detection Program ######## ######## ######## ######## 
#pragma region Stand Alone Track Detection Program

    void LoopTrackDetection(cv::Mat frameImage) {
        if (frameImage.size() != imageSize) {
            cv::resize(frameImage, frameImage, imageSize);
        }

        // Possibly crop image
        if (bottomPixelsToCrop > 0) {
            cv::Rect mask(0, 0, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT - bottomPixelsToCrop);
            frameImage = frameImage(mask);
        }

        DetectTracks(frameImage, true);
        UpdateLaneStatus();

        // Draw on the output image

        // Draw the magnitude
        /*magnitudeMat /= 1.8;
        cv::cvtColor(magnitudeMat, magnitudeMat, cv::COLOR_GRAY2BGR);
        magnitudeMat.convertTo(magnitudeMat, CV_8UC3);
        cv::resize(magnitudeMat, outputImage, outputImageSize);*/

        // Draw color image
        resizedImage.copyTo(outputImage);

        // Draw color image with lower saturation
        /*cv::Mat grayColor;
        cv::cvtColor(grayScaleImage, grayColor, cv::COLOR_GRAY2BGR);
        cv::addWeighted(resizedImage, 0.7, grayColor, 0.2, 0, outputImage);*/

        DrawOverlayImages(outputImage, ALWAYS_DRAW_WARNING_ZONE, false);

        // Save frame
        if (saveVideo)
            video.write(outputImage);

        cv::imshow(windowName, outputImage);
        cv::waitKey(1);
    }

    /*
    * Runs the track detection independently of the object detection. 
    * Runs from RunTrackDetection.cpp
    */
    void RunTrackDetection()
    {
        trackDetectionStandAlone = true;

        // Import video
        string videoPath = (std::string)videoFolderPath + (std::string)videoName;
        cv::VideoCapture cap(videoPath);

        windowName = "Video Window";
        namedWindow(windowName);

        //Allocate memory for various objects
        InizializeArrays();

        // Set up video saver
        if (saveVideo) {
            string savePathName = (std::string)saveFolderPath + (std::string)saveName;
            video = cv::VideoWriter(savePathName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsSavedVideo, imageSize);
        }

        // Start at set frame number
        cap.set(cv::CAP_PROP_POS_FRAMES, frameNumber);

        cv::Mat frameImage;
        while (true) {
            cap >> frameImage;
            if (frameImage.empty())
                break;

            LoopTrackDetection(frameImage);
            frameNumber++;
        }
        cap.release();
        video.release();
    }

#pragma endregion Stand Alone Track Detection Program


}