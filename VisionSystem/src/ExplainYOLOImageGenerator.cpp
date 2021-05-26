// About
/*
* Generate images explaining how YOLO detects.
* Uses Darknet.
* 
* Set the number of horizontal and vertical lines in Grid() manually. 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#define OPENCV
#include "yolo_v2_class.hpp"    // imported functions from DLL

#include "SharedParameters.h"

using std::cout;
using std::endl;
using std::string;

std::string  namesFile = "../../../../../darknet/data/yolo-tiny-authours-VOC-COCO-poles/obj.names";
std::string  cfgFile = "../../../../../darknet/cfg/yolo-tiny-obj-authours-VOC-COCO-poles.cfg";
std::string  weightsFile = "../../../../../darknet/backup/yolo-tiny-obj-authours-VOC-COCO-poles_best.weights";

std::vector<std::string> ObjectNames;
Detector* detector;
float detectedObjectThreshold;

int imageWidth = 1280;
int imageHeight = 720;
cv::Size imageSize(imageWidth, imageHeight);
cv::Mat frameImage;
cv::Mat outputImage;
std::vector<bbox_t> resultVector;
std::string windowName = "Image Window";

// Set the names of the input and output image
std::string imageFolderPath = pathToVisionSystem + "../Images/";
std::string inputImageName = "TestImage.png";
std::string outputFileName = "yolo";

void DrawBoxes(cv::Mat image) {
    //Purple, Red, Yellow, Green, Cyan, Blue
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    //Car, person, sign_speed, sign_v, sign_triangle_warning, sign_speed_down, sign_speed_up, signal_forsignal, signal_1, signal_2-5, train, signal_triangle, bus, barrier, truck, motorcycle, crossing, bicycle
    int classColorIndices[18] = { 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 1, 5, 1, 1, 5, 1 };
    int detectedSpeedSignIndex = 0;
    int detectedSignalIndex = 0;

    for (auto& i : resultVector) {
        cv::Scalar color(colors[classColorIndices[i.obj_id]][0], colors[classColorIndices[i.obj_id]][1], colors[classColorIndices[i.obj_id]][2]);
        color *= 255;
        if (detectedObjectThreshold < 0.1) {
            color = cv::Scalar(0, 0, 0);
            cv::rectangle(image, cv::Rect(i.x, i.y, i.w, i.h), color, i.prob * 20);
        } else
            cv::rectangle(image, cv::Rect(i.x, i.y, i.w, i.h), color, 5);
    }
}

void DetectObjects(cv::Mat frameImage) {
    // detection by Yolo
    std::shared_ptr<image_t> det_image = detector->mat_to_image_resize(frameImage); // resize
    resultVector = detector->detect_resized(*det_image, imageSize.width, imageSize.height, detectedObjectThreshold, true);
}

std::vector<std::string> GetObjectsNamesFromFile(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    return file_lines;
}

void SetupYoloNetwork() {
    detector = new Detector(cfgFile, weightsFile);
    ObjectNames = GetObjectsNamesFromFile(namesFile);
}

void DrawEdge(cv::Mat frameImage) {
    int ramLineWidth = 10;
    cv::line(frameImage, cv::Point(0, 0), cv::Point(0, imageSize.height), cv::Scalar(0, 0, 0), ramLineWidth);
    cv::line(frameImage, cv::Point(imageSize.width - 1, 0), cv::Point(imageSize.width - 1, imageSize.height), cv::Scalar(0, 0, 0), ramLineWidth);
    cv::line(frameImage, cv::Point(0, 0), cv::Point(imageSize.width, 0), cv::Scalar(0, 0, 0), ramLineWidth);
    cv::line(frameImage, cv::Point(0, imageSize.height - 1), cv::Point(imageSize.width, imageSize.height - 1), cv::Scalar(0, 0, 0), ramLineWidth);
}

void YOLONonFiltered() {
    detectedObjectThreshold = 0.000001;
    frameImage.copyTo(outputImage);
    DetectObjects(outputImage);
    DrawBoxes(outputImage);
    DrawEdge(outputImage);
    cv::imshow(windowName, outputImage);
    cv::waitKey(0);
    string fileName = outputFileName + "_bounding_boxes.jpg";
    cv::imwrite(imageFolderPath + fileName, outputImage);
}

void YOLOFiltered() {
    detectedObjectThreshold = 0.2;
    frameImage.copyTo(outputImage);
    DetectObjects(outputImage);
    DrawBoxes(outputImage);
    DrawEdge(outputImage);
    cv::imshow(windowName, outputImage);
    cv::waitKey(0);
    string fileName = outputFileName + "_filtered.jpg";
    cv::imwrite(imageFolderPath + fileName, outputImage);
}

void Grid() {
    frameImage.copyTo(outputImage);
    int gridLineWidth = 2;
    int nVerticalLines = 23;
    for (int i = 0; i < nVerticalLines; i++) {
        cv::line(outputImage, cv::Point(i * imageSize.width / (float)nVerticalLines, 0), cv::Point(i * imageSize.width / (float)nVerticalLines, imageSize.height), cv::Scalar(0, 0, 0), gridLineWidth);
    }
    int nHorizontalLines = 15;
    for (int i = 0; i < nHorizontalLines; i++) {
        cv::line(outputImage, cv::Point(0, i * imageSize.height / (float)nHorizontalLines), cv::Point(imageSize.width, i * imageSize.height / (float)nHorizontalLines), cv::Scalar(0, 0, 0), gridLineWidth);
    }
    DrawEdge(outputImage);
    cv::imshow(windowName, outputImage);
    cv::waitKey(0);
    string fileName = outputFileName + "_grid.jpg";
    cv::imwrite(imageFolderPath + fileName, outputImage);
}

int main(int argc, char** argv)
{
    // Setup
    SetupYoloNetwork();
    cv::namedWindow(windowName);
    frameImage = cv::imread(imageFolderPath + inputImageName);
    imageSize = frameImage.size();

    // Create images
    Grid();
    YOLONonFiltered();
    YOLOFiltered();

    return 0;
}