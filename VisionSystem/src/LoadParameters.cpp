// About
/*
* Loads parameters for the vision system from Parameter.txt file.
*/

#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

#include "SharedParameters.h"

using std::cout;
using std::endl;

void LoadParameters(){
    std::ifstream myfile;
    std::string line;
    std::string parametersFilePath = pathToVisionSystem + "Parameters.txt";

    myfile.open(parametersFilePath);
    std::vector<std::string> loadedParameters;
    if (myfile.is_open())
    {
        while (std::getline(myfile, line))
        {
            if (line[0] == '#') {
                continue;
            }
            std::string parameter = line.substr(0, line.find(" "));
            std::string value = line.substr(line.find(" ") + 1);
            value = value.substr(0, value.find("#")); // remove comments
            value = value.substr(0, value.find("\t")); // remove tabs
            value = value.substr(0, value.find("\r")); // remove return
            value = value.substr(0, value.find(" ")); // remove space

            if (parameter == "")
                continue;
            else if (parameter == "PROPORTION_UNDER_HORIZON")
                horizonHeightFraction = std::stof(value);
            else if (parameter == "FRACTION_OF_TRACK_WIDTH")
                trackWidthFraction = std::stof(value);
            else if (parameter == "USE_CAMERA_INPUT") {
                std::transform(value.begin(), value.end(), value.begin(), ::tolower);
                useCameraInput = (value == "true");
            }
            else if (parameter == "DRAW_VISION_OUTPUT") {
                std::transform(value.begin(), value.end(), value.begin(), ::tolower);
                drawVisionOutput = (value == "true");
            }
            else if (parameter == "SAVE_VIDEO") {
                std::transform(value.begin(), value.end(), value.begin(), ::tolower);
                saveVideo = (value == "true");
            }
            else if (parameter == "SAVED_VIDEO_NAME")
                saveName = value;
            else if (parameter == "START_FRAME_NUMBER")
                frameNumber = std::stoi(value);
            else if (parameter == "VIDEO_NAME")
                videoName = value;
            else if (parameter == "CROP_BOTTOM_FRACTION")
                cropBottomFraction = std::stof(value);
        }
        myfile.close();
        cout << "[OK] Loaded parameters from file" << endl;
    }
    else {
        cout << "[ERROR] Unable to open parameters file!" << endl;
        exit(-1);
    }
}