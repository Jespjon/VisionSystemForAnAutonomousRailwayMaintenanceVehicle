#include "LoadParameters.h"
#include "TrackDetection.h"
#include "ObjectDetection.h"

//Dummy function. Used only for compiling when ObjectDetection.cpp is not linked. 
object_detection::Train object_detection::GetDetectedTrain() {
    object_detection::Train tmp;
    return tmp;
}

int main(int argc, char** argv)
{
    LoadParameters();
    track_detection::RunTrackDetection();
    std::cin.get();
    return 0;
}