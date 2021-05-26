// About
/*
* Run the Vision system as a stand alone program.
*/

#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>
#include <thread>
#include <atomic>

#include "VisionSystem.h"

int main(int argc, char** argv)
{
	vision::SetupVision();

	bool runLoop = true;
	while (runLoop) {
		runLoop = vision::VisionDetect();
	}

	vision::VisionDestroy();

	return 0;
}