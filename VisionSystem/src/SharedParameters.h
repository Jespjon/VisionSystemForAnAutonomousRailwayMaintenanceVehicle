#pragma once
#include <string>

/*
* Most parameters are set in Parameters.txt
*/

// General Parameters 
inline const int OUTPUT_IMAGE_WIDTH = 1280;
inline const int OUTPUT_IMAGE_HEIGHT = 720;

inline bool useCameraInput = false; // Use camera input. False means use video input.

inline bool drawVisionOutput;

inline float horizonHeightFraction; // Proportion of the image under horizon
inline float trackWidthFraction; // Proportion of the width of the track at the bottom

inline int frameNumber;

inline float cropBottomFraction; // fraction of image to crop at the bottom

inline std::string videoName;

// Paths
#ifdef _WIN32
inline const std::string pathToVisionSystem = "../../../";
#endif
#ifdef linux
inline const std::string pathToVisionSystem = "/home/jetson/VisionAndNavigationForAnAutonomousTrackTrolley/VisionSystem/";
#endif

// Load video
inline const std::string videoFolderPath = pathToVisionSystem + "../Videos/";			// Path to video data
//#ifdef _WIN32
//inline const std::string videoFolderPath = pathToVisionSystem + "../Videos/";             // Path to video data
//#endif
//#ifdef linux
//inline const std::string videoFolderPath = pathToVisionSystem + "../Videos/";             // Path to video data
//#endif

// Save video
inline bool saveVideo;																	// 'true' if video should be recorded
inline int fpsSavedVideo = 15;                                                         // FPS used for recorded/saved video
inline std::string saveName;
inline const std::string saveFolderPath = pathToVisionSystem + "../Saved Videos/";              // Path to save folder
//#ifdef _WIN32
//inline const std::string saveFolderPath = pathToVisionSystem + "../Saved Videos/";              // Path to save folder
//#endif
//#ifdef linux
//inline const std::string saveFolderPath = pathToVisionSystem + "../Saved Videos/";      // Path to save folder
//#endif
