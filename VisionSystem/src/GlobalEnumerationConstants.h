/*
* Global enumeration constants. 
* Used by variables in GlobalInformation.h and in some cpp files.
*/

#pragma once
#include <string>
#include <vector>


namespace OBJECTS {
    enum OBJECTS {
        CAR,    //0
        PERSON,
        SIGN_SPEED,
        SIGN_V,
        SIGN_TRIANGLE_WARNING,
        SIGN_SPEED_DOWN,        //5
        SIGN_SPEED_UP,
        DISTANT_SIGNAL,
        ROAD_CROSSING_SIGNAL,
        MAIN_SIGNAL,
        TRAIN,          //10
        DISTANT_ROAD_CROSSING_SIGNAL,
        BUS,
        BARRIER,
        TRUCK,
        MOTORCYCLE,     //15
        CROSSING,
        BICYCLE,
        POLE,
    };
    const std::vector<std::string> TEXT({ "Car", "Person", "Sign speed", "Sign V", "Sign triangle warning", "Sign ATC speed down", "Sign ATC speed up",
        "Distant signal", "Road crossing signal", "Main signal", "Train", "Distant road crossing signal", "Bus", "Barrier", "Truck", "Motorcycle", "Road crossing", 
        "Bicycle", "Pole"});
}

namespace SIDE {
    enum SIDE {
        LEFT,
        RIGHT,
    };
}

namespace MAIN_SIGNAL_MESSAGE {
    enum MAIN_SIGNAL_MESSAGE {
        NONE,
        GO_80,
        GO_40,
        GO_40_GENTLY,
        STOP,
        SHORT_TO_STOP,
    };
    const std::vector<std::string> TEXT({"", "Go 80 km/h", "Go 40 km/h", "Go 40 km/h, gently", "Stop", "Short distance to next stop"});
}

namespace ROAD_CROSSING_MESSAGE {
    enum ROAD_CROSSING_MESSAGE {
        NONE,
        UPCOMING,
        DETECTED,
        GO,
        STOP,
    };
    const std::vector<std::string> TEXT({ "", "Upcoming", "Detected", "Go", "Stop"});
}

namespace LANE_STATUS {
    enum LANE_STATUS {
        SINGLE_TRACK,
        LEFT_TRACK,
        RIGHT_TRACK,
        MIDDLE_TRACK,
    };
    const std::vector<std::string> TEXT({ "Middle track", "Right track", "Left track", "Single track" });
}

namespace SWITCH {
    enum SWITCH {
        NONE,
        CONTINUED_ON_TRACK,
        SWITCHED_RIGHT,
        SWITCHED_LEFT,
        INCOMING_FROM_LEFT,
        INCOMING_FROM_RIGHT,
    };
    const std::vector<std::string> TEXT({ "None", "Continued on track", "Switched right", "Switched left", "Incoming from left", "Incoming from right" });
}

namespace WARNING_ZONE {
    enum WARNING_ZONE {
        OUTSIDE_ZONE = 0,
        INSIDE_TRACK_ZONE = 1,
        INSIDE_WARNING_ZONE = 100,
        LEFT = 10,
        RIGHT = 20,
        NEAR = 2,
        MID = 3,
        FAR = 4,

        INSIDE_TRACK_ZONE_NEAR = INSIDE_TRACK_ZONE + NEAR,
        INSIDE_TRACK_ZONE_MID = INSIDE_TRACK_ZONE + MID,
        INSIDE_TRACK_ZONE_FAR = INSIDE_TRACK_ZONE + FAR,
        INSIDE_WARNING_ZONE_NEAR_LEFT = INSIDE_WARNING_ZONE + NEAR + LEFT,
        INSIDE_WARNING_ZONE_NEAR_RIGHT = INSIDE_WARNING_ZONE + NEAR + RIGHT,
        INSIDE_WARNING_ZONE_MID_LEFT = INSIDE_WARNING_ZONE + MID + LEFT,
        INSIDE_WARNING_ZONE_MID_RIGHT = INSIDE_WARNING_ZONE + MID + RIGHT,
        INSIDE_WARNING_ZONE_FAR_LEFT = INSIDE_WARNING_ZONE + FAR + LEFT,
        INSIDE_WARNING_ZONE_FAR_RIGHT = INSIDE_WARNING_ZONE + FAR + RIGHT,
    };
}

namespace DRIVING_MODE {
    enum DRIVING_MODE {
        FORWARD,
        BACKWARD,
    };
}

namespace SET_PARAMETER {
    enum SET_PARAMETER {
        DRAW_VISION_OUTPUT_ON,
        DRAW_VISION_OUTPUT_OFF,
        FORWARD_DRIVING_MODE,
        BACKWARD_DRIVING_MODE,
        EXIT,
    };
}
