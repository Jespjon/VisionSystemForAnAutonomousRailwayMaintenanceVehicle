#!/usr/bin/env python3

import rospy
import time
#import busio
import adafruit_gps
import serial
from gps.msg import gps_message

# Connect to GPS receiver and send out data through ROS
def publish_gps():
    uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)
    gps = adafruit_gps.GPS(uart, debug=False)

    gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    gps.send_command(b"PMTK220,1000")

    pub = rospy.Publisher('gps_topic', gps_message, queue_size=10)
    rospy.init_node('gps_node', anonymous=True)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        gps.update()
        message = gps_message()
        #message.timeStamp = int(time.mktime((1999, 11, 20, 5, 27, 35, 0, 1, 0)))
        if gps.has_fix:
            message.connected = True
            if gps.altitude_m is not None:
                message.altitude = gps.altitude_m
            if gps.latitude is not None:
                message.latitude = gps.latitude
            if gps.longitude is not None:
                message.longitude = gps.longitude
            if gps.speed_knots is not None:
                message.speedKnots = gps.speed_knots
            if gps.satellites is not None:
                message.numberOfSatellites = gps.satellites
            if gps.timestamp_utc is not None:
                #tmp = int(time.mktime((gps.timestamp_utc.tm_year,gps.timestamp_utc.tm_mon,gps.timestamp_utc.tm_mday, gps.timestamp_utc.tm_hour,gps.timestamp_utc.tm_min,gps.timestamp_utc.tm_sec, 0, 1, 0)))
                #print(tmp)
                message.timeStamp = int(time.mktime((gps.timestamp_utc.tm_year, gps.timestamp_utc.tm_mon, 
                    gps.timestamp_utc.tm_mday, gps.timestamp_utc.tm_hour, gps.timestamp_utc.tm_min, gps.timestamp_utc.tm_sec, 0, 1, 0)))
        else:
            print("Waiting for connection...")
            if gps.satellites is not None:
                message.numberOfSatellites = gps.satellites
            message.connected = False

        pub.publish(message)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_gps()
    except rospy.ROSInterruptException:
        pass
