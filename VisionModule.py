import cv2
from imutils.video import VideoStream
import numpy as np
from numpy import savetxt
import imutils
import argparse
import logging
import time
from networktables import NetworkTables


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-l", "--log", type=str, default="log.txt",
	help="path to output log file")
args = vars(ap.parse_args())


logging.basicConfig(filename=args["log"], level=logging.DEBUG)
# initialize the video stream and allow the cammera sensor to
# warmup


vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
# define the lower and upper boundaries of the "green"
# ball in the HSV color space

NetworkTables.initialize(server="10.61.62.2")
sd = NetworkTables.getTable("SmartDashboard")




def empty(a):
	pass



font = cv2.FONT_HERSHEY_SIMPLEX


lower_green = np.array([44, 40, 64])
upper_green = np.array([87, 255, 255])


while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	

	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	
	
	#Define contours
	contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, 
								cv2.CHAIN_APPROX_SIMPLE)

	contours = imutils.grab_contours(contours)



	for c in contours:
		M = cv2.moments(c)

		area = cv2.contourArea(c)
		approx = cv2.approxPolyDP(c, 0.025*cv2.arcLength(c, True), True)
		x = approx.ravel()[0] #to put text on contour of object
		y = approx.ravel()[1]

		if area > 800: # only detect if object is large enough

			cX = int(M["m10"]/M["m00"]) # Coordinates of centroid
			cY = int(M["m01"]/M["m00"]) 
			coordinates = [cX, cY] #Y-axis is inverted (above-to-below)

			cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
			if len(approx) == 6:
				logging.info("Target found!")
				cv2.putText(frame, "Hexagon", (x, y), font, 1, (0,0,0))
				
				file = open("coordinates.txt", "w+")
				for n in coordinates:
					n = str(n) 
					file.write("%s\n" % n)

				file.close()

				sd.putNumber("X", cX)
				sd.putNumber("Y", cY)
		




logging.info("[{}] cleaning up".format(
	datetime.datetime.now()))




cv2.destroyAllWindows()	
vs.stop()

#299, 253




































