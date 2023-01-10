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

logging.info("[{}] waiting for camera to warmup".format(
	datetime.datetime.now()))
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)




def empty(a):
	pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("HUE_Min", "Trackbars", 24, 179, empty)
cv2.createTrackbar("HUE_Max", "Trackbars", 83, 179, empty)
cv2.createTrackbar("SAT_Min", "Trackbars", 94, 255, empty)
cv2.createTrackbar("SAT_Max", "Trackbars",255, 255, empty)
cv2.createTrackbar("VAL_Min", "Trackbars", 183, 255, empty)
cv2.createTrackbar("VAL_Max", "Trackbars", 255, 255, empty)



font = cv2.FONT_HERSHEY_SIMPLEX




while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	

	h_min = cv2.getTrackbarPos("HUE_Min", "Trackbars")
	h_max = cv2.getTrackbarPos("HUE_Max", "Trackbars")
	s_min = cv2.getTrackbarPos("SAT_Min", "Trackbars")
	s_max = cv2.getTrackbarPos("SAT_Max", "Trackbars")
	v_min = cv2.getTrackbarPos("VAL_Min", "Trackbars")
	v_max = cv2.getTrackbarPos("VAL_Max", "Trackbars")

	lower_green = np.array([h_min, s_min, v_min])
	upper_green = np.array([h_max, s_max, v_max])


	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = cv2.erode(mask_green, None, iterations=2)
	mask = cv2.dilate(mask_green, None, iterations=2)

	
	#Define contours
	contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, 
								cv2.CHAIN_APPROX_SIMPLE)

	contours = imutils.grab_contours(contours)



	for c in contours_green:
		M = cv2.moments(c)

		area = cv2.contourArea(c)
		approx = cv2.approxPolyDP(c, 0.025*cv2.arcLength(c, True), True)
		x = approx.ravel()[0] #to put text on contour of object
		y = approx.ravel()[1]

		if area > 100: # only detect if object is large enough

			cX = int(M["m10"]/M["m00"]) # Coordinates of centroid
			cY = int(M["m01"]/M["m00"]) 
			coordinates = [cX, cY] #Y-axis is inverted (above-to-below)

			cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
			if len(approx) == 6:
				cv2.putText(frame, "Hexagon", (x, y), font, 1, (0,0,0))
				
		
				file = open("coordinates.txt", "w+")
				for n in coordinates:
					n = str(n) 
					file.write("%s\n" % n)

				file.close()


		
	
	result = cv2.bitwise_and(frame, frame, mask = mask)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	hStack = np.hstack([frame, mask])

	cv2.imshow("RESULT", hStack)


	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	
	

cv2.destroyAllWindows()	
vs.stop()
