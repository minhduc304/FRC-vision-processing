import cv2
#from opencv import *
import numpy as np
from numpy import savetxt
import imutils




cap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
	pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("HUE_Min", "Trackbars", 28, 179, empty)
cv2.createTrackbar("HUE_Max", "Trackbars", 87, 179, empty)
cv2.createTrackbar("SAT_Min", "Trackbars", 3, 255, empty)
cv2.createTrackbar("SAT_Max", "Trackbars",255, 255, empty)
cv2.createTrackbar("VAL_Min", "Trackbars", 64, 255, empty)
cv2.createTrackbar("VAL_Max", "Trackbars", 255, 255, empty)


def hexagonDetector(c):
	shape = ""
	perimeter = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

	if len(approx) == 6:
		shape = "hexagon" 
	
	return shape

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

	_, frame = cap.read()
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
	kernel = np.ones((5, 5), np.uint8) #remove noise pixels
	mask = cv2.erode(mask, kernel)
	
	#Define contours
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE)

	# points = cv2.findNonZero(mask) 
	# a = points[:,0,0].min()
	# b = points[:,0,0].max()
	# c = points[:,0,1].min()
	# d = points[:,0,1].max()
	# c0 = (a+b)/2
	# c1 = (c+d)/2
	# y = c0.astype(int)
	# x = c1.astype(int)
	# coordinates = [x, y]


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
			if len(approx) == 4: 
				cv2.putText(frame, "Rectangle", (x, y), font, 1, (0,0,0))
			elif len(approx) == 6:
				cv2.putText(frame, "Hexagon", (x, y), font, 1, (0,0,0))
		
	
	result = cv2.bitwise_and(frame, frame, mask = mask)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	hStack = cv2.hconcat([frame, mask])

	cv2.imshow("RESULT", hStack)
	#cv2.imshow("MASK", mask)


	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	elif cv2.waitKey(1) & 0xff == ord('c'):
		file = open("coordinates.txt", "w+")
		for n in coordinates:
			n = str(n) 
			file.write("%s\n" % n)

		file.close()


cap.release()
cv2.destroyAllWindows()	







































