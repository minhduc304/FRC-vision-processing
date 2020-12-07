import cv2
#from opencv import *
import numpy as np
from numpy import savetxt



frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)

#cap = cv2.resize(cap, (400, 540))
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
	pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 500, 140)


user = input("What color do you want to detect?: ")

profiles = [[96, 179, 90, 255, 62, 255], #blue
		   [40, 93, 72, 255, 0, 255],  #green
		   [3, 25, 52, 255, 104, 212], 	 #yellow
		   [150, 179, 92, 255, 66, 255]] #red

def changeTrackbarPos(user, profiles):
	profile = []
	if user == "y":
		profile = profiles[2]
	elif user =="b":
		profile = profiles[0]
	elif user == "g":
		profile = profiles[1]
	elif user == "r":
		profile = profiles[3]

	cv2.createTrackbar("HUE Min", "HSV", profile[0], 179, empty)
	cv2.createTrackbar("HUE Max", "HSV", profile[1], 179, empty)
	cv2.createTrackbar("SAT Min", "HSV", profile[2], 255, empty)	
	cv2.createTrackbar("SAT Max", "HSV", profile[3], 255, empty)
	cv2.createTrackbar("VALUE Min", "HSV", profile[4], 255, empty)
	cv2.createTrackbar("VALUE Max", "HSV", profile[5], 255, empty)

changeTrackbarPos(user, profiles)




while True:
	_, imageFrame = cap.read()
	imageHSV = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)	

	h_min = cv2.getTrackbarPos("HUE Min", "HSV")
	h_max = cv2.getTrackbarPos("HUE Max", "HSV")
	s_min = cv2.getTrackbarPos("SAT Min", "HSV")
	s_max = cv2.getTrackbarPos("SAT Max", "HSV")
	v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
	v_max = cv2.getTrackbarPos("VALUE Max", "HSV")


	lower = np.array([h_min, s_min, v_min])
	upper = np.array([h_max, s_max, v_max])
	mask = cv2.inRange(imageHSV, lower, upper)
	

	points = cv2.findNonZero(mask) 
	a = points[:,0,0].min() 
	b = points[:,0,0].max()
	c = points[:,0,1].min()
	d = points[:,0,1].max()
	c0 = (a+b)/2
	c1 = (c+d)/2
	y = c0.astype(int)
	x = c1.astype(int)
	coordinates = [x, y] # find coordinates of pixels that
                             # are not blacked out

	if user == "y":
		coordinates.insert(0,"yellow")
	elif user =="b":
		coordinates.insert(0,"blue")
	elif user == "g":
		coordinates.insert(0,"green")
	elif user == "r":
		coordinates.insert(0,"red")


	

	if cv2.waitKey(10) & 0xff == ord('q'):
		break
	elif cv2.waitKey(10) & 0xff == ord('p'):
		file = open("coordinates.txt", "w+")
		for n in coordinates:
			n = str(n)
			file.write("%s\n" % n) # Write coordinates to file
                                               # to send to robot
		
		file.close()	


	
	result = cv2.bitwise_and(imageFrame, imageFrame, mask = mask)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	hStack = np.hstack([result, mask])


	cv2.imshow("Horizontal Stacking", result)

	


cap.release()
cv2.destroyAllWindows()	



#b(lue) = 98
#r(ed) = 114
#y(ellow) = 121
#g(reen) = 103









































