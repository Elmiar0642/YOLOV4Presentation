import numpy as np
import cv2
import os, sys
import imutils
from scipy import spatial as sp
import collections
#import pandas as pd

## List of Lists FOR KEEPING THE CENTROIDS OF THE OBJECTS IN TRACK

personwise_track = []*0

NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2



def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []
	count = 0

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			#print(detection)
			#print(scores)
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:
				
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				## Arima : Color codes can be created from Centroids and Area. Centroids are relatively easier. 
				# If centroid goes on going towards up and vanish, then its going away, 
				# if the centroids are going towards down, then its coming towards.
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	after_NMS = []*0
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			count+=1

			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			after_NMS.append(centroids[i])
			res = (confidences[i], (x, y, x + w, y + h), centroids[i], count)
			results.append(list(res))
	# return the list of results
	return results, len(after_NMS)



labelsPath = "/home/arima/Downloads/yolo_pres/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "/home/arima/Downloads/yolo_pres/yolov4-tiny.weights"
config_path = "/home/arima/Downloads/yolo_pres/yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

#cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/video_2022-04-07_13-07-25.mp4")
#cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/video_2022-04-07_13-09-50.mp4")
#cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/video_2022-04-07_13-09-52.mp4")
#cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/video_2022-04-07_13-09-55.mp4")
#cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/video_2022-04-07_13-11-48.mp4")
cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/video_2022-04-07_13-07-25.mp4")
#cap = cv2.VideoCapture("/home/arima/Downloads/yolo_pres/mixkit-people-in-the-subway-hall-in-tokyo-4454.mp4")

writer = None

f_c = 0
while True:
	(grabbed, image) = cap.read()

	if not grabbed:
		break
	image = imutils.resize(image, width=700)
	h, w = image.shape[0], image.shape[1]
	#print(h,w)
	results, numbers = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))
	
	centroids_of_this_frame = []*0
	areas_of_bounding_boxes_of_this_frame = {}
	for res in results:
		centroids_of_this_frame.append(res[2])
		areas_of_bounding_boxes_of_this_frame.update({res[2]:res[1][2]*res[1][3]})
	
	abbtf = sorted(areas_of_bounding_boxes_of_this_frame.items(), key = lambda kv:(kv[1], kv[0]))

	print(abbtf)
	
	'''
	for i in range(len(abbtf)):
		if(i<=len(abbtf)):
			abbtf[i] = list(abbtf[i])
			abbtf[i].append("")

	input()
	'''
	neigh = {"orange":[], "green":[], "red":[]}
	vote = {}
	vote_status = {}
	for i in centroids_of_this_frame:
		vote.update({i:[]*0})
	
	'''for j in range(1, len(centroids_of_this_frame)):
		if(sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0]) >= 25.0 and sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0]) <= 50.0):
			neigh["orange"].append((centroids_of_this_frame[0], centroids_of_this_frame[j], sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0])))
		elif(sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0]) != 0.0 and sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0]) <= 25.0):
			neigh["red"].append((centroids_of_this_frame[0], centroids_of_this_frame[j], sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0])))
		elif(sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0]) >= 50.0):
			neigh["green"].append((centroids_of_this_frame[0], centroids_of_this_frame[j], sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[0])))
	'''

	for i in range(len(centroids_of_this_frame)):
		for j in range(i, len(centroids_of_this_frame)):
			if(sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i]) >= 25.0 and sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i]) <= 50.0):
				neigh["orange"].append((centroids_of_this_frame[i], centroids_of_this_frame[j], sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i])))
				vote[centroids_of_this_frame[i]].append("orange")
				vote[centroids_of_this_frame[j]].append("orange")
			
			elif(sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i]) != 0.0 and sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i]) <= 25.0):
				neigh["red"].append((centroids_of_this_frame[i], centroids_of_this_frame[j], sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i])))
				vote[centroids_of_this_frame[i]].append("red")
				vote[centroids_of_this_frame[j]].append("red")
				
			elif(sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i]) >= 50.0):
				neigh["green"].append((centroids_of_this_frame[i], centroids_of_this_frame[j], sp.distance.euclidean(centroids_of_this_frame[j],centroids_of_this_frame[i])))
				vote[centroids_of_this_frame[i]].append("green")
				vote[centroids_of_this_frame[j]].append("green")
				

	#colors = collections.OrderedDict(neigh)
	#print(neigh)
	#print(vote)
	#print()
	#print()
	#print("centroids : ", centroids_of_this_frame)
	#neighbours = sp.KDTree(centroids_of_this_frame)
	#print("neigbours", neighbours.count_neighbors(neighbours, r = 1))
	#dd, ii = neighbours.query(centroids_of_this_frame, k=1)
	#print("query : ", dd, ii, sep="\n")
	#input()

	for res in results:
		if("orange" in vote[res[2]]):
			# going away : script kiddie version
			if (h*w) - (res[1][2]*res[1][3])<100:
				cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 165, 255), 2)
				cv2.putText(image, str(res[-1])+"D", (res[1][0],res[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #Depart
			#coming towards : script kiddie version
			else:
				cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 165, 255), 2)
				cv2.putText(image, str(res[-1])+"A", (res[1][0],res[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #Approach
		
		elif("red" in vote[res[2]]):
			# going away : script kiddie version
			if (h*w) - (res[1][2]*res[1][3])<100:
				cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 0, 255), 2)
				cv2.putText(image, str(res[-1])+"D", (res[1][0],res[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #Depart
			#coming towards : script kiddie version
			else:
				cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 0, 255), 2)
				cv2.putText(image, str(res[-1])+"A", (res[1][0],res[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #Approach

		else:
			# going away : script kiddie version
			if (h*w) - (res[1][2]*res[1][3])<100:
				cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
				cv2.putText(image, str(res[-1])+"D", (res[1][0],res[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #Depart
			#coming towards : script kiddie version
			else:
				cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
				cv2.putText(image, str(res[-1])+"A", (res[1][0],res[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #Approach
		
	# No. Of Frames
	cv2.putText(image, str(numbers), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)

	cv2.imshow("Detection",image)
	f_c += 1
	if(f_c%10==0):
		cv2.imwrite("/home/arima/Downloads/yolo_pres/frames_op_with_cnt/frame_{}.jpg".format(f_c), image)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()