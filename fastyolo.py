#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import argparse

import os
import globals
from scipy.spatial import distance
import collections

def yoloCall():
    img_path = 'E:/BE PROJECT/Flask/static/frames/'
    images = findKeyframeImage()
    for i in range(len(images)):
        print(images[i][0])
        foo(img_path+images[i][0])
        
    globals.classIDS_2=np.array(globals.classIDS_2)
    print(globals.classIDS_2)
    freq = CountFrequency(globals.classIDS_2)
    freq = freq.most_common(5)
    globals.keys =  [key for key, _ in freq]
    values =  [value for _, value in freq]
    countFrames = sum(values)
    globals.values2 = np.array(values)/countFrames
    return images

def findKeyframeImage():    
    images = []
    cntIndex=0
    for c in range(globals.n_clusters_q) :
        cnt0 = globals.df_vectors[(globals.df_vectors['Cluster']==c)].count()
        match = np.zeros((1,cnt0['ImgName']))
        cntIndex=0
        dfnew = pd.DataFrame(data=None,columns = ["Img","Dist"])
        for index,row in globals.df_vectors.iterrows(): 
            if row['Cluster'] ==  c:
                match[0][cntIndex] = distance.euclidean(globals.queryKeyframes[0], row['FeatureVector'])
                #print(match[0][cntIndex])
                dfnew = dfnew.append({"Img": row['ImgName'],"Dist" :match[0][cntIndex] },ignore_index=True)
                cntIndex = cntIndex+1
        minValue = match.min(axis=1)
        indexOfMin = np.where(match[0]==minValue)
        finalimg = dfnew.loc[indexOfMin]['Img'].tolist()
        #print(finalimg)
        images.append(finalimg)
    return images
    



def foo(fname):
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args('--yolo yolo-coco'.split()))
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
    
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    df=pd.read_csv('yolo-coco/names.csv')
    image = cv2.imread(fname)
    print('Image grabbed : '+fname)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    print('Done: '+fname)
    print('ClassIDs')
    for x in range(len(classIDs)):
        print (classIDs[x])
        
    for obj in classIDs:
        print("Inside : ",fname)
        val=df.iloc[obj,:].to_string(header=None, index=None)
        globals.df2 = globals.df2.append({'ClassID':obj,'Object Found': val}, ignore_index=True)
        globals.classIDS_2.append(obj)
    print(globals.classIDS_2)
    
    #return pd.Series(classIDS_2)

def CountFrequency(arr): 
    return collections.Counter(arr)



