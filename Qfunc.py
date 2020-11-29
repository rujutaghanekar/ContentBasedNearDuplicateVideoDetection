# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:02:24 2019

@author: JUIE
"""
import cv2
import numpy as np
import pandas as pd
import pickle
import os
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras import backend
#from sklearn.cluster import DBSCAN
from scipy import spatial
from kneed import KneeLocator 
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from fastyolo import yoloCall
from sklearn.cluster import DBSCAN
import globals
from globals import initialize
import glob
from shutil import copyfile
#new query video extraction
def VideoToFrames(pathName,vidName):
   
    globals.videoName= vidName
    imagesFolder = "E:/BE PROJECT/Flask/static/frames/"
    print(vidName)
    vidcap = cv2.VideoCapture(pathName + '.mp4')
    fps = vidcap.get(cv2.CAP_PROP_FPS)      
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    globals.duration = frameCount/int(fps)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,((globals.count)*1000))    # added this line 
      success,image = vidcap.read()
      if success == True :
          print ('Read a new frame: ', success)
          filename = imagesFolder + globals.videoName +"_" +  str(int(globals.count)) + ".jpg"
          cv2.imwrite(filename, image)    # save frame as JPEG file
          globals.count = globals.count + 1
    return ('For '+globals.videoName + ' Frame Extraction completed')
	
def createFeatureVectors():
    model_pkl = open("E:/BE PROJECT/Flask/inception/classifier.pkl", 'rb')
    inception_model = pickle.load(model_pkl)
    print('Model successfully loaded')
    backend.set_learning_phase(0)
	
    img_path = 'E:/BE PROJECT/Flask/static/frames/'

    filenames = os.listdir(r'E:/BE PROJECT/Flask/static/frames/')

    for i, fname in enumerate(filenames):
        img = image.load_img(img_path+fname, target_size=(299, 299))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature_q = inception_model.predict(img_data)
        feature_np_q = np.array(feature_q)
        globals.feature_list_q.append(feature_np_q.flatten())
        globals.df_vectors = globals.df_vectors.append({'ImgName': fname,"FeatureVector" : feature_np_q.flatten()},ignore_index=True)
        print("FV generated for:",i)
    print ('Feature extraction done')
    backend.clear_session()
    return ('For '+globals.videoName + ' Feature Extraction completed')
    
def DBScanOnFeatureVectors(radius,density):
    
    feature_list_np_q = np.array(globals.feature_list_q)

    db = DBSCAN(radius,density).fit(globals.feature_list_q)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    globals.n_clusters_q = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_q = list(labels).count(-1)
    print('Estimated number of clusters: %d' % globals.n_clusters_q)
    print('Estimated number of noise points: %d' % n_noise_q)
    globals.df_vectors['Cluster'] = pd.Series(labels)
    globals.df_vectors = globals.df_vectors.sort_values(by=['Cluster'])
    globals.indices = db.core_sample_indices_
    
    unique_labels = set(labels)
    if(len(unique_labels)==1):
         globals.silhouette_index = 0.20
    else:
        globals.silhouette_index= silhouette_score(feature_list_np_q,labels)
    print(globals.silhouette_index)
    
    x = np.array( [ num for num in globals.df_vectors["Cluster"] if num >= 0 ] )
    neg = np.array( [ num for num in globals.df_vectors["Cluster"] if num < 0 ] )
    globals.y = np.bincount(x)
    fCount = globals.count-sum(1 for i in neg if i < 0)
    if fCount != 0 :
        globals.y = globals.y/fCount
    return ('For '+globals.videoName + ' Clustering completed')

def findKneeValue(mu):
    
    filenames = os.listdir(r'E:/BE PROJECT/Flask/static/frames')
    feature_list_np = np.array(globals.feature_list_q)
    myu=mu
    ysize = len(filenames)+1 
    neigh = NearestNeighbors(n_neighbors=myu) 
    nbrs = neigh.fit(feature_list_np) 
    dist, ind = nbrs.kneighbors(feature_list_np,return_distance=True) 
    distanceDec = sorted(dist[:,myu-1], reverse=False) 
   
    kn = KneeLocator(list(range(1,ysize)), distanceDec, curve='convex', direction='increasing') 
    epsilon = np.interp(kn.knee, list(range(1,ysize)), distanceDec) 
     
    kn.plot_knee() 
    plt.xlabel('Sample points') 
    plt.ylabel('Epsilon') 
    plt.plot(list(range(1,ysize)), distanceDec) 
    plt.hlines(epsilon, plt.xlim()[0], plt.xlim()[1], linestyles='dashed') 
    print("Knee is at : {},{}".format(kn.knee,epsilon))
 
    return epsilon


def AutomateDBParameters():

    mu = 2
    outlier = globals.count
    eps = 0
    while globals.silhouette_index < 0.20 or outlier > 0.2:
         mu = mu + 1
         eps = findKneeValue(mu)
         print("eps selected",eps)
         DBScanOnFeatureVectors(eps,mu)
         outlier = globals.n_noise_ / globals.count
       # globals.silhouette_index = globals.silhouette_score(feature_list_np,labels)   
    
    print("Final eps selected",eps)
    print("Final mu selected",mu)
    
    X = TSNE(n_components=2,random_state=123)
    feature_list_np_q = np.array(globals.feature_list_q)
    globals.X_embedded = X.fit_transform(feature_list_np_q)
    scaler = MinMaxScaler(feature_range=(0, 1),copy=False)
    globals.scaled = scaler.fit_transform(globals.X_embedded)
    
    for c in range(globals.n_clusters_q) :
        NewFVList = []
        for index,row in globals.df_vectors.iterrows(): 
            if row['Cluster'] ==  c:
                NewFVList.append(np.array(globals.scaled[0]).tolist())
        color = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                             for k in range(1)]
        globals.df_scatterPlot=globals.df_scatterPlot.append({"Cluster":c,"Cluster FV":NewFVList,"Color":color},ignore_index=True)       
    print(globals.df_scatterPlot)
    
#    db = DBSCAN(eps,mu).fit(globals.scaled)
#    labels = db.labels_
#    core_samples_mask = np.zeros_like(labels, dtype=bool)
#    core_samples_mask[db.core_sample_indices_] = True
#    unique_labels = set(labels)
#    plt.cla()
#    colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels))]
#    for k, col in zip(unique_labels, colors):
#        if k == -1:
#        # Black used for noise.
#            col = [0, 0, 0, 1]
#
#        class_member_mask = (labels == k)
#
#        xy = globals.scaled[class_member_mask & core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)
#
#        xy = globals.scaled[class_member_mask & ~core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
#
#    plt.title('Estimated number of clusters: %d' % globals.n_clusters_q)
    return ("Done")

    
	
#Keyframe extraction Query Video
def generateKeyFrames():
    
    queryImgCnt= np.zeros(globals.n_clusters_q)
    globals.queryKeyframes = np.zeros((globals.n_clusters_q,2048))
    for c in range(globals.n_clusters_q) :
        for index,row in globals.df_vectors.iterrows(): 
            if row['Cluster'] ==  c:
                print (row['ImgName'])
                queryImgCnt[c] = queryImgCnt[c]+1
                for x in range(2048) :
                    globals.queryKeyframes[c][x] = globals.queryKeyframes[c][x] + row['FeatureVector'][x]
    for c in range(globals.n_clusters_q) :           
        for x in range(2048):
            globals.queryKeyframes[c][x]=globals.queryKeyframes[c][x]/queryImgCnt[c]
    return ('For '+globals.videoName + ' KeyFrame Extraction completed')

#perform Yolo to extract tags
#Run yolo code and store tag for each video
def DatabaseVideoIDExtraction():

    globals.VideoIdx=[]
    df_tags = pd.DataFrame(data=None,columns = ["VideoID","Tags","Contribution"])
    df_tags = pd.read_csv("E:/BE PROJECT/Flask/yolotags.csv")
        
    
    arr = df_tags.drop("VideoID",axis=1)
    arr=df_tags.iloc[:,1] 
    arr2=df_tags.iloc[:,2]
    arr2 = pd.Series(arr2)
    
    newarr = df_tags['Contribution']
    newarr.tolist()
    """
    # arr2=  pd.np.array(arr2)
    #arr2.tolist()
    newarr = [[]]
    
    finalListDB = []
    finalListContriDB = []
    for index,row in df_tags.iterrows():
        for i in range(len(row['Contribution'])):
            row['Contribution'][i]>0.1
            finalListDB.append(row['Tags'][i])
                    
	"""
    query_tags = globals.keys 
    #query_tags = [41,45]
    query_tags_contri = globals.values2
    
    finalList = []
    finalListContri = []
    for i in range(len(query_tags)):
        if query_tags_contri[i]>0.1:
            finalList.append(query_tags[i])
            finalListContri.append(query_tags_contri[i])
            
 
    finalList = [int(i) for i in finalList]
    finalList = pd.to_numeric(finalList)
    
    
    for j in range(len(df_tags)):
        vid_tags = arr[j]
        vid_tags = vid_tags[1:-1]
        vid_tags = vid_tags.split(',')
        vid_tags = [int(i) for i in vid_tags]
        vid_tags = pd.to_numeric(vid_tags)
        intersection = list(set(vid_tags) & set(finalList))
        if (len(intersection)/len(vid_tags))>=0.5:
            print("get videoID please")
            globals.VideoIdx.append(df_tags.iloc[j]['VideoID'])
            #VideoIdx.append(df_tags.iloc[j]['Name'])
            #appends all video ids which have corresponding tags    
    return ('For '+globals.videoName + ' DatabaseVideoIDExtraction completed')
	
def findSimilarityMatrix():
    globals.dbCompared = 0
    for filename in os.listdir('E:/BE PROJECT/Flask/Video Keyframes/'):
        if filename.endswith(".csv"):
            val = os.path.splitext(filename)[0]
            val = pd.to_numeric(val)
            if val in globals.VideoIdx:
                globals.dbCompared = globals.dbCompared + 1
                print(filename)
                df_xx = pd.read_csv("E:/BE PROJECT/Flask/Video Keyframes/"+filename)
                arr_key=np.asarray(df_xx)
                print(arr_key)
                similarity = np.zeros((globals.queryKeyframes.shape[0],arr_key.shape[0]))
                for i in range(globals.queryKeyframes.shape[0]) :
                    for j in range(arr_key.shape[0]) :
                        similarity[i][j] = spatial.distance.euclidean(globals.queryKeyframes[i], arr_key[j])
                        print(similarity[i][j])
                        
                minArray = np.min(similarity,axis=1)
                thresholdArray = minArray
                minArray = np.reciprocal(minArray)
                sumMinArray = sum(minArray)
                minArray = minArray/sumMinArray
                
                finalContribution = np.array(minArray)
                for i in range(minArray.size) :
                   finalContribution[i] = minArray[i] * (globals.y[i])
                   
                Sum = sum(finalContribution)
                finalContribution = finalContribution/Sum
                
                for i in range(minArray.size):
                    globals.color = ["#"+"".join([random.choice('0123456789ABCDEF') for j in range(6)])
                             for k in range(globals.n_clusters_q)]
                    print(globals.color)
                    #vid = str(val)
                    #print(vid)
                    s1 ,s2 = replaceWithImages(val,i,similarity[i].argmin())
                    if thresholdArray[i]<15:
                        globals.df_similarity=globals.df_similarity.append({'videoName':val,'Contribution': finalContribution[i],'Label':'Similar','ClusterPairImage':(s1,s2),'ClusterPairNumeric':(i,similarity[i].argmin())},ignore_index=True)       
                    else:
                        globals.df_similarity=globals.df_similarity.append({'videoName':val,'Contribution': finalContribution[i],'Label':'Dissimilar','ClusterPairImage':(s1,s2),'ClusterPairNumeric':(i,similarity[i].argmin())},ignore_index=True)       
    print (globals.df_similarity)
    return ('For '+globals.videoName + ' Result calculated')

def replaceWithImages(vid,a,b):
    s1 = ""
    s2 = ""
    df_Q = globals.df_storeImageThumbnailQuery.loc[globals.df_storeImageThumbnailQuery['Cluster name']==a]
    
    df_Q = df_Q.iloc[: , 0]
    s1 = df_Q.tolist()
    s1 = s1[0]
    print("s1",s1)
    
    print("Vid:",vid)
    globals.df_storeImageThumbnail = pd.read_csv("E:/BE PROJECT/Flask/DBThumbnails.csv")
    print( globals.df_storeImageThumbnail )
    
    df_DB = globals.df_storeImageThumbnail.loc[globals.df_storeImageThumbnail['Video id']==vid]
    print(df_DB)
    df_DB2 = df_DB.loc[df_DB['Cluster name']==b]
    print(df_DB2)
    df_DB2 = df_DB2.iloc[: , 0]
    print(df_DB2)
    s2 = df_DB2.tolist()
    print(s2)
    s2 = s2[0]
    print("s2",s2)
    return s1,s2 #Query,DB
    

def generateClusterImages():
    df_thumnail = pd.DataFrame(data=None,columns = ["ImgName","Cluster","FeatureVector"])
    ImgNames = ""
    ImageList = []
    for c in range(globals.n_clusters_q) :
        print(c)
        df_thumnail=globals.df_vectors.loc[globals.df_vectors['Cluster'] == c]
        df_thumnail = df_thumnail.iloc[random.randrange(5) , : ]
        ImgNames = df_thumnail['ImgName']
        print(ImgNames)
        xyz = "E:/BE PROJECT/Flask/static/frames\\"
        ImageList.append(str(xyz+ImgNames))  
    return ImageList
        
def make_image_thumbnail(filename):
    base_filename, file_extension = os.path.splitext(filename)
    thumbnail_filename = f"{base_filename}_thumbnail{file_extension}"
    image = Image.open(filename)
    image.thumbnail(size=(128, 128))
    image.save(thumbnail_filename, "JPEG")

    return thumbnail_filename

def createThumbnail(img):
    cnt = 0
    ImageList = []
    for c in range(globals.n_clusters_q) :
        ImgNames = img[c][0]
        xyz = "E:/BE PROJECT/Flask/static/frames\\"
        ImageList.append(str(xyz+ImgNames))
    print(ImageList)
    for image_file in glob.glob("E:/BE PROJECT/Flask/static/frames/*.jpg"):
        print(image_file)
        if image_file in ImageList:
            print('here')
            thumbnail_file = make_image_thumbnail(image_file)
            print(thumbnail_file)
            newSrcname = globals.videoName+"_"+str(int(cnt))+"_"+'.jpg'
            globals.df_storeImageThumbnailQuery = globals.df_storeImageThumbnailQuery.append({'Image Name':newSrcname,'Cluster name':cnt},ignore_index=True)
            copyfile(thumbnail_file, 'E:/BE PROJECT/Flask/static/VideoThumbPrints/'+newSrcname)
            cnt = cnt + 1
    return ("Thumbnails generated")

def getMetadata():
    globals.df_metadata = globals.df_metadata.append({"VideoName":globals.videoName+".mp4","Query Clusters":globals.n_clusters_q,"DBvideosCompared":globals.dbCompared,"Duration":globals.duration,"TotalFrames":globals.count,"Tags":globals.keys},ignore_index=True)
    print(globals.df_metadata)

def deleteFrames():
    folder = 'E:/BE PROJECT/Flask/static/frames'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        #print(file_path)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    return("Frames Deleted")
    
if __name__ == '__main__':
    initialize()
    result1 = VideoToFrames('query')
    result2 = createFeatureVectors()
    result3 = AutomateDBParameters()
    result4 = generateKeyFrames()
    yoloR = yoloCall()
    print(globals.keys)
    print(globals.values2)
    result5 = DatabaseVideoIDExtraction()
    img = generateClusterImages() #list of all images to generate thumbnails
    #print(img)
    #print("generateClusterImages done")
    result7 = createThumbnail(img)
    print( globals.df_storeImageThumbnailQuery )
    result6 = findSimilarityMatrix()
    
