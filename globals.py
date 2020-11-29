import pandas as pd

def initialize():
    global count,df_vectors,feature_list_q,videoName,n_clusters_q,queryKeyframes,y,VideoIdx,count,df_similarity,df2,keys,silhouette_index,n_noise_,W,H,values2,classIDS_2,df_storeImageThumbnailQuery,df_storeImageThumbnail,X_embedded,scaled,df_metadata,duration,dbCompared,df_scatterPlot,color
	
    
    df_vectors = pd.DataFrame(data=None,columns = ["ImgName","Cluster","FeatureVector"])
    feature_list_q = []
    videoName = ""
    n_clusters_q = 0
    y = 0
    VideoIdx = []
    count = 0
    #df_similarity = pd.DataFrame(data=None,columns = ["videoName","Contribution","Label","Cluster pair"])
    df_similarity = pd.DataFrame(data=None,columns = ["videoName","Contribution","Label","ClusterPairImage","ClusterPairNumeric"])
    df2 = pd.DataFrame(data=None,columns = ["ClassID","Object Found"])
    df_storeImageThumbnailQuery = pd.DataFrame(data=None,columns=["Image Name","Cluster name"])
    df_storeImageThumbnail = pd.DataFrame(data=None,columns=["Image Name","Cluster name","Video id"])
    keys = []
    silhouette_index = -1
    n_noise_ = 0
    W = 0
    H = 0
    values2 = []
    classIDS_2 = []
    X_embedded = []
    scaled = []
    df_scatterPlot = pd.DataFrame(data=None,columns = ["Cluster","ClusterFV","Color"])
    df_metadata = pd.DataFrame(data=None,columns = ["VideoName","Query Clusters","DBvideosCompared","Duration","TotalFrames","Tags"])