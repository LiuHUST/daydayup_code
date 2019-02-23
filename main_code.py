# -*- coding: utf-8 -*-

'''
   Dataset source : https://www.cse.ust.hk/scrg/
   This dataset is consits of 4316 files,each file records the gps track of a taxi on 2007-02-20.
   
    modules to install:
        pip install pandas
        pip install numpy
        pip install matplotlib
        pip install basemap
        pip install sklearn
        pip install os
        
    file to download: CHN_adm

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from sklearn.cluster import DBSCAN
import os
import geohash
from sklearn.cluster import KMeans

# plot recommendation_model's result
def DrawPointMap(lat,lon,val,hour):
    colors = {}  
    cmap = plt.cm.RdYlBu_r
    colors =  300*(val-0.1)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])#[left,bottom,width,height]
    map = Basemap(projection='mill',llcrnrlat=np.min(lat) ,urcrnrlat=np.max(lat),llcrnrlon=np.min(lon),urcrnrlon=np.max(lon),\
		     ax=ax1,rsphere=6371200.,resolution='h',area_thresh=1)
    shp_info = map.readshapefile('CHN_adm/CHN_adm3','states',drawbounds=False)
    for info, shp in zip(map.states_info, map.states):
        proid = info['NAME_1']
        if proid == 'Shanghai':
            poly = Polygon(shp,facecolor='w',edgecolor='k', lw=1.0, alpha=0.1)
            ax1.add_patch(poly)		
    # plot the regional boundary
    map.drawmapboundary()  
    map.drawstates()        
    map.drawcountries() 
    # plot the parallels and meridians
    parallels = np.arange(30,32,0.5) 
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10) #parallels
    meridians = np.arange(120,122.5,0.5)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10) #meridians
    # plot the location
    x,y = map(lon,lat)
    map.scatter(x, y,s=20, c = colors,vmin=0, vmax=50,cmap=cmap) 
    font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
    plt.title('recommended place at %d:00' % hour,font1)# title
    plt.show()

# plot the map of (31.20~31.25,121.5~121.55)   
def DrawPointMap_cluster(lat,lon,val,hour):
    colors = {}  
    cmap = plt.cm.RdYlBu
    colors =20*val

    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])#[left,bottom,width,height]
    map = Basemap(projection='mill',llcrnrlat=31.20 ,urcrnrlat=31.25,llcrnrlon=121.5,urcrnrlon=121.55,\
		     ax=ax1,rsphere=6371200.,resolution='h',area_thresh=1)
    shp_info = map.readshapefile('CHN_adm/CHN_adm3','states',drawbounds=False)
    for info, shp in zip(map.states_info, map.states):
        proid = info['NAME_1']
        if proid == 'Shanghai':
            poly = Polygon(shp,facecolor='w',edgecolor='k', lw=1.0, alpha=0.1)
            ax1.add_patch(poly)		
    # plot the regional boundary
    map.drawmapboundary()  
    map.drawstates()        
    map.drawcountries() 
    # plot the parallels and meridians
    parallels = np.arange(30,32,0.5) 
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10) #parallels
    meridians = np.arange(120,122.5,0.5)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10) #meridians
    # plot the location
    x,y = map(lon,lat)
    map.scatter(x, y,s=20, c = colors,cmap=cmap) 
    font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 23,
         }
    plt.title('Shanghai(31.20~31.25,121.5~121.55) at %d:00' % hour,font1)# tile
    plt.show()

# Plot the number of passengers at various times of the day
def plot_hour_passenger(taxi_data):
    fig = plt.figure(figsize=(10,5))
    hour_passenger = taxi_data.groupby(['hour'],as_index=False)['is_passenger_location'].agg({'passengers':'sum'})
    plt.plot(hour_passenger['hour'].values, hour_passenger['passengers'].values, marker='o')
    plt.xlabel(u"hours") # x lable
    plt.ylabel(u"Number of car hirers") # y label
    plt.title("Statistics on the number of passengers rented per hour") # title
    plt.xticks(range(24), hour_passenger['hour'].values)
    plt.xlim(0, 24)  

# Determine if the location is a passenger point
def is_passenger_location(taxi_data):
    taxi_data['is_vacant'] = (taxi_data['is_vacant'] > 0).astype(int)
    taxi_data.sort_values(['id','time'],inplace=True,ascending=False)
    taxi_data['is_passenger_location'] = taxi_data['is_vacant'] - taxi_data['is_vacant'].shift(1)
    taxi_data['is_passenger_location'] = (taxi_data['is_passenger_location'] > 0).astype(int)
    return taxi_data
    
# Visualized the clustering result at a specific hour
def plot_clustering_result(taxi_data,hour):
   car_hirer_data = taxi_data[taxi_data['hour']==hour]
   print(car_hirer_data.shape)
   # using DBSCAN cluster algrithom
   db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(car_hirer_data[['lon','lat']])
   car_hirer_data['block_id'] = db.labels_
   DrawPointMap_cluster(car_hirer_data['lat'].values,car_hirer_data['lon'].values,car_hirer_data['block_id'].values,hour)

# recommendation_model
def recommendation_model(taxi_data,hour):
    car_hirer_data = taxi_data[taxi_data['hour']==hour]
    # using DBSCAN cluster algrithom
    db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(car_hirer_data[['lon','lat']])
    car_hirer_data['block_id'] = db.labels_
    # calculate recommendation rate
    recommend = car_hirer_data.groupby(['block_id'],as_index=False)['is_passenger_location'].agg({'car_nums':'count','passenger':'sum'})
    recommend['recommend_val'] = recommend['passenger']/recommend['car_nums']
    # merge the "recommend_val" attribute for car_hirer_data 
    car_hirer_data = pd.merge(car_hirer_data,recommend[['block_id','recommend_val']],how='left',on=['block_id'])
    print(car_hirer_data.head(10))
    #  recommended passenger aboard location which has the recommend_val above 0.1 for driver.
    recommend_locations = car_hirer_data[car_hirer_data['recommend_val'] > 0.1]
    # Visualize the recommended passenger aboard location
    DrawPointMap(recommend_locations['lat'].values,recommend_locations['lon'].values,recommend_locations['recommend_val'].values,hour)

def read_files():
    pathDir =  os.listdir('F:\car_recomment\Taxi_070220')
    path = 'F:\car_recomment\Taxi_070220\\'
    merge_data = pd.DataFrame(columns=['id','time','lon','lat','speed','angle','is_vacant'])
    pd.set_option('max_columns',80)
    print(len(pathDir))

    for i in pathDir:
      taxi = pd.read_csv(path + i)
      taxi.columns = ['id','time','lon','lat','speed','angle','is_vacant']
      merge_data = merge_data.append(taxi)
      merge_data.to_csv('taxi_all.csv',index=False)  
    return merge_data

# use geohash to encode latitude and longitude
def latitude_longitude_to_go(data):
    tmp = data[['lon','lat']]
    start_geohash = []
    for i in tmp.values:
        start_geohash.append(geohash.encode(i[1],i[0],7))
    data['geohashed_loc'] = start_geohash
    return data

def calculate_precision_recall(recommend,passenger_location):
     # precision= ture location numbers/recommend set numbers
    yy = set(recommend['geohashed_loc']).intersection(set(passenger_location['geohashed_loc']))
    precission = float(len(yy))/ float(len(set(recommend['geohashed_loc'])))
    print("precission is : "+str(precission))  
    
    # recall = recommend passenger numbers/total passenger numbers
    tmp = [val for val in list(passenger_location['geohashed_loc']) if val in set(recommend['geohashed_loc'])]
    recall = float(len(tmp)) / float(len(list(passenger_location['geohashed_loc'])))
    print("recall is : " + str(recall))
    
    return precission,recall

# method1: using DBSCAN cluster method
def run_dbscan_method(taxi_data):
    precission_set = []
    recall_set = []
    for t in range(24):
         hour_taxi_data = taxi_data[taxi_data['hour']==t]
         passenger_location =  hour_taxi_data[hour_taxi_data['is_passenger_location']==1]
         print(passenger_location.shape)
         db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(hour_taxi_data[['lon','lat']])
         hour_taxi_data['block_id'] = db.labels_
         # hour_taxi_data recommendation rate
         recommend = hour_taxi_data.groupby(['block_id'],as_index=False)['lon','lat'].mean()
         recommend = latitude_longitude_to_go(recommend)
    
         passenger_location = passenger_location[['lon','lat']]
         passenger_location = latitude_longitude_to_go(passenger_location)
         precission,recall = calculate_precision_recall(recommend,passenger_location)
         precission_set.append(precission)
         recall_set.append(recall)
    result = {'precission': precission_set,'recall': recall_set}
    df = pd.DataFrame(result)
    df.to_csv("DBSCAN_cluster_result.csv")

# method2: using Kmeans cluster method   
def run_kmeans_method(taxi_data):
    precission_set = []
    recall_set = []
    for t in range(24):
         hour_taxi_data = taxi_data[taxi_data['hour']==t]
         passenger_location =  hour_taxi_data[hour_taxi_data['is_passenger_location']==1]
         print(passenger_location.shape)
         estimator = KMeans(n_clusters=1500)# create cluster
         estimator.fit(hour_taxi_data[['lon','lat']])
         centroids = estimator.cluster_centers_ #cluster centroid
         recommend = pd.DataFrame(centroids,columns=['lon','lat'])
         recommend = latitude_longitude_to_go(recommend)
         passenger_location = passenger_location[['lon','lat']]
         passenger_location = latitude_longitude_to_go(passenger_location)
         precission,recall = calculate_precision_recall(recommend,passenger_location)
         precission_set.append(precission)
         recall_set.append(recall)
    result = {'precission': precission_set,'recall': recall_set}
    df = pd.DataFrame(result)
    df.to_csv("kmeans_cluster_result.csv")

# method3: using hot_region cluster method
def run_hot_region_method(taxi_data):
    precission_set = []
    recall_set = []
    for t in range(24):
        car_hirer_data = taxi_data[taxi_data['hour']==t]
        passenger_location =  car_hirer_data[car_hirer_data['is_passenger_location']==1]
        db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(car_hirer_data[['lon','lat']])
        car_hirer_data['block_id'] = db.labels_
        # calculate recommendation rate
        recommend = car_hirer_data.groupby(['block_id'],as_index=False)['is_passenger_location'].agg({'car_nums':'count','passenger':'sum'})
        recommend['recommend_val'] = recommend['passenger']/recommend['car_nums']
        # merge the "recommend_val" attribute for car_hirer_data 
        car_hirer_data = pd.merge(car_hirer_data,recommend[['block_id','recommend_val']],how='left',on=['block_id'])
        print(car_hirer_data.head(10))
        #  recommended passenger aboard location which has the recommend_val above 0.05 for driver.
        recommend_locations = car_hirer_data[car_hirer_data['recommend_val'] > 0.05]
        recommend_locations = recommend_locations[['lon','lat']]
        print(recommend_locations.shape)
        recommend_locations = latitude_longitude_to_go(recommend_locations)
        passenger_location = passenger_location[['lon','lat']]
        passenger_location = latitude_longitude_to_go(passenger_location)
        precission,recall = calculate_precision_recall(recommend_locations,passenger_location)
        precission_set.append(precission)
        recall_set.append(recall)
    result = {'precission': precission_set,'recall': recall_set}
    df = pd.DataFrame(result)
    df.to_csv("hot_region_result.csv")

    
def plot_precission_recall():
    ds_result = pd.read_csv('DBSCAN_cluster_result.csv')
    hot_result = pd.read_csv('hot_region_result.csv')
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(24), ds_result['precission'].values, marker='o')
    plt.plot(range(24), hot_result['precission'].values, marker='o')
    plt.xlabel(u"hours") # x lable
    plt.ylabel(u"precision") # y label
    plt.title("The precision of the recommendation algorithm ") # title
    plt.xticks(range(24))
    plt.xlim(0, 24) 
    plt.legend(['DBSCAN clustering recommendation algorithm',' Hotspot region recommendation algorithm '])
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(24), ds_result['recall'].values, marker='o')
    plt.plot(range(24), hot_result['recall'].values, marker='o')
    plt.xlabel(u"hours") # x lable
    plt.ylabel(u"recall") # y label
    plt.title("The recall of the recommendation algorithm ") # title
    plt.xticks(range(24))
    plt.xlim(0, 24) 
    plt.legend(['DBSCAN clustering recommendation algorithm',' Hotspot region recommendation algorithm '])
    

if __name__ == '__main__':
    '''
        preprocesing data
    '''
    # I have merge the files into one csv file--'taxi_all.csv' by read_files()
   
    taxi_data = pd.read_csv('taxi_all.csv')
    taxi_data.columns = ['id','time','lon','lat','speed','angle','is_vacant']
    print(taxi_data.info())
    taxi_data['time'] = pd.to_datetime(taxi_data['time'])
    taxi_data['hour']= taxi_data['time'].dt.hour
    #Determine if the location is a passenger point
    taxi_data = is_passenger_location(taxi_data)
    
    #  run three method
    run_hot_region_method(taxi_data)
    run_dbscan_method(taxi_data)
    #run_kmeans_method(taxi_data) #very slow
    
    # plot recommend method precission and recall
    plot_precission_recall()
    
    #Plot the number of passengers at various times of the day
    plot_hour_passenger(taxi_data) 
    
    # Visualized the clustering result at a 9:00
    plot_clustering_result(taxi_data,9)
    
    # Visualized the clustering result at a 18:00
    plot_clustering_result(taxi_data,18)
    
    # using recommendation model at 17:00
    recommendation_model(taxi_data,19)
    

    

    
        
       


   

    
    
    
    

    
    
    
