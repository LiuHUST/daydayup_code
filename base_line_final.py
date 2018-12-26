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
from matplotlib.colors import rgb2hex
from sklearn.cluster import DBSCAN
import os

pd.set_option('max_columns',80)

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
            poly = Polygon(shp,facecolor='w',edgecolor='k', lw=1.0, alpha=0.1)#注意设置透明度alpha，否则点会被地图覆盖
            ax1.add_patch(poly)		
    # plot the regional boundary
    map.drawmapboundary()  #边界线
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
    plt.title('recommended place at %d:00' % hour)# title
    plt.show()
    
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
    plt.title('Shanghai(31.20~31.25,121.5~121.55) at %d:00' % hour)# tile
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


if __name__ == '__main__':
    '''
        preprocesing data
    '''
    # I have merge the files into one csv file--'taxi_all.csv' by read_files()
    taxi_data = pd.read_csv('taxi_all.csv')
    taxi_data.columns = ['id','time','lon','lat','speed','angle','is_vacant']
    print(taxi_data.info())
    taxi_data['time'] = pd.to_datetime(taxi_data['time'])#转化开始时间
    taxi_data['hour']= taxi_data['time'].dt.hour
    #Determine if the location is a passenger point
    taxi_data = is_passenger_location(taxi_data)
    
    #Plot the number of passengers at various times of the day
    plot_hour_passenger(taxi_data) 
    
    # Visualized the clustering result at a 9:00

    plot_clustering_result(taxi_data,18)
    
    # Visualized the clustering result at a 18:00
    plot_clustering_result(taxi_data,18)
    
    # using recommendation model at 17:00
    recommendation_model(taxi_data,17)
    

    
    
    
