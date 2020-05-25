'''
Author: Tejaswi Paruchuri
ASU ID: 1213268054
Project2: Unsupervised learning (Kmeans)
'''

import scipy.io
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-plot_scatters", metavar="No", dest='plot_scatters', default='No', help="give Yes to plot scatter graphs else No", type=str)

class Kmeans:
    def __init__(self,plot_scatters,dataset):
        self.dataset=dataset
        self.plot_scatters=plot_scatters
        
        objective_function_strategy1,scatter_plot_dictionary1=self.KmeansStrategy1(self.dataset) #get the objective function value for k in range of 2 to 10 using strategy 1
        objective_function_strategy2,scatter_plot_dictionary2=self.KmeansStrategy2(self.dataset)  #get the objective function value for k in range of 2 to 10 using strategy 2
        print('\n\nK means strategy 1 objective function')
        for i in objective_function_strategy1:
            print(i) 
        print('\n\nK means strategy 2 objective function')
        for i in objective_function_strategy2:
            print(i)
        
        #plot objective function value vs k graph for strategy 1&2
        plt.figure(1)
        plt.plot([i for i in range(2,11)],objective_function_strategy1,marker='*')
        plt.xlabel('k (number of centroids)')
        plt.ylabel('objective_function value')
        plt.title('Kmeans Strategy 1')
        
        plt.figure(2)
        plt.plot(range(2,11),objective_function_strategy2,marker='*')
        plt.xlabel('k (number of centroids)')
        plt.ylabel('objective_function value')
        plt.title('Kmeans Strategy 2')
        
        #in case if plot scatters value is passed as yes during execution the below logic will plot scatter graphs
        if(plot_scatters.lower()=='yes'):
            for i in range(2,11):
                data_centroid=scatter_plot_dictionary1[i][0]
                centroids=scatter_plot_dictionary1[i][1]
                plt.figure(i+1)
                plt.xlabel(str(i)+' (number of centroids)')
                plt.ylabel('clusters')
                plt.title('Kmeans Strategy 1')
                for j in range(i):
                    centroid_dataSamples=np.array(self.getCentroid_datasamples(centroids,self.dataset,data_centroid,j))
                    plt.scatter(centroid_dataSamples[:, 0], centroid_dataSamples[:, 1])
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='*',s=200,c='Black')
                
            for i in range(2,11):
                data_centroid=scatter_plot_dictionary1[i][0]
                centroids=scatter_plot_dictionary1[i][1]
                plt.figure(i+10)
                plt.xlabel(str(i)+' (number of centroids)')
                plt.ylabel('clusters')
                plt.title('Kmeans Strategy 2')
                for j in range(i):
                    centroid_dataSamples=np.array(self.getCentroid_datasamples(centroids,self.dataset,data_centroid,j))
                    plt.scatter(centroid_dataSamples[:, 0], centroid_dataSamples[:, 1])
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='*',s=200,c='Black')
            
        plt.show()
        
    def KmeansStrategy1(self,dataset):
        objective_function=np.array([0] * 9, dtype = np.float64)
        scatter_plot_dictionary={}
        for k in range(2,11):
            centroids_random=random.sample(list(dataset),k) #get k random centroids initially
            centroids=np.array(centroids_random,dtype = np.float64)
            centroids_previous=np.zeros(centroids.shape)
            error=self.distance(centroids,centroids_previous,None)
            while error>0: # will update centroids till error is zero
                data_centroid=self.getCentroidNumber(centroids,dataset) #assign samples to their corresponding centroids
                centroids_previous=copy.deepcopy(centroids)
                centroids=self.updateCentroids(centroids,dataset,data_centroid) #update centroids based on the newly assigned samples
                error=self.distance(centroids,centroids_previous,None) #calculate the error between previous and updated centroids
            objective_function[k-2]=self.getObjectiveValue(centroids,dataset,data_centroid) #calculate the objective function value with the final centroids
            scatter_plot_dictionary[k]=[copy.deepcopy(data_centroid),copy.deepcopy(centroids)]
            
        return objective_function,scatter_plot_dictionary
    
    def KmeansStrategy2(self,dataset):
        '''This function will implement k-means algorithm using strategy 2'''
        objective_function=np.array([0] * 9, dtype = np.float64)
        scatter_plot_dictionary={}
        for k in range(2,11):
            centroids=self.getFarthestcentroids(dataset,k) #get k centroids based on strategy 2
            centroids_previous=np.zeros(centroids.shape)
            error=self.distance(centroids,centroids_previous,None)
            while error>0:  # will update centroids till error is zero
                data_centroid=self.getCentroidNumber(centroids,dataset) #assign samples to their corresponding centroids
                centroids_previous=copy.deepcopy(centroids)
                centroids=self.updateCentroids(centroids,dataset,data_centroid) #update centroids based on the newly assigned samples
                error=self.distance(centroids,centroids_previous,None) #calculate the error between previous and updated centroids
            objective_function[k-2]=self.getObjectiveValue(centroids,dataset,data_centroid)  #calculate the objective function value with the final centroids
            scatter_plot_dictionary[k]=[copy.deepcopy(data_centroid),copy.deepcopy(centroids)]
            
        return objective_function,scatter_plot_dictionary
    
    def getFarthestcentroids(self,dataset,k):
        '''this function will take 1st centroid randomly and remaining k-1 centroids that are farthest from the previously selected centroids'''
        centroids=np.array(random.sample(list(dataset), 1),dtype = np.float64)
        for x in range(2,k+1):
            max_avg_distance = 0
            x_coord = 0
            y_coord = 0
            for j in range(len(dataset)):
                if dataset[j] not in centroids:
                    distances = self.distance(dataset[j], centroids)
                    avg_distance = np.average(distances)
                    if avg_distance > max_avg_distance:
                        max_avg_distance = avg_distance
                        x_coord = dataset[j][0]
                        y_coord = dataset[j][1]
            new_point = [x_coord,y_coord]
            centroids = np.vstack([centroids, new_point])
        return centroids
        
    def distance(self,sample,centroids,axis=1):
        '''This funciton will give the eucleadian distance between points'''
        return np.linalg.norm(sample-centroids,axis=axis)
        
    def getCentroidNumber(self,centroids,dataset):
        '''This function will return the the centroid number to which particular data sample belongs to'''
        data_centroid=np.zeros(len(dataset))
        for sample_number in range(len(dataset)):
            distances=self.distance(dataset[sample_number],centroids)
            centroid_number=np.argmin(distances)
            data_centroid[sample_number]=centroid_number
        return data_centroid
    
    def updateCentroids(self,centroids,dataset,data_centroid):
        '''This function will update the centroids by calculate the average of all the data points belonging to particular cluster '''
        for i in range(len(centroids)):
            centroid_dataSamples=self.getCentroid_datasamples(centroids,dataset,data_centroid,i) 
            centroids[i]=np.mean(centroid_dataSamples,axis=0) 
        return centroids
    
    def getCentroid_datasamples(self,centroids,dataset,data_centroid,i): 
        '''This function will return the set of samples that belong to a particular centroid '''
        return [dataset[j] for j in range(len(dataset)) if data_centroid[j]==i]
    
    def getObjectiveValue(self,centroids,dataset,data_centroid):  
        '''This function will calculate the objective value function for the clustered datasets'''
        objective_function=0
        for i in range(len(centroids)):
            centroid_dataSamples=self.getCentroid_datasamples(centroids,dataset,data_centroid,i) 
            distances=self.distance(centroids[i],centroid_dataSamples)
            objective_function+=np.sum(np.square(distances)) 
        return objective_function
        
if __name__ == "__main__":
    # This is the first function that will execute on executing the program
    args = parser.parse_args()
    Numpyfile= scipy.io.loadmat('AllSamples.mat')  #loading the samples 
    Kmeans(args.plot_scatters,np.array(Numpyfile['AllSamples'])) #calling Kmeans funciton 