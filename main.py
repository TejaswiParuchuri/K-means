import scipy.io
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-plot_scatters", metavar="Yes", dest='plot_scatters', default='Yes', help="give Yes to plot scatter else No", type=str)

class Kmeans:
    def __init__(self,plot_scatters,dataset):
        self.dataset=dataset
        self.plot_scatters=plot_scatters
        
        objective_function_strategy1,scatter_plot_dictionary1=self.KmeansStrategy1(self.dataset)
        objective_function_strategy2,scatter_plot_dictionary2=self.KmeansStrategy2(self.dataset)
        print('\n\nK means strategy 1 objective function')
        for i in objective_function_strategy1:
            print(i) 
        print('\n\nK means strategy 2 objective function')
        for i in objective_function_strategy2:
            print(i) 
        plt.figure(1)
        plt.plot([i for i in range(2,11)],objective_function_strategy1)
        plt.xlabel('k (number of centroids)')
        plt.ylabel('objective_function')
        plt.title('Kmeans Strategy 1')
        
        plt.figure(2)
        plt.plot(range(2,11),objective_function_strategy2)
        plt.xlabel('k (number of centroids)')
        plt.ylabel('objective_function')
        plt.title('Kmeans Strategy 2')
        
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
        centroids_random=random.sample(list(dataset),2)
        centroid_initial=np.array(centroids_random,dtype = np.float64)
        scatter_plot_dictionary={}
        for k in range(2,11):
            centroids=copy.deepcopy(centroid_initial)
            centroids_previous=np.zeros(centroids.shape)
            error=self.distance(centroids,centroids_previous,None)
            while error>0:
                data_centroid=self.getCentroidNumber(centroids,dataset)
                centroids_previous=copy.deepcopy(centroids)
                centroids=self.updateCentroids(centroids,dataset,data_centroid)
                error=self.distance(centroids,centroids_previous,None)
            objective_function[k-2]=self.getObjectiveValue(centroids,dataset,data_centroid)
            scatter_plot_dictionary[k]=[copy.deepcopy(data_centroid),copy.deepcopy(centroids)]
            centroid_initial=np.vstack([centroid_initial, self.getRandomCentroid(centroid_initial,dataset)])
        return objective_function,scatter_plot_dictionary
    
    def KmeansStrategy2(self,dataset):
        objective_function=np.array([0] * 9, dtype = np.float64)
        centroids_initial=np.array(random.sample(list(dataset), 1))
        scatter_plot_dictionary={}
        for k in range(2,11):
            centroids=self.getFarthestcentroids(centroids_initial,dataset,k)
            centroids_previous=np.zeros(centroids.shape)
            error=self.distance(centroids,centroids_previous,None)
            while error>0:
                data_centroid=self.getCentroidNumber(centroids,dataset)
                centroids_previous=copy.deepcopy(centroids)
                centroids=self.updateCentroids(centroids,dataset,data_centroid)
                error=self.distance(centroids,centroids_previous,None)
            objective_function[k-2]=self.getObjectiveValue(centroids,dataset,data_centroid)
            scatter_plot_dictionary[k]=[copy.deepcopy(data_centroid),copy.deepcopy(centroids)]
        return objective_function,scatter_plot_dictionary
    
    def getRandomCentroid(self,centroid_initial,dataset):
        new_row=np.array(random.sample(list(dataset),1))
        while new_row in centroid_initial:
            new_row=np.array(random.sample(list(dataset),1))
        return new_row
        
    def getFarthestcentroids(self,centroids_initial,dataset,k):
        centroids=centroids_initial
        for x in range(2,k+1):
            max_avg = 0
            x = 0
            y = 0
            for j in range(len(dataset)):
                if dataset[j] not in centroids:
                    distances = self.distance(dataset[j], centroids)
                    avg = np.average(distances)
                    if avg > max_avg:
                        max_avg = avg
                        x = dataset[j][0]
                        y = dataset[j][1]
            newrow = [x,y]
            centroids = np.vstack([centroids, newrow])
        return centroids
        
    def distance(self,sample,centroids,axis=1):
        return np.linalg.norm(sample-centroids,axis=axis)
        
    def getCentroidNumber(self,centroids,dataset):
        data_centroid=np.zeros(len(dataset))
        for sample_number in range(len(dataset)):
            distances=self.distance(dataset[sample_number],centroids)
            centroid_number=np.argmin(distances)
            data_centroid[sample_number]=centroid_number
        return data_centroid
    
    def updateCentroids(self,centroids,dataset,data_centroid):
        for i in range(len(centroids)):
            centroid_dataSamples=self.getCentroid_datasamples(centroids,dataset,data_centroid,i) 
            centroids[i]=np.mean(centroid_dataSamples,axis=0) 
        return centroids
    
    def getCentroid_datasamples(self,centroids,dataset,data_centroid,i):    
        return [dataset[j] for j in range(len(dataset)) if data_centroid[j]==i]
    
    def getObjectiveValue(self,centroids,dataset,data_centroid):  
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
    Kmeans(args.plot_scatters,np.array(Numpyfile['AllSamples'])) #calling ClassifyData class with input file data and parameters