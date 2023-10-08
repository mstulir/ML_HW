import numpy as np
import sys
import matplotlib.pyplot as plt
import random

#K_means.py hip1000.txt 3 defined 4a
#run K means with different conditions based on user input from command line
#input: dataFile, K, initialMeans (random, defined or test), runtype (question #)
def run_K_means(dataFile, K, initialMeans, runtype):
    #open datafile and read in as numpy matrix
    data=np.loadtxt("mouse-data/"+dataFile,delimiter=",")
    numMice=len(data[:,0])
    numGenes=len(data[0,:])
    print("numMice",numMice,"numGenes",numGenes)
    #create initial correlation matrix for question 4b
    if runtype=="4b":
        dataT=data.transpose()
        rho=np.corrcoef(dataT)
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        ax=axs[0]
        img=ax.imshow(rho)
        ax.set_title("Before Clustering, K=3")
    #if we are doing 4c, we make these plots and run the algorithm this many times
    if runtype=="4c":
        fig, axs = plt.subplots(nrows=2,ncols=5)
        fig.suptitle('10 Random initializations for K=3', fontsize=20)
        #run 10 times
        for f in range(10):
            #pick k data points randomly without replacement
            indices=random.sample(range(numGenes),K)
            means=np.zeros(shape=(numMice,K))
            for i in range(K):
                means[:,i]=data[:,indices[i]]
            clusters,objectives=Run_K_means_once(data,means, K, runtype)
            #initialize array to use
            ordered=np.empty((numMice))
            #order the data based on their class assignment
            for i in range(K):
                for j in range(len(clusters)):
                    if clusters[j]==i:
                        ordered=np.vstack((ordered,data[:,j]))
            #subplot the correlation matrix and save
            rho=np.corrcoef(ordered)
            if f<5:
                ax=axs[0,f]
            else:
                ax=axs[1,f-5]
            im=ax.imshow(rho)
            ax.set_title("Trial %i, \n Objective: %1.3f" %(f,objectives[-1]))
            ax.title.set_size(5)
            ax.axis('off')
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig('4c.png')
    #if we are doing 4d, we make these plots and run the algorithm this many times
    elif runtype=="4d":
        fig, axs = plt.subplots(nrows=2,ncols=5)
        fig.suptitle('Best Plots for K=3 through K=12', fontsize=20)
        #for k=3 to 12
        for k in range(3,13):
            #run 10 times
            objs=np.empty([10])
            for f in range(10):
                #pick k data points randomly without replacement
                indices=random.sample(range(numGenes),k)
                means=np.zeros(shape=(numMice,k))
                for i in range(k):
                    means[:,i]=data[:,indices[i]]
                if f==0:
                    clustersStack,objectives=Run_K_means_once(data,means, k, runtype)
                    clustersStack=np.array(clustersStack)
                else:
                    clusters,objectives=Run_K_means_once(data,means, k, runtype)
                    clusters=np.array(clusters)
                    clustersStack=np.dstack((clustersStack,clusters))
                objs[f]=objectives[-1]
            #get index of the lowest objective value
            ind=np.argmin(objs)
            obj=np.min(objs)
            bestClustering=clustersStack[:,:,ind]
            #initialize array to use
            ordered=np.empty((numMice))
            #order the data based on their class assignment
            for i in range(k):
                for j in range(len(bestClustering[0,:])):
                    if bestClustering[0,j]==i:
                        ordered=np.vstack((ordered,data[:,j]))
            #subplot the correlation matrix and save
            rho=np.corrcoef(ordered)
            if k<8:
                ax=axs[0,k-3]
            else:
                ax=axs[1,k-8]
            im=ax.imshow(rho)
            ax.set_title("K=%i, \n Objective: %1.3f" %(k,obj))
            ax.title.set_size(5)
            ax.axis('off')
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig('4d.png')
    #just run the algorithm once for 4a or 4b
    else:
        if initialMeans=="defined":
            #if defined, read in the defined initial means
            means=np.loadtxt("test_mean.txt")
        elif initialMeans=="test":
            means=np.loadtxt("test_madison.txt")
        elif initialMeans=="random":
            #pick k data points randomly without replacement
            indices=random.sample(range(numGenes),K)
            means=np.zeros(shape=(numMice,K))
            for i in range(K):
                means[:,i]=data[:,indices[i]]
        clusters,objectives=Run_K_means_once(data,means, K, runtype)
        #make output plot for 4b
        if runtype=="4b":
            #initialize array to use
            ordered=np.empty((numMice))
            #order the data based on their class assignment
            for i in range(K):
                for j in range(len(clusters)):
                    if clusters[j]==i:
                        ordered=np.vstack((ordered,data[:,j]))
            #subplot the correlation matrix and save
            rho2=np.corrcoef(ordered)
            ax=axs[1]
            im=ax.imshow(rho2)
            ax.set_title("After Clustering, K=3")
            fig.colorbar(im, ax=ax)
            plt.savefig('4b.png')
        #make plot for question 4a
        if runtype=="4a":
            x=np.arange(0, iters, 1, dtype=int)
            plt.scatter(x,objectives,s=20)
            plt.plot(x,objectives)
            plt.title(r'Objective Function with defined starting point')
            plt.ylabel('Objective Value')
            plt.xlabel('Iteration')
            plt.savefig('4a.png')



#run K means (for 1 input dataset and means)
#input: data, means, K, runtype (which question -- so I generate the correct plots)
#output: cluster assignments for each datapoint and list of objective function values
def Run_K_means_once(data, means, K, runtype):
    numMice=len(data[:,0])
    numGenes=len(data[0,:])
    iters=0
    run=True
    prev=[0]*numGenes
    objectives=[]

    while run:
        iters+=1
        if iters>1000:
            print("Reached max iterations!")
            break
        clusters=GenerateClusters(data,means,K)
        obj=CalcObjective(clusters, data, means, K)
        objectives.append(obj)
        means=UpdateMeans(clusters,data,K)
        if prev==clusters:
            run=False
        prev=clusters
    return clusters,objectives

#euclidian distance
#input: 2 datapoints
#output: float
def EuclidianDist(a,b):
    dist = np.linalg.norm(a - b)
    return dist

#GenerateClusters -- assigns clusters
#input: data, means, k
#output: cluster assignment for each datapoint (1xn)
def GenerateClusters(data,means,k):
    #take euclidian distance from each point to each of the means
    #assign it to the minimum
    numGenes=len(data[0,:])
    clusters=[]
    #loop through data points (numMice=208)
    for i in range(numGenes):
        distToMeans=[]
        #loop through the means (k)
        for j in range(k):
            dist=EuclidianDist(data[:,i],means[:,j])
            distToMeans.append(dist)
        #get index of the minimum entry of distToMeans
        clusters.append(np.argmin(distToMeans))
    return clusters

#update means based on cluster assignment
#input: cluster_labels, data, k
#output: updated means (kxd)
def UpdateMeans(cluster_labels, data,k):
    #initialize a 208xk matrix of zeroes
    means=np.zeros([len(data[:,0]),k])
    #initialize a counter vector
    counter=np.zeros([k])

    for i in range(len(cluster_labels)):
        idx=cluster_labels[i]
        means[:,idx]+=data[:,i]
        counter[idx]+=1

    #normalize
    for i in range(k):
        means[:,i]/=counter[i]

    return means

#Calculate the Objective function
#input: cluster assignments, data, means and k
#output: objective value as a float
def CalcObjective(clusterAssignments, data, means, k):
    # calc euclidian dist of every point from its mean and sum it
    obj=0
    #loop over cluster clusterAssignments
    for i in range(len(clusterAssignments)):
        classNum=clusterAssignments[i]
        mean=means[:,classNum]
        dist=EuclidianDist(mean, data[:,i])*EuclidianDist(mean, data[:,i])
        obj+=dist
    return obj


#main function to take command line inputs
if __name__ == "__main__":

    dataFile = sys.argv[1]
    K = sys.argv[2]
    initialMeans = sys.argv[3] #random or defined
    question=sys.argv[4]


    print ( " The input number of dataset is : % s " % ( dataFile ))
    print ( " The input number of clusters is : % s " % ( K ))
    print ( " The input initialization is : % s " % ( initialMeans))
    print ( " The question is : % s " % ( question ))

    run_K_means(dataFile, int(K),initialMeans, question)
