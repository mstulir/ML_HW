import numpy as np
from matplotlib import pyplot as plt
import time

def MultiVarNormal(x,mean,cov):
    """
    MultiVarNormal implements the PDF for a mulitvariate gaussian distribution
    (You can do one sample at a time or all at once)
    Input:
        x - An (d) numpy array
            - Alternatively (n,d)
        mean - An (d,) numpy array; the mean vector
        cov - a (d,d) numpy arry; the covariance matrix
    Output:
        prob - a scaler
            - Alternatively (n,)

    Hints:
        - Use np.linalg.pinv to invert a matrix
        - if you have a (1,1) you can extrect the scalar with .item(0) on the array
            - this will likely only apply if you compute for one example at a time
    """
    #print("cov",cov)
    ans=np.empty(0)
    for i in range(len(x)):
        #temp=np.linalg.pinv(cov)
        ans=np.append(ans,np.linalg.det(2*np.pi*cov)**(-1/2) * np.exp(-np.dot(np.dot((x[i]-mean).T,np.linalg.pinv(cov)),(x[i]-mean))/2))

    return ans

def UpdateMixProps(hidden_matrix):
    """
    Returns the new mixing proportions given a hidden matrix
    Input:
        hidden_matrix - A (n, k) numpy array
    Output:
        mix_props - A (k,) numpy array
    Hint:
        - See equation in Lecture 10 pg 42

    """
    N=hidden_matrix.shape[0]
    mix_props=np.sum(hidden_matrix,axis=0)/N
    return mix_props

def UpdateMeans(X, hidden_matrix):
    """
    Returns the new means for the gaussian distributions given the data and the hidden matrix
    Input:
        X - A (n, d) numpy array
        hidden_matrix - A (n, k) numpy array
    Output:
        new_means - A (k,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    k=hidden_matrix.shape[1]
    d=X.shape[1]
    means=np.zeros([k,d])
    for j in range(len(hidden_matrix[0])):
        num=0
        for i in range(len(X)):
            num+=hidden_matrix[i,j]*X[i]
        denom=np.sum(hidden_matrix[:,j], axis=0)
        means[j]=np.divide(num,denom)
    return means

def UpdateCovar(X, hidden_matrix_col, mean):
    """
    Returns new covariance for a single gaussian distribution given the data, hidden matrix, and distribution mean
    Input:
        X - A (n, d) numpy array
        hidden_matrix - A (n,) numpy array
        mean - A (d,) numpy array; the mean for this distribution
    Output:
        new_cov - A (d,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    mat=np.zeros([X.shape[1],X.shape[1]])
    for i in range(len(X)): #879
        mat+=np.dot(((X[i]-mean).reshape((10,1))),((X[i]-mean).reshape(10,1)).T)*hidden_matrix_col[i]
    denom=np.sum(hidden_matrix_col,axis=0)
    mat=mat/denom
    return mat


def UpdateCovars(X, hidden_matrix, means):
    """
    Returns a new covariance matrix for all distributions using the function UpdateCovar()
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
        u - A (k,d) numpy array; All means for the distributions
    Output:
        new_covs - A (k,d,d) numpy array
    Hint:
        - Use UpdateCovar() function
    """
    k=means.shape[0]
    d=means.shape[1]
    new_covs=np.zeros([k,d,d])
    for i in range(k):
        new_covs[i,:,:]=UpdateCovar(X,hidden_matrix[:,i],means[i,:])
    return new_covs


def HiddenMatrix(X, means, covs, mix_props):
    """
    Computes the hidden matrix for the data. This function should also compute the log likelihood
    Input:
        X - An (n,d) numpy array
        means - An (k,d) numpy array; the mean vector
        covs - a (k,d,d) numpy arry; the covariance matrix
        mix_props - a (k,) array; the mixing proportions
    Output:
        hidden_matrix - a (n,k) numpy array
        ll - a scalar; the log likelihood
    Hints:
        - Construct an intermediate matrix of size (n,k). This matrix can be used to calculate the loglikelihood and the hidden matrix
            - Element t_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
            P(X_i | c = j)P(c = j)
        - Each rows of the hidden matrix should sum to 1
            - Element h_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
                P(X_i | c = j)P(c = j) / (Sum_{l=1}^{k}(P(X_i | c = l)P(c = l)))
    """
    n=X.shape[0]
    k=means.shape[0]
    mat=np.zeros([n,k])
    sum=np.zeros([n])
    for i in range(k):
        col=MultiVarNormal(X,means[i],covs[i])*mix_props[i]
        sum+=col
        mat[:,i]=col
    for i in range(k):
        #print("mat", mat[1:10,i])
        #print("sum",sum[1:10])
        mat[:,i]=np.divide(mat[:,i],sum)
        #print("updated mat",mat[1:10,i])

    #log likelihood
    ll=np.sum(np.log(sum))
    return mat,ll



def GMM(X, init_means, init_covs, init_mix_props, thres):
    """
    Runs the GMM algorithm
    Input:
        X - An (n,d) numpy array
        init_means - a (k,d) numpy array; the initial means
        init_covs - a (k,d,d) numpy arry; the initial covariance matrices
        init_mix_props - a (k,) array; the initial mixing proportions
    Output:
        - clusters: and (n,) numpy array; the cluster assignment for each sample
        - ll: th elog likelihood at the stopping condition
    Hints:
        - Use all the above functions
        - Stoping condition should be when the difference between your ll from
            the current iteration and the last iteration is below your threshold
    """
    #set maximum number of iterations to perform this optimization
    max_count = 1000

    #initialize prev log likelihood to a large number
    prev_ll = 0
    lls=np.array(prev_ll)
    #set means,covs, mix props equal to init_means, init_covs, init_mix_props initially
    means = init_means
    covs = init_covs
    mix_props = init_mix_props
    count=0

    #start looping
    while count < max_count:
        count+=1
    	#TODO: compute clusters and curr_ll
        hidden_matrix, curr_ll = HiddenMatrix(X, means, covs, mix_props)

    	#compute delta_ll
        delta_ll = abs(curr_ll-prev_ll)
        prev_ll=curr_ll
        lls=np.append(lls,curr_ll)

    	#check delta_ll against threshold
        if delta_ll < thres:
            break
        means = UpdateMeans(X, hidden_matrix)
        covs=UpdateCovars(X, hidden_matrix, means)
        mix_props=UpdateMixProps(hidden_matrix)


    #TODO: assign clusters based on highest probability
    #print("probability for 1st datapoint",hidden_matrix[0])
    #print("lls", lls)
    '''
    x_axis=np.arange(0, len(lls), 1, dtype=int)
    plt.scatter(x_axis[1:],lls[1:],s=20)
    plt.plot(x_axis[1:],lls[1:])
    plt.title(r'Log likelihood Over Iterations')
    plt.ylabel('Log likelihood Value')
    plt.xlabel('Iteration')
    plt.savefig('6a.png')
    '''
    return lls


def RandomParams(gmm_data, k, n_features, epsilon=0.005, eye_covar = False):
    means = gmm_data[np.random.choice(range(gmm_data.shape[0]),k,replace=False),:]
    if not eye_covar:
        covars = []
        for _ in range(k):
            covar = (np.eye(n_features,n_features)*np.random.sample())+np.random.normal(size=(n_features,n_features))
            covars += [covar.T.dot(covar)]
    else:
        covars = np.stack([np.diag(([1]*10))]*k)

    mix_props = np.random.sample(size=(k))
    mix_props = mix_props / np.sum(mix_props)

    return means, np.stack([x+np.eye(n_features,n_features)*epsilon for x in covars]), mix_props

if __name__ == "__main__":
    k=3
    data = np.loadtxt("hip1000.txt", dtype=np.float32,delimiter=',')
    data=data[:10,].T
    test_means = np.loadtxt("test_mean.txt")
    test_means=test_means[:10,].T
    print('Data shape:',data.shape)
    print('test_means shape: ',test_means.shape)

    init_cov=np.identity(10)
    init_covs=b = np.repeat(init_cov[np.newaxis,:, :], k, axis=0)
    print("init_covs shape:",init_covs[0].shape)
    init_mix_props=np.array([0.3,0.3,0.4])
    print("init_mix_props shape:",init_mix_props.shape)

    thres=0.001


    #GMM(data, test_means, init_covs, init_mix_props, thres)
    ll_k=np.empty(0)
    while k<11:
        means, covars, mix_props = RandomParams(data, k, 10)
        print("data",data.shape)
        print("means",means.shape)
        print("covars",covars.shape)
        print("mix_props",mix_props.shape)
        lls=GMM(data, means, covars, mix_props, thres)

        #check this syntax
        print("K",k,lls[-1])
        ll_k=np.append(ll_k,lls[-1])
        k+=1
    x_axis=np.arange(3, 11, 1, dtype=int)
    plt.scatter(x_axis,ll_k,s=20)
    plt.plot(x_axis,ll_k)
    plt.title(r'Log likelihood At Convergence for k')
    plt.ylabel('Log likelihood Value')
    plt.xlabel('K')
    plt.savefig('6c.png')
