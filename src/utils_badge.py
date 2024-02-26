import numpy as np
import pdb

from scipy import stats
from sklearn.metrics import pairwise_distances

# X : embedding / K : selected num 
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    
    while len(mu) < K:
        mu_array = np.vstack(mu)  # Convert mu into 2D numpy array
        
        # mu 의 길이가 1일 수 있나? ㅇㅇㅇ argmax 값을 가져오는 거니까. 오히려 1개 일 때가 더 많겠지 
        if len(mu) == 1:

            D2 = pairwise_distances(X, mu_array).ravel().astype(float) # ravel(): flatten a multi-dimensional numpy array into a 1d-array
        else:
            newD = pairwise_distances(X, [mu_array[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDict = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist)) # creates a random variable for a custom discrete probability distribution, which can then be used to generate random samples from that distribution.
        ind = customDict.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll