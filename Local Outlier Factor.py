def LocalOutlierFactor(X, k):
    '''
    input:
        X = dataset : Pandas dataframe
        k = number of nearest neighbors
    this function would use inbuilt kD tree to get distance between neighors
    '''
    # KDtree is a space-partitioning data structure for organizing points in a k-dimensional space. 
    BT = KDTree(X, leaf_size=k, p=2)

    distance, index = BT.query(X, k)
    distance, index = distance[:, 1:], index[:, 1:] 
    radius = distance[:, -1]

    #Calculate Local Reachability Distance.
    LRD = np.mean(np.maximum(distance, radius[index]), axis=1)
    r = 1. / np.array(LRD)

    #Calculate outlier score.
    outlier_score = np.sum(r[index], axis=1) / np.array(r, dtype=np.float16)
    outlier_score *= 1. / k

    return outlier_score


def Lof_prdict(outlier_score, outlier_threshold = 1.2):
    '''
    input:
        outlier_score = return values of above function
        outlier_threshold 
    this function would tell which instance is outlier
    '''
    # target class
    y = []

    # check whether given instance in outlier or not based on their outlier_score and outlier_threshold
    for i, score in enumerate(outlier_score):
        if score > outlier_threshold:
            y.append(1)
        else :
            y.append(0)

    return y