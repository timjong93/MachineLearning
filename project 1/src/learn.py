def readTrainingSet(filename): 
    # Open a file and get the number of lines
    fr = open(filename) 
    numberOfLines = len(fr.readlines()) 
  
    # Make a result matrix with NOL rows and 3 columns
    returnMat = zeros((numberOfLines,3)) 
    classLabelVector = [] 
  
    fr = open(filename) 
    index = 0
    # Read each line and split by tabs.
    for line in fr.readlines(): 
        listFromLine = line.strip().split('\t') 
        # Use the columns 0, 1 and 2 for values (put them in the matrix)
        returnMat[index,:] = listFromLine[0:3] 
        # Use negative indexing (to begin at the end of the array) and the value to an int (1, 2 or 3)
        classLabelVector.append(int(listFromLine[-1])) 
        index += 1
    return returnMat,classLabelVector 
        
""" 
 Show two properties of the data in a scatterplot 
""" 
def showScatterPlot(data, labels, idx1, idx2): 
    import matplotlib.pyplot as plt 
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    # X-axis data, Y-axis data, Size for each sample, Color for each sample 
    ax.scatter(data[:,idx1], data[:,idx2], 15.0*array(labels), 15.0*array(labels)) 
    plt.show()
    
"""
Normalizes features to the 0..1 scale 
Very useful (important!) when the ranges of different attributes are not the same!
"""
def autoNorm(dataSet): 
    # Define mins and maxs vectors for each column (therefore the parameter)
    minVals = dataSet.min(0) 
    maxVals = dataSet.max(0) 
    # Define the ranges per column
    ranges = maxVals - minVals 
    # Create a result matrix and normalize values between 0 and 1.
    normDataSet = zeros(shape(dataSet)) 
    # Get the number of samples (rows)
    m = dataSet.shape[0] 
    # Subtract min value from each feature
    normDataSet = dataSet - tile(minVals, (m,1))
    # Divide by the range
    normDataSet = normDataSet / tile(ranges, (m,1)) 
    return normDataSet, ranges, minVals 
    
"""
KNN algorithm implementation.
"""
def knnClassify(testSample, dataSet, labels, k): 
    # Size of the featureset
    dataSetSize = dataSet.shape[0] 
  
    # Calculate the difference between the original matrix and the dataset (for each item)
    diffMat = tile(testSample, (dataSetSize,1)) - dataSet 
  
    # Square each cell
    sqDiffMat = diffMat**2
  
    # Sum all the distances per dataitem (row) and take the square root
    sqDistances = sqDiffMat.sum(axis=1) 
    distances = sqDistances**0.5
  
    # Sort the distances
    sortedDistIndicies = distances.argsort() 
    # Take the first k elements and count for each label the number of occurences
    classCount={} 
    for i in range(k): 
        voteIlabel = labels[sortedDistIndicies[i]] 
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
  
    # Sort the class count (by the number of occurences)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) 
  
    # Return the most occuring label
    return sortedClassCount[0][0]

"""
 Test a subset of the data on the remaining data (test set vs training set)
"""
def testClassifier(features, labels, k):
    # Percentage of data to hold back from the data set (used as test set)
    hoRatio = 0.10
    # Size of the data set
    m = features.shape[0]
    # Calculate the size of the test set
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # Execute the tests
    for i in range(numTestVecs):
        classifierResult = knnClassify(features[i,:], features[numTestVecs:m,:],  
              labels[numTestVecs:m], k)
   
        if (classifierResult != labels[i]): errorCount += 1.0
    print("the total error rate is: %f", (errorCount/float(numTestVecs)))    


