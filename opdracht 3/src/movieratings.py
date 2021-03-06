#!usr/bin/python
import sys
import operator
import functools
import numpy as np
import math

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

genresList = ["unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def readTrainingSet(filename, translation): 
    # Open a file and get the number of lines
    fr = open(filename)
    tempLines = fr.readlines()
    lines = []
    for line in tempLines:
        if(not "?" in line and len(line.split(",")) == 26):
            lines.append(line)    
    numberOfLines = len(lines) 
    print(numberOfLines)
    
    # Make a result matrix with NOL rows and 3 columns
    returnMat = np.zeros((numberOfLines,26)) 
    classLabelVector = [] 
     
    index = 0
    # Read each line and split by tabs.
    for line in lines:
        listFromLine = line.strip().split(',')
        # Use the columns 0, till 14 for values (put them in the matrix)
        for i in range(0,26):
            if(i in translation):
                if(listFromLine[i] in translation[i]):
                    returnMat[index,i] = translation[i].index(listFromLine[i])
                else:
                    translation[i].append(listFromLine[i]);
                    returnMat[index,i] = len(translation[i]) - 1
            else:
                #print(listFromLine[i])
                returnMat[index,i] = float(listFromLine[i])
            
        # Use negative indexing (to begin at the end of the array) and the value to an int (1, 2 or 3)
        classLabelVector.append(int(listFromLine[-1])) 
        index += 1
    
    for k in translation.keys():
        print(str(k) + ": " + ",".join(translation[k]) + "\n-------\n");
    return returnMat,classLabelVector 
    
def readMovies(filename):
    movs = {};
    fr = open(filename)
    for line in fr.readlines():
        vals = line.split("|")
        mov = {"genres" : []};
        mov["id"] = vals[0];
        mov["name"] = vals[1];
        for i in range(5, len(vals)):
            if(int(vals[i]) == 1):
                mov["genres"].append(genresList[i-5])
        movs[mov["id"]] = mov;
    return movs

def readUsers(filename):
    users = {};
    fr = open(filename)
    for line in fr.readlines():
        vals = line.split("|");
        user = {};
        user['id'] = vals[0]
        user['age'] = vals[1]
        user['sex'] = vals[2]
        user['profession'] = vals[3]
        user['postalCode'] = vals[4]
        users[user["id"]] = user;
    return users

def readRatings(filename):
    Rs = [];
    fr = open(filename)
    for line in fr.readlines():
        vals = line.split("\t");
        r = {};
        r['user'] = vals[0]
        r['movie'] = vals[1]
        r['rating'] = float(vals[2])
        r['timestamp'] = vals[3]
        Rs.append(r)
    return Rs
    
def mostWatchedGenrePSex(movies, users, ratings):
    result = {'M':{}, 'F':{}};
    
    for r in ratings:
        sex = users[r["user"]]["sex"]
        genres = movies[r["movie"]]["genres"]
        for genre in genres:
            if(genre in result[sex]):
                result[sex][genre] += 1
            else:
                result[sex][genre] = 1
    
    sortM = sorted(result['M'].items(), key=operator.itemgetter(1), reverse= True)
    sortF = sorted(result['F'].items(), key=operator.itemgetter(1), reverse= True)
    print("Males: ", sortM[:5]);
    print("Females: ", sortF[:5]);

#  PAge     
def mostWatchedGenrePAge(movies, users, ratings):
    result = {};
    
    for r in ratings:
        age = int(users[r["user"]]["age"])
        cat = age - age % 5
        genres = movies[r["movie"]]["genres"]
        if(not cat in result):
            result[cat] = {}
        for genre in genres:
            if(genre in result[cat]):
                result[cat][genre] += 1
            else:
                result[cat][genre] = 1
                
    for key in result:
        sortM = sorted(result[key].items(), key=operator.itemgetter(1), reverse= True)
        print(str(key) + " : ", sortM[:5]);
        
def mostWatchedGenrePProf(movies, users, ratings):
    result = {};
    
    for r in ratings:
        profession = users[r["user"]]["profession"]
        
        genres = movies[r["movie"]]["genres"]
        if(not profession in result):
            result[profession] = {}
        for genre in genres:
            if(genre in result[profession]):
                result[profession][genre] += 1
            else:
                result[profession][genre] = 1
                
    for key in result:
        sortM = sorted(result[key].items(), key=operator.itemgetter(1), reverse= True)
        print(str(key) + " : ", sortM[:5]);

def mostWatchedGenrePRegion(movies, users, ratings):   
    result = {};
    
    for r in ratings:
        postalCode = users[r["user"]]["postalCode"]
        postalCode = postalCode[0]
        
        genres = movies[r["movie"]]["genres"]
        if(not postalCode in result):
            result[postalCode] = {}
        for genre in genres:
            if(genre in result[postalCode]):
                result[postalCode][genre] += 1
            else:
                result[postalCode][genre] = 1
                
    for key in result:
        sortM = sorted(result[key].items(), key=operator.itemgetter(1), reverse= True)
        print(str(key) + " : ", sortM[:5]);
        
def avgRatingPGenre(movies, ratings):
    result = {};
    for r in ratings:
        genres = movies[r["movie"]]["genres"]
        for genre in genres:
            if(not genre in result):
                result[genre] = []
            result[genre].append(int(r["rating"]))
    
    for genre in result:
        print(genre + " : ", functools.reduce(lambda x, y: x + y, result[genre]) / len(result[genre]))

def avgRatingPMov(movies, ratings):
    result = {};
    for r in ratings:
        movie = movies[r["movie"]]["name"]
        if(not movie in result):
            result[movie] = []
        result[movie].append(int(r["rating"]))
    
    for movie in result:
        result[movie] = functools.reduce(lambda x, y: x + y, result[movie]) / len(result[movie])
              
    sortM = sorted(result.items(), key=operator.itemgetter(1), reverse= True)
    for i in sortM:
        print(i);

def createFeatureSet(users, ratings, movies):
    userResults = {};
    
    for rating in ratings:
        userid = rating["user"]
        if(not userid in userResults):
            userResults[userid] = {};
        genres = movies[rating["movie"]]["genres"]
        for genre in genres:
            if(not genre in userResults[userid]):
                userResults[userid][genre] = 0
            if(rating["rating"] >= 3):
                userResults[userid][genre] += 1
    
    size = len(userResults.keys())
    returnMat = np.zeros((size,len(genresList)))
    returnUID = []
    count = 0
    for user in userResults:
        total = 0
        for genre in userResults[user]:
            total += userResults[user][genre]
        for genre in userResults[user]:
            returnMat[count][genresList.index(genre)] = userResults[user][genre] / total *100
        returnUID.append(user)
        count += 1
    return returnMat, returnUID
    
def createClusters(data):
    #Euclidean
    db = DBSCAN(eps=7, min_samples=3).fit(data)
    #Cosin
    #db = DBSCAN(eps=3, min_samples=2, metric=cosine_similarity).fit(data)
    labels = db.labels_
    return labels
    
def findSimilarUsers(users, ratings, movies, userId):
    data, userIds = createFeatureSet(users, ratings, movies)
    labels = createClusters(data)
    userCluster = labels[userId]  
    indices = [i for i, x in enumerate(labels) if x == userCluster]
    similarUsers = []
    for index in indices:
        similarUsers.append(userIds[index])
    print ('List of users: similar to user', userId)
    print(similarUsers)        
    
# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item]=1
    
    # Find the number of elements
    n=len(si)
    
    # if they are no ratings in common, return 0
    if n==0: return 0
    
    # Add up all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    
    # Sum up the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
    
    # Sum up the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    
    # Calculate Pearson score
    num=pSum-(sum1*sum2/n)
    den=math.sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0
    
    r=num/den
    
    return r
    
def groupRatingsByUser(ratings):
    result = {}
    for rating in ratings:
        if(not rating["user"] in result):
            result[rating["user"]] = {}
        result[rating["user"]][rating["movie"]] = rating["rating"]
    return result

def get5Recomendations(ratings, movies, user):
    resultsgrouped = groupRatingsByUser(ratings)
    similarity_users = {}
    for u in resultsgrouped:
        if(not u == user):
            similarity_users[u] = sim_pearson(resultsgrouped, user, u)
    
    similarity_users = sorted(similarity_users.items(), key=operator.itemgetter(1), reverse= True)
    print(similarity_users)
    
    result = []
    
    for u, score in similarity_users:
        uniques = {}
        for item in resultsgrouped[u]:
            if not item in resultsgrouped[user]: uniques[item] = resultsgrouped[u][item]
        
        for item in uniques:
            if uniques[item] == 5:
                result.append(movies[item]["name"])
                if len(result) >= 5:
                    break
                    
        if len(result) >= 5:
            break
                    
    return result
    
def main():
    movies = readMovies(sys.argv[1]);
    users = readUsers(sys.argv[2]);
    ratings = readRatings(sys.argv[3]);
    
    #mostWatchedGenrePSex(movies, users, ratings);
    print('--------------------------------------------')
    #mostWatchedGenrePAge(movies, users, ratings);
    print('--------------------------------------------')
    #mostWatchedGenrePProf(movies, users, ratings);
    print('--------------------------------------------')
    #mostWatchedGenrePRegion(movies, users, ratings);
    print('--------------------------------------------')
    #avgRatingPGenre(movies, ratings);
    print('--------------------------------------------')
    #avgRatingPMov(movies, ratings);
    print('--------------------------------------------')
    findSimilarUsers(users, ratings, movies, 5);
    print('--------------------------------------------')
    print(get5Recomendations(ratings, movies, '485'))
    print('--------------------------------------------')
    
if __name__ == "__main__":
    main()


