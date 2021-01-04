#modules used are: 
#pandas, scikit-learn
import pandas as pd

#Reading the data set
df1=pd.read_csv(r'dataset.csv')

df1.columns=['index', 'budget', 'genres', 'homepage', 'id' , 'keywords','original_language', 'original_title', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime' ,'spoken_languages', 'status', 'tagline','title', 'vote_average', 'vote_count', 'cast', 'crew', 'director' ]


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Remove all english stop words such as 'a','is','this',etc. from the overview of the movie
tfidf=TfidfVectorizer(stop_words='english')

#Replacing NaN vector with empty string
df1['overview']= df1['overview'].fillna('')

#TF IDF is used to measurecompare the number of times a word is occuring in a document with the number 
#of documents containing the word
tfidf_matrix= tfidf.fit_transform(df1['overview'])
#Matrix is formed of the form: term frequency- inverse term frequency
tfidf_matrix.shape

#Import linear kernel from scikit-learn 
from sklearn.metrics.pairwise import linear_kernel

#Computing cosine similarity
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)

#Mapping index of the movie to its title 
indices=pd.Series(df1.index,index=df1['title']).drop_duplicates()


#Takes input as a movie title and gives output of top 10 most similar movies based on content
def get_recom(title, cosine_sim=cosine_sim):
    
    #Retrieving the index of the movie with the given title
    idx=indices[title]
    
    #Pairwise similarity score of every movie with given movie is stored
    sim_scores=list(enumerate(cosine_sim[idx]))
    
    #Sort the list according to similarity score in descending order
    sim_scores= sorted(sim_scores, key=lambda x:x[1],reverse=True)
    
    #Stores the list of 10 recommended movies
    sim_scores=sim_scores[1:11]
    
    #Stores the indices of the movies
    movie_indices=[i[0] for i in sim_scores]

    counter=1
    for x in movie_indices:
        print(counter,".",df1.title[x],end="\n")
        counter+=1
