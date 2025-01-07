import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# Check if the 'stopwords' resource is available; download it if not
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


#text preprocessing
ps = PorterStemmer()

def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line) #leave only characters from a to z
    review = review.lower() #lower the text
    review = review.split() #turn string into list of words
    #apply Stemming
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #delete stop words like I, and ,OR   review = ' '.join(review)
    #trun list into sentences
    return " ".join(review)