#######################################################################
##########                      utils 1.0                    ##########
#######################################################################

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn import cross_validation as cv
#from nltk.metrics import edit_distance
from bs4 import BeautifulSoup
import difflib
from nltk.util import bigrams
from nltk.stem.porter import *
#stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')
import re
import random
import time
import xgboost as xgb
import pickle

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_
    
class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand','query_unigram','title_unigram','description_unigram','brand_unigram','query_bigram','title_bigram','description_bigram','brand_bigram','query_trigram','title_trigram','description_trigram','brand_trigram','query_letter_unigram','title_letter_unigram','brand_letter_unigram','query_letter_twogram','title_letter_twogram','brand_letter_twogram','query_letter_threegram','title_letter_threegram','brand_letter_threegram','query_letter_fourgram','title_letter_fourgram','brand_letter_fourgram','query_letter_fivegram','title_letter_fivegram','brand_letter_fivegram','query_letter_sixgram','title_letter_sixgram','brand_letter_sixgram','query_letter_sevengram','title_letter_sevengram','brand_letter_sevengram']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)
    
RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

def str_stem1(s): 
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = re.sub(r"[ ]?[[(].+?[])]", r"", s) # remove description text in bracket (e.g. (water proof))
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("&amp;", "and")  ## newly added
        s = s.replace("&#39;", "'")  ## newly added
        s = s.replace("&quot;", "")  ## newly added
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in_ ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft_ ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb_ ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sqft_ ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cuft_ ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal_ ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz_ ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm_ ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm_ ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg_ ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt_ ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt_ ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp_ ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
 
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"

def str_stem(s): 
    if isinstance(s, str):
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s
    else:
        return "null"

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def autocorrect_query(query,df,cutoff=0.8,warning_on=True):
    """
    autocorrect a query based on the training set
    """	
    train_data = df.values[df['search_term'].values==query,:]
    s = ""
    for r in train_data:
        w = r
        s = "%s %s %s"%(s,BeautifulSoup(r[1]).get_text(" ",strip=True),BeautifulSoup(r[2]).get_text(" ",strip=True))
    s = re.findall(r'[\'\"\w]+',s.lower())
    s_bigram = [' '.join(i) for i in bigrams(s)]
    s.extend(s_bigram)
    corrected_query = []	
    for q in query.lower().split():
        if len(q)<=2:
            corrected_query.append(q)
            continue
        if bool(re.search('\d', q)): # skip if it is word with number, like 4.5in_
            corrected_query.append(q)
            continue
        corrected_word = difflib.get_close_matches(q, s,n=1,cutoff=cutoff)
        if len(corrected_word) >0:
            corrected_query.append(corrected_word[0])
        else :
            if warning_on:
                print("WARNING: cannot find matched word for '%s' -> used the original word"%(q))
            corrected_query.append(q)	
    return ' '.join(corrected_query)
    
def build_query_correction_map(print_different=True):
    # get all query
    queries = set(df_all['search_term'].values)
    correct_map = {}
    counter = 0
    if print_different:
        print("%30s \t %30s"%('original query','corrected query'))
    for q in queries:
        counter = counter + 1
        if counter % 100 == 0:
            print ("Processed: %s queries" % counter)
        corrected_q = autocorrect_query(q,df=df_all[['search_term', 'product_title', 'product_description', 'brand']],warning_on=False)
        if print_different and q != corrected_q:
            print ("%30s \t %30s"%(q,corrected_q))
        correct_map[q] = corrected_q
    return correct_map

#################################################################################################
####################             From Chen Chenglong                         ####################
#################################################################################################
def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    return val


def get_sample_indices_by_relevance(dfTrain, additional_key=None):
	""" 
		return a dict with
		key: (additional_key, median_relevance)
		val: list of sample indices
	"""
	dfTrain["sample_index"] = range(dfTrain.shape[0])
	group_key = ["relevance"]
	if additional_key != None:
		group_key.insert(0, additional_key)
	agg = dfTrain.groupby(group_key, as_index=False).apply(lambda x: list(x["sample_index"]))
	d = dict(agg)
	dfTrain = dfTrain.drop("sample_index", axis=1)
	return d


def dump_feat_name(feat_names, feat_name_file):
	"""
		save feat_names to feat_name_file
	"""
	with open(feat_name_file, "wb") as f:
	    for i,feat_name in enumerate(feat_names):
	        if feat_name.startswith("count") or feat_name.startswith("pos_of"):
	            f.write("('%s', SimpleTransform(config.count_feat_transform)),\n" % feat_name)
	        else:
	            f.write("('%s', SimpleTransform()),\n" % feat_name)
                
                
###################
### Stats Feats ###
###################

def cosine_sim(x, y):
    try:
        d = cosine_similarity(np.atleast_2d(x), np.atleast_2d(y)) 
        d = d[0][0]
    except:
        print(x)
        print(y)
        d = 0.
    return d

###########################
### For TFIDF distance  ###
###########################
def generate_dist_stats_feat_onTheFly(metric, X_train, ids_train, X_test, ids_test, indices_dict, verbose = False):
    ## stats parameters 
    quantiles_range = np.arange(0, 1.5, 0.5)
    stats_func = [ np.mean, np.std ]
    stats_feat_num = len(quantiles_range) + len(stats_func)
    n_class_relevance = 13
    
    if metric == "cosine":
        stats_feat = 0 * np.ones((len(ids_test), stats_feat_num*n_class_relevance), dtype=float)
        
    elif metric == "euclidean":
        stats_feat = -1 * np.ones((len(ids_test), stats_feat_num*n_class_relevance), dtype=float)

    for i in range(len(ids_test)):
        if verbose:
            if i % verbose == 0:
                print("Finished %s of data." %i)
        id = ids_test[i]
        sim = 1. - pairwise_distances(np.atleast_2d(X_test[i,:]), X_train, metric=metric, n_jobs=1)
        for j in range(n_class_relevance):
            key = j
            if key in indices_dict:
                inds = indices_dict[key]
                # exclude this sample itself from the list of indices
                inds = [ ind for ind in inds if id != ids_train[ind] ]
                sim_tmp = sim[0][inds]
                if len(sim_tmp) != 0:
                    feat = [ func(sim_tmp) for func in stats_func ]
                    ## quantile
                    sim_tmp = pd.Series(sim_tmp)
                    quantiles = sim_tmp.quantile(quantiles_range)
                    feat = np.hstack((feat, quantiles))
                    stats_feat[i,j*stats_feat_num:(j+1)*stats_feat_num] = feat
    return stats_feat

## generate distance stats feat 
def generate_dist_stats_feat(metric, X_train, ids_train, X_test, ids_test, indices_dict):
    ## stats parameters 
    quantiles_range = np.arange(0, 1.5, 0.5)
    stats_func = [ np.mean, np.std ]
    stats_feat_num = len(quantiles_range) + len(stats_func)
    n_class_relevance = 13
    
    if metric == "cosine":
        stats_feat = 0 * np.ones((len(ids_test), stats_feat_num*n_class_relevance), dtype=float)
        sim = 1. - pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)
    elif metric == "euclidean":
        stats_feat = -1 * np.ones((len(ids_test), stats_feat_num*n_class_relevance), dtype=float)
        sim = pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)

    print("pairwise_distances generated!")
    for i in range(len(ids_test)):
        id = ids_test[i]
        for j in range(n_class_relevance):
            key = j
            if key in indices_dict:
                inds = indices_dict[key]
                # exclude this sample itself from the list of indices
                inds = [ ind for ind in inds if id != ids_train[ind] ]
                sim_tmp = sim[i][inds]
                if len(sim_tmp) != 0:
                    feat = [ func(sim_tmp) for func in stats_func ]
                    ## quantile
                    sim_tmp = pd.Series(sim_tmp)
                    quantiles = sim_tmp.quantile(quantiles_range)
                    feat = np.hstack((feat, quantiles))
                    stats_feat[i,j*stats_feat_num:(j+1)*stats_feat_num] = feat
    return stats_feat
 
#####################
### For word dist ###
#####################
 
 
 
 

 