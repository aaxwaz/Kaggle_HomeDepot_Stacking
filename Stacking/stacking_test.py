from utils import * 
from stacking_params import *
import gc
#########################################################################################
######################          Data Preparation          ###############################
#########################################################################################

if DEBUG_MODE:
    print("Warning: Debug Mode!!")

df = pd.read_csv('df_all3.2_letter_ngrams_complete_2to7.csv', encoding="ISO-8859-1")

print("Data read in.")

df_train = pd.read_csv('./origin_data/train.csv', encoding="ISO-8859-1") #update here
num_train = df_train.shape[0]
train_vec = pd.read_csv('word2vec_train_50dim.csv').values
test_vec = pd.read_csv('word2vec_test_50dim.csv').values

df_train = df.iloc[:num_train]
df_test = df.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
del df_train
del df_test

##### prepare array #####
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals())
                        ],
                    transformer_weights = {
                        'cst': 1.0, 
                        }
                ))])
df_all_transformed = clf.fit_transform(df)
X_train = df_all_transformed[:num_train, :]
X_test = df_all_transformed[num_train:, :]
del df_all_transformed
del df

#### Combine TFIDF features #### 
with open('./TFIDF_Feat/selected_features_in_tfidf.csv', 'r') as f:
    features = f.read().splitlines()
features = [temp for temp in features if 'test' not in temp]

for feature in features:
    with open(feature, 'rb') as f:
        temp_train = pickle.load(f)
    with open(feature.replace('train', 'test'), 'rb') as f:
        temp_test = pickle.load(f)
    X_train = np.concatenate((X_train, temp_train), axis = 1)
    X_test = np.concatenate((X_test, temp_test), axis = 1)

del temp_train
del temp_test

#### Combine CO_TFIDF features #### 
with open('./TFIDF_Feat/selected_features_in_co_tfidf.csv', 'r') as f:
    features = f.read().splitlines()
features = [temp for temp in features if 'test' not in temp]

for feature in features:
    with open(feature, 'rb') as f:
        temp_train = pickle.load(f)
    with open(feature.replace('train', 'test'), 'rb') as f:
        temp_test = pickle.load(f)
    X_train = np.concatenate((X_train, temp_train), axis = 1)
    X_test = np.concatenate((X_test, temp_test), axis = 1)
    
del temp_train
del temp_test

#### Combine doc2vec features ####
X_train = np.concatenate((X_train, train_vec), axis = 1)
X_test = np.concatenate((X_test, test_vec), axis = 1) 

X_train[np.isnan(X_train)] = 0.0 # tree models data sets ready
X_test[np.isnan(X_test)] = 0.0

del train_vec
del test_vec

##### delete const cols #####
cols_to_del = []
for c in range(X_train.shape[1]):
    if len(set(X_train[:,c])) == 1:
        print("Deleting col: {0}".format(c))
        cols_to_del.append(c)
X_train = np.delete(X_train, cols_to_del, 1)
X_test = np.delete(X_test, cols_to_del, 1)

X_train_dim = X_train.shape[1]

## prepare other data sets ##
X_all = np.concatenate((X_train, X_test), axis = 0)
## for SGD
#X_all_min = np.apply_along_axis(lambda x: (x-np.min(x))/(np.max(x) - np.min(x)), 0, X_all) # max_min standardization
#X_train_min = X_all_min[:num_train,:] # all other models data sets ready
#X_test_min = X_all_min[num_train:,:]
#del X_all_min
## for knn & ada & 100-xgb
#X_all = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x), 0, X_all) # z-score standardization
#svd = SVD(n_components = 100)
#X_all_small = svd.fit_transform(X_all)
#X_train_100 = X_all_small[:num_train,:] # knn & ada & 100-xgb data sets ready
#X_test_100 = X_all_small[num_train:,:]
#del X_all_small
## for all other models 
#X_all = np.apply_along_axis(lambda x: (x-np.min(x))/(np.max(x) - np.min(x)), 0, X_all) # max_min standardization
#X_train_std_min = X_all[:num_train,:] # all other models data sets ready
#X_test_std_min = X_all[num_train:,:]

gc.collect()


########################### Data Sets ########################### 
## X_train
## X_test
## X_train_min
## X_test_min
## X_train_100
## X_test_100
## X_train_std_min
## X_test_std_min
    
final_test_X = 0

print("Start generating stacking for test data ...")

# ET 
final_preds = 0
for b in range(et_param['BAG_SIZE']):
    etr = ExtraTreesRegressor(n_estimators=et_param['n_estimators'],
                              max_features=et_param['max_features'],
                              max_depth=et_param['max_depth'],
                              verbose=et_param['verbose'],
                              min_samples_split = et_param['min_samples_split'],
                              n_jobs=et_param['n_jobs'])
    etr.fit(X_train, y_train)
    preds = etr.predict(X_test)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = final_preds     
print("ET Done")   

gc.collect()

# RF
final_preds = 0
for b in range(rf_param['BAG_SIZE']):
    rfr = RandomForestRegressor(n_estimators=rf_param['n_estimators'],
                                min_samples_split=rf_param['min_samples_split'],
                                max_features=rf_param['max_features'],
                                verbose=rf_param['verbose'],
                                max_depth = rf_param['max_depth'],
                                n_jobs=rf_param['n_jobs'])
    rfr.fit(X_train, y_train)
    preds = rfr.predict(X_test)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all  
print("RF Done")   

gc.collect()

# XGB Tree
final_preds = 0
dtrain_fold = xgb.DMatrix(X_train, label=y_train)
dtest_fold = xgb.DMatrix(X_test)
del X_train
del X_test
gc.collect()

for b in range(xgb_tree_param['BAG_SIZE']):
    param = {
    'task': xgb_tree_param['task'],
    'booster': xgb_tree_param['booster'],
    'objective': xgb_tree_param['objective'],
    'eta': xgb_tree_param['eta'],
    'gamma': xgb_tree_param['gamma'],
    'min_child_weight': xgb_tree_param['min_child_weight'], 
    'max_depth': xgb_tree_param['max_depth'], 
    'subsample': xgb_tree_param['subsample'], 
    'num_round': xgb_tree_param['num_round'],
    'colsample_bytree': xgb_tree_param['colsample_bytree'], 
    'silent': xgb_tree_param['silent'],
    'seed': b+2016,  
    "eval_metric" : xgb_tree_param["eval_metric"],
    'verbose' : xgb_tree_param['verbose'],
    'lambda': xgb_tree_param['lambda'],
    'alpha': xgb_tree_param['alpha'],
    }
    bst = xgb.train(param, dtrain_fold, param['num_round'])
    preds = bst.predict(dtest_fold)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all  
print("XGB Done")   

gc.collect()

# XGB linear
final_preds = 0
for b in range(xgb_linear_param['BAG_SIZE']):
    param = {
    'task': xgb_linear_param['task'],
    'booster': xgb_linear_param['booster'],
    'objective': xgb_linear_param['objective'],
    'eta': xgb_linear_param['eta'],
    'lambda' : xgb_linear_param['lambda'],
    'alpha' : xgb_linear_param['alpha'],
    'lambda_bias' : xgb_linear_param['lambda_bias'],
    'num_round': xgb_linear_param['num_round'],
    'silent': xgb_linear_param['silent'],
    'seed': b+2016,  
    "eval_metric" : xgb_linear_param["eval_metric"],
    'verbose' : xgb_linear_param['verbose'],
    }
    bsl = xgb.train(param, dtrain_fold, param['num_round'])
    preds = bsl.predict(dtest_fold)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("XGB Linear Done")   

gc.collect()

# SGD
X_all_std = np.apply_along_axis(lambda x: (x-np.min(x))/(np.max(x) - np.min(x)), 0, X_all) # max_min standardization
X_train_min = X_all_std[:num_train,:] # all other models data sets ready
X_test_min = X_all_std[num_train:,:]
del X_all_std

final_preds = 0
for b in range(sgd_param['BAG_SIZE']):
    sgd = SGD(loss=sgd_param['loss'], 
              penalty=sgd_param['penalty'], 
              alpha=sgd_param['alpha'],
              l1_ratio=sgd_param['l1_ratio'],
              fit_intercept=sgd_param['fit_intercept'],
              n_iter=sgd_param['n_iter'],
              verbose=sgd_param['verbose'])
    sgd.fit(X_train_min, y_train)
    preds = sgd.predict(X_test_min)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("SGD Done")   

del X_train_min
del X_test_min
gc.collect()
   
# lasso
X_all_std_min = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x), 0, X_all) # z-score standardization
X_all_std_min = np.apply_along_axis(lambda x: (x-np.min(x))/(np.max(x) - np.min(x)), 0, X_all_std_min) # max_min standardization
X_train_std_min = X_all_std_min[:num_train,:] # all other models data sets ready
X_test_std_min = X_all_std_min[num_train:,:]
del X_all_std_min
gc.collect()
                                                     
final_preds = 0
for b in range(lasso_param['BAG_SIZE']):
    clf = linear_model.Lasso(alpha=lasso_param['alpha'])
    clf.fit(X_train_std_min, y_train)
    preds = clf.predict(X_test_std_min)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("Lasso Done")   

gc.collect()

# ridge                                                     
final_preds = 0
for b in range(ridge_param['BAG_SIZE']):
    clf = linear_model.Ridge(alpha=ridge_param['alpha'])
    clf.fit(X_train_std_min, y_train)
    preds = clf.predict(X_test_std_min)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("Ridge Done")   

gc.collect()

# linear svr                                                     
final_preds = 0
for b in range(linearSVR_param['BAG_SIZE']):
    svr = LinearSVR(C=linearSVR_param['C'], 
                    loss=linearSVR_param['loss'], 
                    epsilon=linearSVR_param['epsilon'], 
                    tol=linearSVR_param['tol'],
                    dual=linearSVR_param['dual'],
                    max_iter=linearSVR_param['max_iter'],
                    verbose=linearSVR_param['verbose'])
    svr.fit(X_train_std_min, y_train)
    preds = svr.predict(X_test_std_min)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("linear SVR Done")    
 
gc.collect()
 
# bayes
final_preds = 0
for b in range(bayes_param['BAG_SIZE']):
    clf = linear_model.BayesianRidge()
    clf.fit(X_train_std_min, y_train)
    preds = clf.predict(X_test_std_min)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("Bayes Done")   

gc.collect()

# keras
final_preds = 0
for b in range(keras_param['BAG_SIZE']):
    INPUT_DIM = X_train_dim       
    model = Sequential()
    ## first hidden layer
    model.add(Dense(keras_param['HIDDEN_LAYER_1'], input_dim = INPUT_DIM, W_regularizer=l2(keras_param['L2_LAMBDA'])))
    model.add(aa.PReLU()) # Activation('linear')
    model.add(Dropout(keras_param['HIDDEN_DROPOUT_1']))
    ## output layer 
    model.add(Dense(1, W_regularizer=l2(keras_param['L2_LAMBDA']))) 
    model.add(aa.PReLU())
    model.compile(loss='mse', optimizer='adam') # mean_square_error
    model.fit(X_train_std_min, y_train, batch_size=keras_param['BATCH_SIZE'], nb_epoch=keras_param['EPOCHS'], verbose=0)
    preds = model.predict(X_test_std_min, verbose=0).flatten()
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("Keras Done")   

del X_train_std_min
del X_test_std_min 
gc.collect()

# ada
X_all = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x), 0, X_all) # z-score standardization
svd = SVD(n_components = 100)
X_all_small = svd.fit_transform(X_all)
X_train_100 = X_all_small[:num_train,:] # knn & ada & 100-xgb data sets ready
X_test_100 = X_all_small[num_train:,:]
del X_all_small
del X_all
del svd
gc.collect()

final_preds = 0
for b in range(ada_param['BAG_SIZE']):
    ada = AdaBoostRegressor(n_estimators=ada_param['n_estimators'],
                            learning_rate=ada_param['learning_rate'],
                            loss=ada_param['loss'])
    ada.fit(X_train_100, y_train)
    preds = ada.predict(X_test_100)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all 
print("ada Done")   

gc.collect()

# xgb-100
final_preds = 0
dtrain_fold = xgb.DMatrix(X_train_100, label=y_train)
dtest_fold = xgb.DMatrix(X_test_100)
for b in range(xgb_tree_param['BAG_SIZE']):
    param = {
    'task': xgb_tree_param['task'],
    'booster': xgb_tree_param['booster'],
    'objective': xgb_tree_param['objective'],
    'eta': xgb_tree_param['eta'],
    'num_round': xgb_tree_param['num_round'],
    'silent': xgb_tree_param['silent'],
    'seed': b+2016,  
    "eval_metric" : xgb_tree_param["eval_metric"],
    'verbose' : xgb_tree_param['verbose'],
    'lambda': xgb_tree_param['lambda'],
    'alpha': xgb_tree_param['alpha'],
    }
    bst = xgb.train(param, dtrain_fold, param['num_round'])
    preds = bst.predict(dtest_fold)
    final_preds = final_preds + preds
final_preds = final_preds/(b+1)
final_preds[final_preds>3.0] = 3.0
final_preds[final_preds<1.0] = 1.0
final_preds = final_preds.reshape(len(final_preds), 1)
final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all  
print("XGB-100 Done")   

gc.collect()

# knn 
for n in [2**i for i in range(10)]:
    print(n)
    final_preds = 0
    knn = KNN(n_neighbors=n,
              weights=knn_param['weights'],
              algorithm=knn_param['algorithm'],
              leaf_size=knn_param['leaf_size'],
              n_jobs=knn_param['n_jobs'])        
    knn.fit(X_train_100, y_train)
    final_preds = knn.predict(X_test_100)
    final_preds[final_preds>3.0] = 3.0
    final_preds[final_preds<1.0] = 1.0
    final_preds = final_preds.reshape(len(final_preds), 1)
    final_test_X = np.concatenate((final_test_X, final_preds), axis = 1)   # add this preds to this fold all  
    print("KNN %s done!" % n)
print("KNN Done")   

final_test_X = np.concatenate((final_test_X, last_col), axis = 1)

del X_train_100
del X_test_100
gc.collect()

final_test_X = pd.DataFrame(final_test_X, columns = ['ET', 'RF', 'XGB', 'XGB_linear', 'sgd', 'lasso', 'ridge', 'linear_svr', 'bayes', 'keras', 'ada', 'xgb_100', 'knn_1','knn_2', 'knn_4', 'knn_8', 'knn_16', 'knn_32', 'knn_64', 'knn_128', 'knn_256', 'knn_512', 'knn_1024'])
final_test_X.to_csv('stacking_test_2.0_4.0data.csv', index = False) 

print("all done.")











    
