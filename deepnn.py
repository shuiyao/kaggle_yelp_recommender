import pandas as pd
import config, utils
import ast

import re
import numpy as np
from collections import Counter

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM

features_sparse = ['business_id', 'user_id']
features_dense = ['stars', 'review_count']

params_deepnn = {
    'category':'restaurants',
    'min_review':5, 'min_category':50,
    'weight': False,
    'scaler':'minmax', 'optimizer':"adam", 'loss':'mse',
    'batch_size':256, 'epochs':10, 'train_size':0.8,
    'deepfm__dnn_hidden_units':(128, 128),
    'deepfm__l2_reg_linear':1e-05,
    'deepfm__l2_reg_embedding':1e-05,
    'deepfm__l2_reg_dnn':0,
    'deepfm__seed':1024,
    'deepfm__dnn_dropout':0,
    'deepfm__dnn_activation':'relu'
}


# features_sparse = ['business_id', 'user_id', 'stars']

class RecommenderDeepNN:
    '''
    Recommender for Yelp dataset using the deepFM model.

    Parameters
    ----------
    category: 'restaurants', Keep only businesses of a certain category
        - Options: 'restaurants', 'automotive', 'shopping'
    min_review: 5, Keep only business with more review_count than this value
    min_category: 50, Keep only categories that apply to more than this amount of businesses
    weight:  False, Whether or not to use weights for the attribute matrix in the DeepFM
    scaler: 'minmax', Scaler for dense features
    optimizer: "adam", Optimizer for the DeepFM
    loss: 'mse', Loss function for the DeepFM
    batch_size: 256, 
    epochs: 10, 
    train_size: 0.8,
    deepfm__dnn_hidden_units: (128, 128),
    deepfm__l2_reg_linear: 1e-05,
    deepfm__l2_reg_embedding: 1e-05,
    deepfm__l2_reg_dnn: 0,
    deepfm__seed: 1024,
    deepfm__dnn_dropout: 0,
    deepfm__dnn_activation: 'relu'

    Example
    -------
    deepnn = RecommenderDeepNN(deepfm__seed=2048)
    deepnn.load_data(config.JSON_BUSINESS, config.CSV_RATINGS)
    deepnn.fit()
    deepnn.topN(260, n=5)

    deepnn = RecommenderDeepNN(scaler='standard', train_size=0.99)    
    deepnn.fit(config.JSON_BUSINESS, config.CSV_RATINGS)
    '''    

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        path_business: Path to the business.json file that contains 'attributes' and 'catogories' as dictionaries for all businesses
        path_ratings: Path to the ratings.csv file that contains 'user_id', 'business_id' and 'stars'. The review text is not needed here.
        '''
        self.path_business = ""
        self.path_ratings = ""
        self.features_sparse = features_sparse
        self.features_dense = features_dense
        
        self.params = params_deepnn
        self.params_deepfm = {}
        self.business = None
        self.data = None

        self.attr2index = {}
        self.raw_to_iid = {}
        self.iid_to_raw = {}
        self.raw_to_uid = {}
        self.uid_to_raw = {}

        # Label encoders
        self.lbe_user = None
        self.lbe_item = None

        self.model = None
        self.features_linear = []
        self.features_dnn = []
        self.model_input = {}


        self.update_params(**kwargs)

    def load_data(self, path_business, path_ratings):
        '''
        Load data and transform it to usable format.
        '''
        print("Loading data ...")
        
        self.path_business = path_business
        self.path_ratings = path_ratings
        
        df = pd.read_json(self.path_business, lines=True, encoding='utf-8')
        df_ratings = pd.read_csv(self.path_ratings)
        df_ratings.rename({'stars':'rating'}, axis=1, inplace=True)

        to_keep = config.Keywords_Categories[self.params['category']]
        keeprows = utils.filter_business_with_categories(df, to_keep)
        df = df[keeprows]

        # Map user_id and business_id encodings to integers
        self.uid_to_raw = dict(df_ratings['user_id'].drop_duplicates().reset_index()['user_id'])
        self.raw_to_uid = {k:v for v, k in self.uid_to_raw.items()}
        self.iid_to_raw = dict(df['business_id'])
        self.raw_to_iid = {k:v for v, k in self.iid_to_raw.items()}

        self.business = df[['business_id', 'name', 'stars', 'review_count', 'categories']]

        df = df[df['review_count'] > self.params['min_review']]
        df = df_ratings.join(df[['business_id', 'stars', 'review_count', 'categories']].set_index('business_id'), on='business_id', how='right')
        # Has to be "right"... otherwise there will be NaNs
        # Also, use df.set_index() because df is smaller in size

        df['user_id'] = df['user_id'].map(self.raw_to_uid)
        df['business_id'] = df['business_id'].map(self.raw_to_iid)
        
        self.lbe_user = LabelEncoder()
        self.lbe_item = LabelEncoder()
        df['user_id'] = self.lbe_user.fit_transform(df['user_id'])
        df['business_id'] = self.lbe_item.fit_transform(df['business_id'])
        # x = lbe_user.inverse_transform(df_ratings['user_id'])
        # y = lbe_item.inverse_transform(df_ratings['business_id'])
        
        if(self.params['scaler'] == 'minmax'):
            scaler = MinMaxScaler(feature_range=(0,1))
        elif(self.params['scaler'] == 'standard'):
            scaler = StandardScaler()
        df[self.features_dense] = scaler.fit_transform(df[self.features_dense])

        lbe = LabelEncoder()
        for var in self.features_sparse:
            if(var not in ['business_id', 'user_id']):
                df[var] = lbe.fit_transform(df[var])

        self.data = df
        
        del df, df_ratings

    def _compile_business_categories(self, df_business):
        '''
        Find all the categories that apply to the businesses in the DataFrame df_business
        '''
        categories = Counter()
        for line in df_business['categories']:
            if(isinstance(line, str)):
                categories.update(re.split(', ', line))
        categories = pd.DataFrame.from_dict(categories, orient='index', columns=['count'])
        return categories

    def _build_category_dict(self, drop_categories=[]):
        attrs = self._compile_business_categories(self.data)
        attrs = attrs[attrs['count'] > self.params['min_category']].sort_values(by='count', ascending=False)
        for cat in drop_categories:
            attrs.drop(cat, inplace=True)
        attrs.index.to_list()
        self.attr2index = {k:v+1 for v, k in enumerate(attrs.index.to_list())}
        del attrs

    def _category_vectorizer(self, x):
        '''
        Label encode categories of any business x into a list of indices. The mapping is given by the dictionary attr2index{catogory:index}.
        '''
        if(isinstance(x, str)):
            spt = re.split(', ', x)
            return list(map(lambda x: self.attr2index[x] if x in self.attr2index else 0, spt))
        else: return []

    def _get_category_matrix(self, df):
        attrs_matrix = [self._category_vectorizer(x) for x in df['categories'].values]
        attrs_max_len = max(np.array(list(map(len, attrs_matrix))))
        attrs_matrix = pad_sequences(attrs_matrix, maxlen=attrs_max_len, padding='post',)

        print("Matrix takes {:5.2f} MB".format(attrs_matrix.nbytes/1024./1024.))
        return attrs_matrix, attrs_max_len

    def _build_model(self):
        to_drop = config.Keywords_Categories[self.params['category']]
        self._build_category_dict(drop_categories=to_drop)
        attrs_matrix, attrs_max_len = self._get_category_matrix(self.data)
        
        vars_fixlen = [SparseFeat(var, self.data[var].nunique(),
                                  embedding_dim=4)
                       for var in self.features_sparse]
        vars_fixlen += [DenseFeat(var, 1,) for var in self.features_dense]
        vars_varlen = [VarLenSparseFeat(SparseFeat('categories',
                        vocabulary_size=len(self.attr2index) + 1,
                        embedding_dim=4),
                        maxlen=attrs_max_len, combiner='mean',
                        weight_name='attrs_weight' if self.params['weight'] else None)]

        self.features_linear = vars_fixlen + vars_varlen
        self.features_dnn = vars_fixlen + vars_varlen

        self.model = DeepFM(self.features_linear, self.features_dnn,
                            task='regression', **self.params_deepfm)
        return attrs_matrix, attrs_max_len

    def get_feature_names(self):
        return get_feature_names(self.features_linear + self.features_dnn)

    def _set_params_deepfm(self):
        for k, v in self.params.items():
            spt = k.split('__')
            if(len(spt) > 1): self.params_deepfm[spt[1]] = v

    def update_params(self, recompile=True, **kwargs):
        '''
        Update parameters for the recommender and re-compile the DeepFM model unless recompile is set to False.

        Example
        -------
        deepnn.update_params(epochs=20, deepfm__l2_reg_linear=2e-4)
        '''
        for (k, v) in kwargs.items():
            if(k in self.params):
                self.params[k] = v
            else:
                raise ValueError('{0} is not a valid parameter for RecommenderDeepNN.'.format(k))
        self._set_params_deepfm()
        if(recompile == True and self.model is not None):
            self.model = DeepFM(self.features_linear, self.features_dnn,
                                task='regression', **self.params_deepfm)

    def fit(self, path_business=None, path_ratings=None):
        if(self.data is None):
            self.load_data(path_business, path_ratings)

        model_input = self._get_model_input(self.data)

        self.model.compile(self.params['optimizer'],
                           self.params['loss'],
                           metrics=[self.params['loss']],)
        self.model.fit(model_input, self.data['rating'].values,
                       batch_size=self.params['batch_size'],
                       epochs=self.params['epochs'], 
                       validation_split=1-self.params['train_size'],
                       verbose=2)

    def _get_model_input(self, df):
        if(self.model is None):
            attrs_matrix, attrs_max_len = self._build_model()
        else:
            attrs_matrix, attrs_max_len = self._get_category_matrix(df)

        features = self.get_feature_names()

        model_input = {name: df[name] for name in features}
        model_input['categories'] = attrs_matrix
        if(self.params['weight']):
            model_input['attrs_weight'] = np.random.randn(df.shape[0], attrs_max_len, 1)
        return model_input

    def predictAllItemsForUser(self, uid):
        '''
        Returns predicted ratings of all businesses for any user (uid)
        '''
        df = self.data.drop_duplicates('business_id').drop('user_id', axis=1)
        df['user_id'] = uid

        model_input = self._get_model_input(df)
        pred = self.model.predict(model_input, 
                                  batch_size=self.params['batch_size'])
        return pd.DataFrame(pred,index=df['business_id'],columns=['pred'])

    def topN(self, uid, n=5):
        inner_uid = self.lbe_user.transform([uid])[0]
        pred = self.predictAllItemsForUser(inner_uid)
        topn = pred.nlargest(n, columns='pred')
        top_n_iid = self.lbe_item.inverse_transform(topn.index)
        predictions = topn['pred'].to_list()
        n_reviews = self.data['user_id'].value_counts()[inner_uid]
        print()
        print("UserID: {0},  Rated: {1}".format(uid, n_reviews))
        print("--------------------------------")
        topN_business = self.business.loc[top_n_iid]
        for i, (_, business) in enumerate(topN_business.iterrows()):
            print(business['name'])
            print(business['categories'])
            print("Pred: %4.2f  Avg: %3.1f out of %d reviews\n" % \
                  (predictions[i], business['stars'], business['review_count']))

# Test
#deepnn = RecommenderDeepNN(deepfm__seed=2048)
#deepnn.load_data(config.JSON_BUSINESS, config.CSV_RATINGS)
#deepnn.fit()
#deepnn.topN(260, 5)

print("DONE")
