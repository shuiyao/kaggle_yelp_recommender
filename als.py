import pandas as pd
import numpy as np
import ast
import os, warnings, argparse

import config, utils

# Default parameters
params = {'category':'restaurants', 'min_reviews':5}
params_surprise = {'method':'als', 'n_epochs':10, 'reg_u':15, 'reg_i':10}
params_sparkALS = {'rank':20, 'maxIter':10, 'regParam':0.1, 'alpha':1.0}

from surprise import Reader, Dataset
from surprise import BaselineOnly
from surprise.model_selection import cross_validate

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

class ALSRecommender:
    '''
    Recommender using collaborative filtering.
    '''
    def __init__(self, path_business=config.JSON_BUSINESS, path_ratings=config.CSV_RATINGS, alslib='surprise'):
        '''
        Parameters
        ----------
        path_business: Path to the business.json file that contains 'attributes' and 'catogories' as dictionaries for all businesses
        path_ratings: Path to the ratings.csv file that contains 'user_id', 'business_id' and 'stars'. The review text is not needed here.
        '''
        self.path_business = path_business
        self.path_ratings = path_ratings
        self.n_users = None
        self.n_items = None
        self.business = None # Placeholder for businesses
        self.ratings = None # Placeholder for the rating matrix
        self.min_reviews = params['min_reviews']
        self.category = params['category']        

        self.rating_matrix = None

        self.alslib = alslib
        if(alslib == 'surprise'):
            self.model = self.ALSModelSurprise(params_surprise)
        if(alslib == 'spark'):
            self.model = self.ALSModelSpark(params_sparkALS)
        
    def _load_data(self, nrows=None):
        '''
        Only keep businesses that have enough reviews.
        '''
        self.nrows = nrows
        df = pd.read_json(self.path_business, lines=True, encoding='utf-8')

        to_keep = config.Keywords_Categories[self.category]
        keeprows = utils.filter_business_with_categories(df, to_keep)
        df = df[keeprows]
        
        self.n_items = df.shape[0]
        df = df[df['review_count'] > self.min_reviews]
        print("Select {0} out {1} businesses have more than {2:d} reviews."\
              .format(df.shape[0], self.n_items, self.min_reviews))
        df = df.reset_index()
        self.n_items = df.shape[0]

        df_ratings = pd.read_csv(self.path_ratings, nrows=nrows)

        # Represent user_id and business_id with unique integers
        self.uid_to_raw = dict(df_ratings['user_id'].drop_duplicates().reset_index()['user_id'])
        self.raw_to_uid = {k:v for v, k in self.uid_to_raw.items()}
        self.iid_to_raw = dict(df['business_id'])
        self.raw_to_iid = {k:v for v, k in self.iid_to_raw.items()}

        df_ratings['UID'] = df_ratings['user_id'].map(self.raw_to_uid)
        df_ratings['IID'] = df_ratings['business_id'].map(self.raw_to_iid)
        
        df_ratings = df_ratings.dropna(subset=['IID'])
        df_ratings['IID'] = df_ratings['IID'].astype('int64')
        # print(df_ratings.dtypes)

        self.ratings = df_ratings[['UID', 'IID', 'stars']]
        self.business = df[['name', 'categories', 'review_count', 'stars']]

        del df, df_ratings

    def set_params(self, **kwargs):
        for (k, v) in kwargs.items():
            if(k in self.model.params):
                self.model.params[k] = v
            else:
                warnings.warn('{0} is not a parameter for ALSModel.'.format(k))
        self.model.update_parameters()

    def _clean(self):
        self.ratings = None
        self.business = None

    def _build_rating_matrix(self):
        '''
        Build the sparse user-item rating matrix from the ratings data
        '''
        print("Building Rating Matrix (may take a while)...")
        self.rating_matrix = pd.pivot_table(
            self.ratings, values='stars', \
            index=['UID'], columns=['IID'])
        # x: user, y: business
        self.n_users, self.n_items = self.rating_matrix.shape
        self.sparsity = self.rating_matrix.isna().sum().sum() \
            / (self.n_users * self.n_items)

        print ("Users: {0:5d}    Businesses: {1:4d}".\
               format(self.n_users, self.n_items))
        print ("Sparsity = %5.2f%%" % (self.sparsity * 100.0))
        # review_count_u = (nx - rating_matrix.isna().sum()).sort_values()
        # review_count_b = (ny - rating_matrix.isna().sum(axis=1)).sort_values()

    def parse_data_to_model(self):
        if(self.alslib == 'surprise'): self.model.parse_data(self.ratings)
        if(self.alslib == 'spark'): self.model.parse_data(self.path_ratings, self.nrows)
        
    def fit(self):
        self.model.fit()

    def grid_search(self):
        self.model.grid_search()

    def topN(self, uid, n=5, return_value=False):
        '''
        Recommend the top N restaurants for any user given uid.
        '''
        top_n_iid, predictions = self.model.top_n_recommendations(uid, n)

        n_reviews = self.ratings['UID'].value_counts()[uid]
        print()
        print("UserID: {0},  Rated: {1}".format(uid, n_reviews))
        print("--------------------------------")
        topN_business = self.business.loc[top_n_iid]
        for i, (_, business) in enumerate(topN_business.iterrows()):
            print(business['name'])
            print(business['categories'])
            print("Pred: %4.2f  Avg: %3.1f out of %d reviews\n" % \
                  (predictions[i], business['stars'], business['review_count']))
        if(return_value):
            return top_n_iid

    class ALSModel:
        def __init__(self, params):
            self.params = params

    class ALSModelSurprise(ALSModel):
        def __init__(self, params):
            super().__init__(params)
            self.algo = BaselineOnly(bsl_options=self.params)

        def parse_data(self, ratings):
            reader = Reader(rating_scale = (1,5))
            self.data = Dataset.load_from_df(ratings, reader)

        def update_parameters(self):
            self.algo.bsl_options = self.params

        def fit(self):
            self.train = self.data.build_full_trainset()
            self.algo.fit(self.train)

        def predict(self, uid, iid):
            '''
            uid, iid should be consistent with ratings['UID','IID']
            '''
            return self.algo.predict(uid, iid).est

        def top_n_recommendations(self, uid, n=5):
            '''
            Obtain the top n recommendation for any user.
            Method for the surprise library
            '''
            scores = []
            for i in range(self.train.n_items):
                iid = self.train.to_raw_iid(i)
                scores.append((iid, self.predict(uid, iid)))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_n_iid = [l[0] for l in scores[:n]]
            pred = [l[1] for l in scores[:n]]
            return top_n_iid, pred

        def cross_validate(self, cv=5, verbose=False):
            cv_result = cross_validate(self.algo, self.data, \
                                       cv=cv, verbose=verbose)
            rmse = cv_result['test_rmse'].mean()
            return rmse

        def grid_search(self):
            self._best_params = self.params
            self._best_rmse = self.cross_validate(cv=5)
            for n_epochs in [5, 10, 15, 20, 25]:
                for reg_u in [5, 10, 15, 20]:
                    for reg_i in [5, 10, 15]:
                        self.set_params(n_epochs=n_epochs, reg_u=reg_u, reg_i=reg_i)
                        rmse = self.cross_validate(cv=5)
                        print(n_epochs, reg_u, reg_i, rmse)
                        if(rmse < self._best_rmse):
                            self._best_rmse = rmse
                            self._best_params = self.params

    class ALSModelSpark(ALSModel):
        def __init__(self, params):
            super().__init__(params)
            self.sqlContext = SQLContext(sc)
            self._als = ALS(userCol='UID', itemCol='IID', ratingCol='stars', \
                            coldStartStrategy="drop")
            self._als.setParams(**self.params)

        def parse_data(self, path_ratings, nrows):
            df_ratings = self.sqlContext.read.csv(path_ratings, header=True, quote='"').limit(nrows)
            # self.data.count()
            raw_to_uid = StringIndexer(inputCol="user_id", outputCol="UID").fit(df_ratings)
            self.data = raw_to_uid.transform(df_ratings)
            raw_to_iid = StringIndexer(inputCol="business_id", outputCol="IID").fit(df_ratings)
            self.data = raw_to_iid.transform(self.data)
            # uid and iid must be integers for spark ALS 
            self.data = self.data.rdd.map(\
                        lambda r: (int(r['UID']), \
                                   int(r['IID']), \
                                   float(r['stars'])))\
                        .toDF(("UID", "IID", "stars"))

        def update_parameters(self):
            self._als.setParams(**self.params)

        def fit(self):
            self._model = self._als.fit(self.data)

        def predict(self, uid, iid):
            return self._model.transform(test)

        def top_n_recommendations(self, uid, n=5):
            '''
            Obtain the top n recommendation for any user.
            Method for the Spark library
            '''
            users = self.data.select(self._als.getUserCol())
            user = users.filter(users['UID'] == uid)
            topN = self._model.recommendForUserSubset(user, 5).collect()
            top_n_iid, predictions = [], []
            for row in topN[0].recommendations:
                top_n_iid.append(row.IID)
                predictions.append(row.rating)
            return top_n_iid, predictions

        def cross_validate(self, train=0.8):
            '''
            Return the RMSE of the cross validation set.

            Parameters
            ----------
            train: The fraction to use for training. Default: 0.8
            '''
            (trainset, testset) = self.data.randomSplit([train, 1.-train])
            _model = self._als.fit(trainset)
            _pred = _model.transform(testset)
            # pred_.count() may be very small
            _eval = RegressionEvaluator(\
                    metricName='rmse', labelCol='stars', \
                    predictionCol='prediction')
            rmse = _eval.evaluate(_pred)
            return rmse

def parse_args():
    parser = argparse.ArgumentParser(prog="Recommender For Yelp Restaurants",
        description="Content-Based, cosine-similarity")
    parser.add_argument('--path', nargs='?', default=config.FOLDER_INPUT,
                        help='input data path')
    parser.add_argument('--business', nargs='?', default='business_pittsburgh.json',
                        help='filename for the businesses')
    parser.add_argument('--ratings', nargs='?', default='ratings_pittsburgh.csv',
                        help='file for the ratings')
    parser.add_argument('--alslib', nargs='?', default='surprise',
                        help='ALS library to use (surprise or spark)')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of recommendations')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    folder_input = args.path
    f_business = args.business
    f_ratings = args.ratings
    alslib = args.alslib
    top_n = args.top_n

    if(alslib == 'spark'):
        sc = SparkContext(appName="YelpSparkCFRS")

    # als = ALSRecommender(config.JSON_BUSINESS, config.CSV_RATINGS)
    als = ALSRecommender(
        os.path.join(folder_input, f_business),
        os.path.join(folder_input, f_ratings),
        alslib
    )
    # set parameters
    # Do not set nrows to use the entire dataset.
    als._load_data(nrows=20000)
    als.fit()

    stopwords = ['stop', 'exit', 'quit', 'q']
    uid = ''
    while(uid not in stopwords):
        uid = input("Recommend Restaurants for user (or 'quit'): ")
        if(uid.isnumeric()):
            als.topN(int(uid), top_n)
        else:
            continue

    if(alslib == 'spark'):
        sc.stop()

# Test Only
def test_surprise_als():
    als = ALSRecommender(config.JSON_BUSINESS, config.CSV_RATINGS, 'surprise')
    als._load_data(nrows=200)
    als.parse_data_to_model()
    als.fit()
    als.topN(260, n=5)
    return als

def test_spark_als():
    # als.model.sc.stop()

    als = ALSRecommender(config.JSON_BUSINESS, config.CSV_RATINGS, 'spark')
    als._load_data(nrows=20000)
    als.parse_data_to_model()

    als.fit()
    als.topN(260, 5)
    return als
    
