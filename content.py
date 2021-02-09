import pandas as pd
import numpy as np
import ast
import os, argparse

import config, utils

from sklearn.feature_extraction import DictVectorizer

from scipy import sparse

# Updating...:
#  - Add features from the "categories". Now only have features from 'attributes'
#  - Account for 'stars' more efficiently

# Default parameters
params = {'category':'restaurants', 'min_idf':10, 'min_reviews':5}

class ContentBasedRecommender:
    '''
    Content based recommender. Construct feature vector for each restaurant based on selected features from its attributes and catogories. Generate user profiles based on previously reviewed restaurants.
    '''
    def __init__(self, path_business=config.JSON_BUSINESS, path_ratings=config.CSV_RATINGS):
        '''
        Parameters
        ----------
        path_business: Path to the business.json file that contains 'attributes' and 'catogories' as dictionaries for all businesses
        path_ratings: Path to the ratings.csv file that contains 'user_id', 'business_id' and 'stars'. The review text is not needed here.
        '''
        self.path_business = path_business
        self.path_ratings = path_ratings
        self.n_items = None
        self.business = None # Placeholder for businesses
        self.ratings = None # Placeholder for the rating matrix
        self.min_idf = params['min_idf']
        self.min_reviews = params['min_reviews']
        self.category = params['category']
        
    def _load_data(self, build_feature_matrix=True, nrows=None):
        '''
        Only keep businesses that have enough reviews.
        '''
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

        self.feature_matrix, self.feature_names, self.feature_idf = \
            self._content_based_matrix_for_items(df, min_idf=self.min_idf)
        
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

        self.user_groups = self.ratings.groupby('UID') # <<<

        del df, df_ratings

    def _clean(self):
        self.ratings = None
        self.business = None

    def _content_based_matrix_for_items(self, df, min_idf=10, verbose=False):
        '''
        Generate the sparse feature matrix for all the items. Prepare for content based recommendation.

        Parameters
        ----------
        df: DataFrame for businesses, must have 'attributes' and 'categories' as dictionaries.
        min_idf: Keep only features with IDF larger than min_idf (default: 10).

        Returns
        -------
        mat: Sparse matrix with shape (n_items, n_features)
        features: The name of the features
        idf: Inverse Document Frequency for each binary feature. log(N/(N|1))

        Notes
        --------
        df['Ambience'].loc[7625] = 
        {'romantic': False, 'intimate': False, 'classy': False, ...}
        will be turned into:
        romantic intemate classy ...
        0        0        0
        '''

        m = df.shape[0]
        for var in config.Attributes:
            df[var] = df['attributes'].map(lambda l: l[var] if l is not None and var in l else None)
            # Savior: ast.literal_eval("u'string'") -> string
            df[var] = df[var].apply(lambda l: str(ast.literal_eval(l)) if l is not None else None)
            if(verbose):
                print(df[var].value_counts())

        att = pd.get_dummies(df[config.Attributes], sparse=True)
        mat = sparse.csr_matrix(att)
        features = att.columns

        # Use DictVectorizer to encode attributes as the dict type
        vec = DictVectorizer()
        for var in config.Attributes_Dict:
            df[var] = df['attributes'].map(lambda l: ast.literal_eval(l[var]) if (l is not None and var in l) else dict())
            df[var] = df[var].apply(lambda l: dict() if l is None else l)
            att_dict = vec.fit_transform(df[var]).astype('uint8')
            mat = sparse.hstack((mat, att_dict))
            features = np.concatenate((features, vec.get_feature_names()))

        mat = mat.toarray()
        # Drop features with too low frequency
        idf = mat.sum(axis=0)
        drop_cols = np.argwhere(idf < min_idf).flatten()
        print("Features to drop: ", features[drop_cols])
        mat = np.delete(mat, drop_cols, axis=1)
        features = np.delete(features, drop_cols)
        idf = np.delete(idf, drop_cols)
        idf = np.log(m/idf)
        return mat, features, idf

    def topN(self, uid, n=5):
        '''
        Recommend the top N restaurants for any user given uid.
        '''
        assert(1) # LATER
        # ISSUE: What is some business have no attributes?
        grp = self.user_groups.get_group(uid)
        grp['score'] = np.maximum(grp['stars'] - 2.9, 0.0)
        # get user profile from previously rated businesses
        user_profile = np.dot(self.feature_matrix[grp['IID']].T, grp['score']) * self.feature_idf
        top_features = np.argsort(user_profile)[-5:][::-1]
        print("UserID: {0},  Rated: {1}".format(uid, grp.shape[0]))
        print("Top Features: ", self.feature_names[top_features])
        print("--------------------------------")
        # consine similarity between user profile and business
        score = np.dot(self.feature_matrix, user_profile) / np.maximum(np.linalg.norm(self.feature_matrix, axis=1), 0.01)
        top_n_iid = np.argsort(score)[-n:][::-1]
        topN_business = self.business.loc[top_n_iid]
        for i, business in topN_business.iterrows():
            print(business['name'])
            print(business['categories'])
            print(self.feature_names[np.argsort(self.feature_matrix[i])[-3:][::-1]])
            print("Avg: %3.1f out of %d reviews\n" % \
                  (business['stars'], business['review_count']))
        return top_n_iid, user_profile

def parse_args():
    parser = argparse.ArgumentParser(prog="Recommender For Yelp Restaurants",
        description="Content-Based, cosine-similarity")
    parser.add_argument('--path', nargs='?', default=config.FOLDER_INPUT,
                        help='input data path')
    parser.add_argument('--business', nargs='?', default='business_pittsburgh.json',
                        help='filename for the businesses')
    parser.add_argument('--ratings', nargs='?', default='ratings_pittsburgh.csv',
                        help='file for the ratings')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of recommendations')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    folder_input = args.path
    f_business = args.business
    f_ratings = args.ratings
    top_n = args.top_n

    # cr = ContentBasedRecommender(config.JSON_BUSINESS, config.CSV_RATINGS)
    cr = ContentBasedRecommender(
        os.path.join(folder_input, f_business),
        os.path.join(folder_input, f_ratings))
    # set parameters
    
    cr._load_data()

    stopwords = ['stop', 'exit', 'quit', 'q']
    uid = ''
    while(uid not in stopwords):
        uid = input("Recommend Restaurants for user (or 'quit'): ")
        if(uid.isnumeric()):
            cr.topN(int(uid), top_n)
        else:
            continue

# cr = ContentBasedRecommender(config.JSON_BUSINESS, config.CSV_RATINGS)
# cr._load_data()
# cr.topN(260,5)
