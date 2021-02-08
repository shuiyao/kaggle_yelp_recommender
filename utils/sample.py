# Purpose: Sub-sampling from the raw data to test algorithms
# Focus on restaurants/users from Pittsburgh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json_to_csv_converter as json2csv
import seaborn as sns
import json
import csv

m = 0
folder_input = "./data/"
fout = ""
encoding = 'utf-8'

def read_raw_data():
    fin_business = folder_input + "yelp_academic_dataset_business.json"

    df_business = pd.read_json(fin_business, lines=True)
    df = df_business
    df.isna().sum()
    # Use Pittsburgh as the city, all of them in PA
    df = df[df.city == 'Pittsburgh']
    m = df.shape[0]
    return df

# business_id         0
# name                0
# address             0
# city                0
# state               0
# postal_code         0
# latitude            0
# longitude           0
# stars               0
# review_count        0
# is_open             0
# attributes      29045
# categories        524
# hours           44843

def show_location(data):
    ''' Is there any pattern based on location? For example, I suppose restaurants near universities are likely better?
    '''
    # sns.scatterplot(x='longitude', y='latitude', data=data, \
    #                 hue='stars', size=3, palette="rainbow")
    sns.scatterplot(data.longitude, data.latitude, \
                    hue=data.stars, size=3, palette="rainbow")
    # U. Pitts: 40.4444° N, 79.9608° W
    plt.plot([-79.9608], [40.4444], "k*", markersize=8)

def show_closed(data):
    ''' Is there a correlation between review count and stars? What's about the closed restaurants? '''
    sns.scatterplot(df.stars+0.4*np.random.random(m), df.review_count, hue=df.is_open,size=3)

# fin_review = folder_input + "yelp_academic_dataset_review.json"    
# cols_review = json2csv.get_superset_of_column_names_from_file(fin_review)
# cols_review = {'date', 'user_id', 'useful', 'funny', 'cool', 'review_id', 'text', 'business_id', 'stars'}

# fin_user = folder_input + "yelp_academic_dataset_user.json"    
# cols_user = json2csv.get_superset_of_column_names_from_file(fin_user)


fcsv = folder_input + "business_pittsburgh.csv"

def write_data(df, name='Pittsburgh'):
    fout = folder_input + "business_"+name.lower()+".json"
    df.to_json(fout, orient='records', lines=True)
    fcsv = '{0}.csv'.format(fout.split('.json')[0])
    cols = json2csv.get_superset_of_column_names_from_file(fout)
    json2csv.read_and_write_file(fout, fcsv, cols)

def load_csv_data(fcsv):
    df = pd.read_csv(fcsv)
    m = df.shape[0]
    return df

    # Rank the categories by frequency
def compile_business_categories():
    import re
    from collections import Counter
    categories = Counter()
    for line in df['categories']:
        if(isinstance(line, str)):
            categories.update(re.split(', ', line[2:-1]))
    categories = pd.DataFrame.from_dict(categories, orient='index')

# Now get all keywords from categories
if __name__ == '__main__':
    df = load_csv_data(fcsv)
    sum_nnan = (m - df.isna().sum()).sort_values()
    frac_nan = (df.isna().sum() / m).sort_values()

# Find checkin data:
def read_write_checkin(df):
    ''' Read raw checkin.json file and output to .csv that contains only business in Pittsburgh and keep only the total number, first and last of the checkin records.
    '''
    fin_checkin = folder_input + "yelp_academic_dataset_checkin.json" 
    df_checkin = pd.read_json(fin_checkin, lines=True)
    df_checkin['business_id'] = df_checkin['business_id'].str.encode(encoding)
    df_checkin['business_id'] = df_checkin['business_id'].astype('str')
    len(df_checkin['date'][0].split(','))
    df_checkin = df_checkin.merge(df['business_id'], how='right', left_on='business_id', right_on='business_id')
    dates = df_checkin['date'].str.split(',')
    df_checkin['checkin_total'] = dates.str.len()
    df_checkin['checkin_first'] = dates.str[0]
    df_checkin['checkin_last'] = dates.str[-1]
    df_checkin = df.checkin.drop('date')
    fcsv_checkin = folder_input + "checkin_pittsburgh.csv"
    df_checkin.to_csv(fcsv_checkin)

def select_review_from_raw():
    df = load_csv_data(fcsv)
    fin_review = folder_input + "yelp_academic_dataset_review.json"
    set_business_id = set(df['business_id'])
    # Too large to fit in RAM
    # df_review = pd.read_json(fin_review, lines=True)
    cols = {'date', 'user_id', 'useful', 'funny', 'cool', 'review_id', 'text', 'business_id', 'stars'}
    fout = open(folder_input + "review_pittsburgh.csv", "w")
    csv_file = csv.writer(fout)
    csv_file.writerow(list(cols))
    i = 0
    with open(fin_review, "r") as fin:
        for line in fin:
            i += 1
            if(i % 100000 == 0): print("Entry %d in yelp_..._review.json" % i)
            contents = json.loads(line)
            bid = str(contents['business_id'])
            if(bid in set_business_id):
                csv_file.writerow(json2csv.get_row(contents, cols))
    fout.close()

def extract_ratings_from_review():
    import ast
    fcsv_review = folder_input + "review_pittsburgh.csv"
    fcsv_ratings = folder_input + "ratings_pittsburgh.csv"
    df_review = pd.read_csv(fcsv_review)
    df_review['business_id'] = df_review['business_id'].apply(lambda l: l[2:-1])
    df_review['user_id'] = df_review['user_id'].apply(lambda l: l[2:-1])
    df_review[['user_id', 'business_id','stars']].to_csv(fcsv_ratings, index=False)

def select_user_from_raw():
    df_review = load_csv_data(fcsv_review)
    fin_user = folder_input + "yelp_academic_dataset_user.json"
    set_user_id = set(df_review['user_id'])
    # Too large to fit in RAM
    # df_review = pd.read_json(fin_review, lines=True)

    cols = {'name', 'compliment_list', 'compliment_cool', 'funny', 'compliment_funny', 'compliment_photos', 'compliment_plain', 'compliment_more', 'compliment_hot', 'review_count', 'compliment_writer', 'useful', 'compliment_cute', 'cool', 'compliment_note', 'average_stars', 'yelping_since', 'user_id', 'elite', 'compliment_profile'}
    # cols = {'name', 'compliment_list', 'compliment_cool', 'funny', 'compliment_funny', 'compliment_photos', 'compliment_plain', 'compliment_more', 'fans', 'compliment_hot', 'review_count', 'compliment_writer', 'friends', 'useful', 'compliment_cute', 'cool', 'compliment_note', 'average_stars', 'yelping_since', 'user_id', 'elite', 'compliment_profile'}
    
    fout = open(folder_input + "user_pittsburgh.csv", "w")
    csv_file = csv.writer(fout)
    csv_file.writerow(list(cols))
    with open(fin_user, "r") as fin:
        for line in fin:
            contents = json.loads(line)
            uid = str(contents['user_id'].encode(encoding))
            if(uid in set_user_id):
                csv_file.writerow(json2csv.get_row(contents, cols))
    fout.close()

print("DONE")
