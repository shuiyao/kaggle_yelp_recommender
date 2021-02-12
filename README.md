# Recommender Systems for Yelp

Making recommendations for businesses using the Yelp dataset.
Include: 
- A recommender based on a deep factoriztion machine model (DeepFM).
- A content based recommender based on feature vectors constructed from the restaurant descriptions and reviews.
- Collaborative filtering recommender systems based on matrix factorization using the Alternating Least Square (ALS) algorithm.

## Data
[Yelp Dataset on Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset)

## Files
- deepnn.py: Recommender based on DeepFM using the [DeepCTR](https://deepctr-doc.readthedocs.io/en/latest/index.html) library.
- content.py: A content based recommender for restaurants based on the 'attributes'.
- als.py: Collaborative filtering recommenders built with the [surprise](https://github.com/NicolasHug/Surprise) and PySpark libraries.
- sentiment.py: Sentiment analysis on the review text using a LSTM model with pre-trained GloVe word embeddings.
- utils.py: User-defined help functions.
- config.py: User-defined global variables.
- utils/: Some extra scripts not called by the recommenders.
  - sample.py: Scripts working with the raw data. Extract all restaurants and users from Pittsburgh, PA.
  - nlp.py: Some experiments with feature extraction with NLP tools.
                                                                                                     
