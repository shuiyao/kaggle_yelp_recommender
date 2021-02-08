# Recommender Systems for Yelp

Making recommendations for restaurants using the Yelp dataset.
Include: 
- A content based recommender based on feature vectors constructed from the restaurant descriptions and reviews.
- Collaborative filtering recommender systems based on matrix factorization using the Alternating Least Square (ALS) algorithm.

## Data
[Yelp Dataset on Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset)

## Files
- content.py: A content based recommender.
- als.py: Collaborative filtering recommenders built with the [surprise](https://github.com/NicolasHug/Surprise) and PySpark libraries.
- utils/: 
  - sample.py: Scripts working with the raw data. Extract all restaurants and users from Pittsburgh, PA.
  - nlp.py: Some experiments with feature extraction with NLP tools.
                                                                                                     
