import numpy as np
import pandas as pd
import re

def filter_business_with_categories(df, to_keep):
    '''
    Return
    ------
    keeprows: bool. If True, keep the row.
    '''
    keeprows = np.array([False] * df.shape[0])
    for i, line in enumerate(df['categories']):
        if(isinstance(line, str)):
            for cat in re.split(', ', line):
                if(cat in to_keep): keeprows[i] = True
    return keeprows
