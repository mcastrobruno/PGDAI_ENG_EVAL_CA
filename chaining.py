import pandas as pd 
import re
from Config import *


# set current iteration of chain type called from main() to programmes y dependent variable
def set_dependent_variable(dependent_variable, df:pd.DataFrame) -> pd.DataFrame:
    df["y"] = dependent_variable
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df

# create chains that later are passed to set_dependent_variable() function i.e. to set y for each level of iteraction through the chain types
def get_chained_variable(df:pd.DataFrame):
    chain_1 = df["y2"]
    chain_2 = df["y2"] + df["y3"]
    chain_3 = df["y2"] + df["y3"] + df["y4"]
    chained_variables = [chain_1, chain_2, chain_3]
    return chained_variables