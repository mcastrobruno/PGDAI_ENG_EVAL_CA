from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random

# create chains that later are passed to set_dependent_variable() function i.e. to set y for each level of iteraction through the chain types
def get_chained_variable(df:pd.DataFrame):
    chain_1 = df["y2"]
    chain_2 = df["y2"] + df["y3"]
    chain_3 = df["y2"] + df["y3"] + df["y4"]
    chained_variables = [chain_1, chain_2, chain_3]
    return chained_variables