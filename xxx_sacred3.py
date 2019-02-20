# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - xxx_sacred3.py
# md
# --------------------------------------------------------------------------------------------------------

from sacred import Experiment

# import the Ingredient and the function we want to use:
from xxx_sacred2 import data_ingredient, load_data

# add the Ingredient while creating the experiment
ex = Experiment('my_experiment', ingredients=[data_ingredient])


@ex.automain
def run():
    data = load_data()  # just use the function
