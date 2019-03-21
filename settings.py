# --------------------------------------------------------------------------------------------------------
# 2019/03/20
# retinopathy - settings.py
# md
# --------------------------------------------------------------------------------------------------------
import random
import numpy as np
from ruamel_yaml import YAML

# Make config global variable
yaml = YAML(typ='safe')
with open('config.yaml') as fl:
    CONFIG = yaml.load(fl)

# set random seeds
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
