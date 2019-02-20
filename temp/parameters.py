from pathlib import Path
from random import randint

# set SEED for reproducible results
# See also https://docs.fast.ai/dev/test.html#getting-reproducible-results
# and https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5
SEED = randint(42, 42)

PATH_DATA = Path('../data')
PATH_ORIGINAL = PATH_DATA / 'original'
PATH_SAMPLE = PATH_DATA / 'tmp'
# PATH_SAMPLE = PATH_DATA / 'bsln_train'  # for making baseline data
# PATH_SAMPLE = PATH_DATA / 'tmp_train'
