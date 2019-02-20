# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - xxx_sacred3.py
# md
# --------------------------------------------------------------------------------------------------------

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('config_demo')


# ex.observers.append(MongoObserver.create())


@ex.config
def my_config():
    """This is my demo configuration"""

    a = 10  # some integer

    # a dictionary
    foo = {
        'a_squared': a ** 2,
        'bar': 'my_string%d' % a
    }
    if a > 8:
        # cool: a dynamic entry
        z = a / 2


@ex.main
def run(a, _seed, _rnd):
    print('xxx')
    print(a)
    print(_rnd)


@ex.capture
def tst(a):
    print('--------', a)


# @ex.automain
if __name__ == '__main__':
    ex.run_commandline()
    run()
    tst()
    print(my_config())
