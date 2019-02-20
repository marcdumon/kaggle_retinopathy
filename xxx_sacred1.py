# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - xxx_sacred1.py
# md
# --------------------------------------------------------------------------------------------------------
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('my_test_sacred')


# ex.observers.append(MongoObserver.create())

@ex.capture
def do_random_stuff(a, b, _rnd, _seed):
    print(_seed)
    print(_rnd.randint(1, 100))
    print('----------', a, b)


@ex.automain
def my_main():
    print('Hello world!')
    print('xxx')

#
# if __name__=='__main__':
#     print('xxx')
#
#
