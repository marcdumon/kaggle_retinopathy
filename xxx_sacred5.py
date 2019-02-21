# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - xxx_sacred3.py
# md
# --------------------------------------------------------------------------------------------------------

from sacred import Experiment
from sacred.observers import MongoObserver
from xxx_sacred4 import my_config
from xxx_sacred1 import do_random_stuff

ex = Experiment('my test')
ex.observers.append(MongoObserver.create())
ex.add_config(my_config())
print(type(my_config()))
print('xxx')

@ex.capture
def test(a, _seed):
    print('test')
    print(a, _seed)
    do_random_stuff(a, 'b')

@ex.main
def xxx():  # xxxx
    print('xxx')
    test()
    # test()



if __name__ == '__main__':
    ex.run_commandline()
    # test()
