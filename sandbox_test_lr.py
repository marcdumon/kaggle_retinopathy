# --------------------------------------------------------------------------------------------------------
# 2019/03/11
# retinopathy - sandbox_test_lr.py
# md
# --------------------------------------------------------------------------------------------------------

import fastai.vision as fv

print(fa.__version__)
d = fv.DataBunch()
l = fv.cnn_learner(d, fv.models.resnet18())
l.lr
