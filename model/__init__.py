from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .model import E_Path_Tracer

def setup(opt):
    model = E_Path_Tracer(opt)
    model.rate_choice = 1
    model.path_related_module_build()
    model.graph_related_module_build()
    model.predictor_module_build()
    return model