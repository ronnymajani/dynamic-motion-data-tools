# -*- coding: utf-8 -*-


# Data Preprocessing Operation
def preprocessingOperation(name):
    def wrap(f):
        f.operation_name = name
        return f
    return wrap
