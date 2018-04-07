# -*- coding: utf-8 -*-


# Data Preprocessing Function
def preprocessingFunction(name):
    def wrap(f):
        f.operation_name = name
        return f
    return wrap
