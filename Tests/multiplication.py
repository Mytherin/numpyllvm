


import jit
import numpy

a = jit.thunk(numpy.arange(100))
b = jit.thunk(numpy.arange(100))
c = jit.thunk(numpy.arange(100))
d = jit.thunk(numpy.arange(100))
e = jit.thunk(numpy.arange(100))
f = jit.thunk(numpy.arange(100))


d = (a * a * a).sort() * (a * b * c)
d.evaluate()


