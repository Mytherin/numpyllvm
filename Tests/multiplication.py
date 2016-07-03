


import numpy
import jit

numpy.random.seed(42)

a_nump = numpy.random.randint(1, 100, size=(100,))
b_nump = numpy.random.randint(1, 100, size=(100,))
c_nump = numpy.random.randint(1, 100, size=(100,))
d_nump = numpy.random.randint(1, 100, size=(100,))
e_nump = numpy.random.randint(1, 100, size=(100,))
f_nump = numpy.random.randint(1, 100, size=(100,))

res_nump = (d_nump * e_nump * f_nump) * (a_nump * b_nump * c_nump)

a = jit.thunk(a_nump)
b = jit.thunk(b_nump)
c = jit.thunk(c_nump)
d = jit.thunk(d_nump)
e = jit.thunk(e_nump)
f = jit.thunk(f_nump)

res = (d * e * f) * (a * b * c)
res.evaluate()

print(str(res) == str(res_nump))