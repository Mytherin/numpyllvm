


import numpy
import jit

numpy.random.seed(42)

a_nump = numpy.random.randint(1, 100, size=(100,))

a = jit.thunk(a_nump)

res = a[a >= 50]
res.evaluate()

print(res)
print(a_nump[a_nump >= 50])

print(str(res) == str(a_nump[a_nump >= 50]))
