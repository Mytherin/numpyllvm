


import numpy
import jit

numpy.random.seed(42)

a_nump = numpy.random.randint(1, 100, size=(100,))

a = jit.thunk(a_nump)

res = (a[a >= 50]).sum()
res2 = res * a
res2.evaluate()

print(res)
print(res2)
print(numpy.array((a_nump[a_nump >= 50]).sum()))
print(a_nump[a_nump >= 50].sum() * a_nump)

print(str(res2) == str(a_nump[a_nump >= 50].sum() * a_nump))
