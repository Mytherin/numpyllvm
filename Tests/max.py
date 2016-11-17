


import numpy
import jit

numpy.random.seed(42)

a_nump = numpy.random.randint(1, 100, size=(100,))

a = jit.thunk(a_nump)

res = (a[a >= 50]).sum()
res.evaluate()

res2 = a.sum()
res2.evaluate()

print(res)
print((a_nump[a_nump >= 50]).sum())

print(res2)
print(a_nump.sum())


print(str(res) == str(a_nump[a_nump >= 50].max()))
