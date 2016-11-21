

import numpy
import jit

numpy.random.seed(42)

a_nump = numpy.random.randint(1, 100, size=(100,))
b_nump = numpy.random.randint(1, 100, size=(100,))

a = jit.thunk(a_nump)
b = jit.thunk(b_nump)

c = a * b
a[0] = 1000
c.evaluate()


print(c)
print(a_nump * b_nump)
