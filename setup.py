from distutils.core import setup, Extension

package_name = "jit" # if you change this you also need to change the init function in jitpackage.c

debug = False
import sys
from os import environ
for v in sys.argv:
    if 'debug' in v:
        debug = True
        sys.argv.remove(v)
        break

if debug:
    environ['CFLAGS'] = (environ['CFLAGS'] if 'CFLAGS' in environ else '') + '-Wall -O0 -g'
else:
    environ['CFLAGS'] = (environ['CFLAGS'] if 'CFLAGS' in environ else '')

environ['CC'] = 'g++'

import numpy

setup(
    name=package_name,
    version='1.0',
    description='JIT NumPy Arrays.',
    author='Mark Raasveldt',
    ext_modules=[Extension(
        name=package_name,
        include_dirs=[numpy.get_include()],
        depends=['config.hpp', 'gencode.hpp', 'initializers.hpp', 'operation.hpp', 'thunk.hpp'],
        sources=['jitpackage.cpp', 'operation.cpp', 'thunk.cpp', 'thunk_as_number.cpp', 'thunk_methods.cpp', 'parser.cpp']
        )])

