from distutils.core import setup, Extension

package_name = "jit" # if you change this you also need to change the init function in jitpackage.c

debug = False
import sys
from os import environ, popen
for v in sys.argv:
    if 'debug' in v:
        debug = True
        sys.argv.remove(v)
        break



if debug:
    environ['CFLAGS'] = '-std=c++11 ' + (environ['CFLAGS'] if 'CFLAGS' in environ else '') + '-Wall -O0 -g'
else:
    environ['CFLAGS'] = '-std=c++11 ' + (environ['CFLAGS'] if 'CFLAGS' in environ else '')

environ['CC'] = 'clang++'
environ['CXX'] = 'clang++'

#environ['CFLAGS'] = popen('llvm-config --ldflags --cxxflags --libs --system-libs').read()

LLVM_MODULES = ""#"core engine"
cxx_flags = popen('llvm-config --cxxflags').read()
llvm_includes = [x.replace('-I','') for x in cxx_flags.strip().split(' ')  if '-I' in x]
llvm_libdirs = [x.replace('-L', '') for x in popen('llvm-config --ldflags').read().strip().split(' ')]
llvm_libs = [x.replace('-l', '') for x in popen('llvm-config --system-libs').read().strip().split(' ')] + [x.replace('-l', '') for x in popen('llvm-config --libs %s' % LLVM_MODULES).read().strip().split(' ')]
llvm_defines = [(x.replace('-D',''), None) for x in cxx_flags.strip().split(' ')  if '-D' in x]

#LLVM_MODULES = ""
#environ['CPPFLAGS'] = os.popen('llvm-config --cppflags %s' % LLVM_MODULES).read() + (environ['CPPFLAGS'] if 'CPPFLAGS' in environ else '')
#environ['LDFLAGS'] = os.popen('llvm-config --ldflags %s' % LLVM_MODULES).read() + (environ['LDFLAGS'] if 'LDFLAGS' in environ else '')
#environ['LIBS'] = os.popen('llvm-config --libs %s' % LLVM_MODULES).read() + (environ['LIBS'] if 'LIBS' in environ else '')

import numpy

setup(
    name=package_name,
    version='1.0',
    description='JIT NumPy Arrays.',
    author='Mark Raasveldt',
    ext_modules=[Extension(
        name=package_name,
        include_dirs=[numpy.get_include()] + llvm_includes,
        libraries=llvm_libs,
        library_dirs=llvm_libdirs,
        define_macros=llvm_defines,
        depends=['config.hpp', 'gencode.hpp', 'initializers.hpp', 'operation.hpp', 'thunk.hpp', 'scheduler.hpp', 'debug_printer.hpp', 'thread.hpp', 'compiler.hpp'],
        sources=['jitpackage.cpp', 'operation.cpp', 'thunk.cpp', 'thunk_as_number.cpp', 'thunk_methods.cpp', 'parser.cpp', 'scheduler.cpp', 'debug_printer.cpp', 'compiler.cpp']
        )])
