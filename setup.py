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
    environ['CFLAGS'] = '-stdlib=libc++ -std=c++11 -pthread ' + (environ['CFLAGS'] if 'CFLAGS' in environ else '') + '-Wall -O0 -g'
else:
    environ['CFLAGS'] = '-stdlib=libc++ -std=c++11 -pthread ' + (environ['CFLAGS'] if 'CFLAGS' in environ else '')

environ['CC'] = '/opt/local/bin/clang++'
environ['CXX'] = '/opt/local/bin/clang++'

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

def setup_package():
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
            depends=['config.hpp', 'gencode.hpp', 'initializers.hpp', 'operation.hpp', 'thunk.hpp', 'scheduler.hpp', 'debug_printer.hpp', 'thread.hpp', 'compiler.hpp', 'llvmjit.hpp', 'optimizer.hpp', 'mutex.hpp'],
            sources=['jitpackage.cpp', 'operation.cpp', 'thunk.cpp', 'thunk_as_number.cpp', 'thunk_as_mapping.cpp', 'thunk_as_sequence.cpp', 'thunk_methods.cpp', 'parser.cpp', 'scheduler.cpp', 'debug_printer.cpp', 'compiler.cpp', 'thread.cpp', 'llvmjit.cpp', 'optimizer.cpp', 'mutex.cpp']
            )])

try:
    setup_package();
except:
    pass
import os
os.system('/opt/local/bin/clang++ -bundle -undefined dynamic_lookup -L/opt/local/lib -Wl,-headerpad_max_install_names -L/opt/local/lib -stdlib=libc++ -std=c++11 -pthread -I/opt/local/include -I/opt/local/include build/temp.macosx-10.4-x86_64-2.7-pydebug/jitpackage.o build/temp.macosx-10.4-x86_64-2.7-pydebug/operation.o build/temp.macosx-10.4-x86_64-2.7-pydebug/thunk.o build/temp.macosx-10.4-x86_64-2.7-pydebug/thunk_as_number.o build/temp.macosx-10.4-x86_64-2.7-pydebug/thunk_as_mapping.o build/temp.macosx-10.4-x86_64-2.7-pydebug/thunk_as_sequence.o build/temp.macosx-10.4-x86_64-2.7-pydebug/thunk_methods.o build/temp.macosx-10.4-x86_64-2.7-pydebug/parser.o build/temp.macosx-10.4-x86_64-2.7-pydebug/scheduler.o build/temp.macosx-10.4-x86_64-2.7-pydebug/debug_printer.o build/temp.macosx-10.4-x86_64-2.7-pydebug/compiler.o build/temp.macosx-10.4-x86_64-2.7-pydebug/thread.o build/temp.macosx-10.4-x86_64-2.7-pydebug/llvmjit.o build/temp.macosx-10.4-x86_64-2.7-pydebug/optimizer.o build/temp.macosx-10.4-x86_64-2.7-pydebug/mutex.o -L/opt/local/libexec/llvm-3.9/lib -L-Wl,-search_paths_first -L-Wl,-headerpad_max_install_names -lcurses -lz -lm -lLLVMLTO -lLLVMObjCARCOpts -lLLVMSymbolize -lLLVMDebugInfoPDB -lLLVMDebugInfoDWARF -lLLVMMIRParser -lLLVMCoverage -lLLVMTableGen -lLLVMOrcJIT -lLLVMXCoreDisassembler -lLLVMXCoreCodeGen -lLLVMXCoreDesc -lLLVMXCoreInfo -lLLVMXCoreAsmPrinter -lLLVMSystemZDisassembler -lLLVMSystemZCodeGen -lLLVMSystemZAsmParser -lLLVMSystemZDesc -lLLVMSystemZInfo -lLLVMSystemZAsmPrinter -lLLVMSparcDisassembler -lLLVMSparcCodeGen -lLLVMSparcAsmParser -lLLVMSparcDesc -lLLVMSparcInfo -lLLVMSparcAsmPrinter -lLLVMPowerPCDisassembler -lLLVMPowerPCCodeGen -lLLVMPowerPCAsmParser -lLLVMPowerPCDesc -lLLVMPowerPCInfo -lLLVMPowerPCAsmPrinter -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMNVPTXAsmPrinter -lLLVMMSP430CodeGen -lLLVMMSP430Desc -lLLVMMSP430Info -lLLVMMSP430AsmPrinter -lLLVMMipsDisassembler -lLLVMMipsCodeGen -lLLVMMipsAsmParser -lLLVMMipsDesc -lLLVMMipsInfo -lLLVMMipsAsmPrinter -lLLVMHexagonDisassembler -lLLVMHexagonCodeGen -lLLVMHexagonAsmParser -lLLVMHexagonDesc -lLLVMHexagonInfo -lLLVMBPFCodeGen -lLLVMBPFDesc -lLLVMBPFInfo -lLLVMBPFAsmPrinter -lLLVMARMDisassembler -lLLVMARMCodeGen -lLLVMARMAsmParser -lLLVMARMDesc -lLLVMARMInfo -lLLVMARMAsmPrinter -lLLVMAMDGPUDisassembler -lLLVMAMDGPUCodeGen -lLLVMAMDGPUAsmParser -lLLVMAMDGPUDesc -lLLVMAMDGPUInfo -lLLVMAMDGPUAsmPrinter -lLLVMAMDGPUUtils -lLLVMAArch64Disassembler -lLLVMAArch64CodeGen -lLLVMGlobalISel -lLLVMAArch64AsmParser -lLLVMAArch64Desc -lLLVMAArch64Info -lLLVMAArch64AsmPrinter -lLLVMAArch64Utils -lLLVMObjectYAML -lLLVMLibDriver -lLLVMOption -lLLVMX86Disassembler -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMSelectionDAG -lLLVMAsmPrinter -lLLVMDebugInfoCodeView -lLLVMX86Desc -lLLVMMCDisassembler -lLLVMX86Info -lLLVMX86AsmPrinter -lLLVMX86Utils -lLLVMMCJIT -lLLVMLineEditor -lLLVMPasses -lLLVMipo -lLLVMVectorize -lLLVMLinker -lLLVMIRReader -lLLVMAsmParser -lLLVMInterpreter -lLLVMExecutionEngine -lLLVMRuntimeDyld -lLLVMObject -lLLVMMCParser -lLLVMCodeGen -lLLVMTarget -lLLVMScalarOpts -lLLVMInstCombine -lLLVMInstrumentation -lLLVMTransformUtils -lLLVMMC -lLLVMBitWriter -lLLVMBitReader -lLLVMAnalysis -lLLVMProfileData -lLLVMCore -lLLVMSupport -o build/lib.macosx-10.4-x86_64-2.7-pydebug/jit.so')

setup_package();