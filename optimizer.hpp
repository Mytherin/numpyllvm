


#ifndef Py_LLVMOPTIMIZER_H
#define Py_LLVMOPTIMIZER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

#include "llvmjit.hpp"

std::unique_ptr<llvm::legacy::FunctionPassManager>
CreatePassManager(llvm::Module *module, LLVMJIT* jit);

#endif /* Py_LLVMOPTIMIZER_H */
