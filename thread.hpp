
#ifndef Py_THREAD_H
#define Py_THREAD_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Interpreter.h"

#include "llvmjit.hpp"

#include <pthread.h>

typedef long long (*jit_function)(void** inputs, void **outputs, long long start, long long end); 

typedef llvm::Value (*gencode_nullary_function)(llvm::IRBuilder<>& builder, llvm::LLVMContext& context);
typedef llvm::Value (*gencode_unary_function)(llvm::IRBuilder<>& builder, llvm::LLVMContext& context, llvm::Value *inp);
typedef llvm::Value (*gencode_binary_function)(llvm::IRBuilder<>& builder, llvm::LLVMContext& context, llvm::Value *left, llvm::Value *right);

class Thread {
public:
    // thread index
    ssize_t index;
    // number of functions? not sure why I added this
    ssize_t functions;
    // LLVM Context of the thread
    llvm::LLVMContext context;
    // IRBuilder for the context
    llvm::IRBuilder<> builder;
    // JIT used by the thread
    std::unique_ptr<LLVMJIT> jit;
    // thread id
    pthread_t thread;

    Thread(void);
};

class JITFunction {
public:
    Pipeline *pipeline;
    LLVMJIT *jit;
    ModuleHandleT handle;
    void **inputs;
    void **outputs;
    ssize_t references;
    ssize_t size;
    jit_function function;
    base_function base;
};

Thread* CreateThread();
void DestroyThread(Thread *thread);

#endif /* Py_THREAD_H */
