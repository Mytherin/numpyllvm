



#ifndef Py_Compiler_H
#define Py_Compiler_H

#include "parser.hpp"
#include "scheduler.hpp"
#include "thread.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MathExtras.h>

typedef struct {
    llvm::IRBuilder<> *builder;
    llvm::LLVMContext *context;
    llvm::Function *function;
    llvm::BasicBlock *loop_entry;
    llvm::BasicBlock *loop_cond;
    llvm::BasicBlock *loop_body; 
    llvm::BasicBlock *loop_inc; 
    llvm::BasicBlock *loop_end;
    llvm::BasicBlock *current;
    llvm::Value *index;
    llvm::Value *index_addr;
} JITInformation;

typedef void (*initialize_gencode)(JITInformation& info, Operation *op);
typedef llvm::Value* (*binary_gencode)(JITInformation& info, Operation *op, llvm::Value *left, llvm::Value *right);

class JITFunction;

JITFunction* CompilePipeline(Pipeline *pipeline, Thread *thread);
JITFunction* GenerateBaseFunction(Pipeline *pipeline, Thread *thread);
void JITFunctionDECREF(JITFunction *f);
void ExecuteFunction(JITFunction *f, size_t start, size_t end);
bool CompilableOperation(Operation *operation);

#endif
