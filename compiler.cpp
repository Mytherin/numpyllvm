
#include "compiler.hpp"
#include "initializers.hpp"

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

#include <iostream>

using namespace llvm;

static JITFunction* 
CreateJITFunction(Thread *thread, Pipeline *pipeline) {
    JITFunction *jf = (JITFunction*) malloc(sizeof(JITFunction));
    jf->pipeline = pipeline;
    jf->execution_engine = NULL;
    jf->function = NULL;
    jf->references = 0;
    jf->size = 0;
    jf->inputs = NULL;
    jf->outputs = NULL;
    return jf;
}

void 
JITFunctionDECREF(JITFunction *f) {
    // TODO: This needs to be thread safe
    f->references--;
    if (f->references <= 0) {
        assert(f->pipeline);
        assert(f->execution_engine);
        // this pipeline has been evaluated completely, schedule all the children
        PipelineNode *node = f->pipeline->children;
        while (node) {
          SchedulePipeline(node->child);
          node = node->next;
        }
        // after the function is evaluated, all output thunks have been evaluated
        for (auto it = f->pipeline->outputData->objects.begin(); it != f->pipeline->outputData->objects.end(); it++) {
            it->object->evaluated = true;
        }
        // cleanup
        if (f->inputs) free(f->inputs);
        if (f->outputs) free(f->outputs);
        DestroyPipeline(f->pipeline);
        delete f->execution_engine;
        free(f);
    }
}

void 
ExecuteFunction(JITFunction *f, size_t start, size_t end) {
    size_t i;
    size_t input_count = f->pipeline->inputData->objects.size();
    size_t output_count = f->pipeline->outputData->objects.size();
    std::vector<GenericValue> params(input_count + output_count + 2);
    for(i = 0; i < input_count; i++) {
        params[i].PointerVal = f->inputs[i];
    }
    for(i = 0; i < output_count; i++) {
        params[input_count + i].PointerVal = f->outputs[i];
    }
    params[input_count + output_count].IntVal = APInt(64, start, true);
    params[input_count + output_count + 1].IntVal = APInt(64, end, true);
    f->execution_engine->runFunction(f->function, params);
}

static Value*
getLLVMConstant(LLVMContext& context, DataElement *entry, Type *type) {
    switch(entry->type) {
        case NPY_BOOL:
        case NPY_INT8:
            return ConstantInt::get(type, *(char*) entry->data, true);
        case NPY_INT16:
            return ConstantInt::get(type, *(short*) entry->data, true);
        case NPY_INT32:
            return ConstantInt::get(type, *(int*) entry->data, true);
        case NPY_INT64:
            return ConstantInt::get(type, *(long long*) entry->data, true);
        case NPY_UINT8:
            return ConstantInt::get(type, *(unsigned char*) entry->data, false);
        case NPY_UINT16:
            return ConstantInt::get(type, *(unsigned short*) entry->data, false);
        case NPY_UINT32:
            return ConstantInt::get(type, *(unsigned int*) entry->data, false);
        case NPY_UINT64:
            return ConstantInt::get(type, *(unsigned long long*) entry->data, false);
        case NPY_FLOAT16:
            return NULL; //todo
        case NPY_FLOAT32:
            return ConstantFP::get(type, *(float*) entry->data);
        case NPY_FLOAT64:
            return ConstantFP::get(type, *(double*) entry->data);
    }
    // non-numeric type, not supported
    return NULL;
}

static Type*
getLLVMType(LLVMContext& context, int numpy_type) {
    switch(numpy_type) {
        case NPY_BOOL:
        case NPY_INT8:
            return Type::getInt8Ty(context);
        case NPY_INT16:
            return Type::getInt16Ty(context);
        case NPY_INT32:
            return Type::getInt32Ty(context);
        case NPY_INT64:
            return Type::getInt64Ty(context);
        case NPY_UINT8:
        case NPY_UINT16:
        case NPY_UINT:
        case NPY_ULONGLONG:
            return NULL; // todo: unsigned types
        case NPY_FLOAT16:
            return Type::getHalfTy(context);
        case NPY_FLOAT32:
            return Type::getFloatTy(context);
        case NPY_FLOAT64:
            return Type::getDoubleTy(context);
    }
    // non-numeric type, not supported
    return NULL;
}

static 
Value *PerformOperation(IRBuilder<>& builder, LLVMContext& context, Operation *op, Value *index, DataBlock *inputData, DataBlock *outputData) {
    Value *v = NULL;
    if (op->Type() == OPTYPE_obj || op->Type() == OPTYPE_pipeline) {
        bool found = false;
        for(auto it = inputData->objects.begin(); it != inputData->objects.end(); it++) {
            if (it->operation == op) {
                Type *column_type = getLLVMType(context, it->type);
                if (it->size == 1) {
                    v = getLLVMConstant(context, &*it, column_type);
                } else {
                    AllocaInst *address = (AllocaInst*) it->alloca_address;
                    LoadInst *value_address = builder.CreateLoad(address);
                    Value *colptr = builder.CreateGEP(column_type, value_address, index, "column[i]");
                    v = builder.CreateLoad(colptr);
                }
                found = true;
                break;
            }
        }
        if (!found) {
            printf("Column/Pipeline operation found, but not found in inputData");
            return NULL;
        }
    } else if (op->Type() == OPTYPE_binop) {
        BinaryOperation *binop = (BinaryOperation*) op;
        Value *l = PerformOperation(builder, context, binop->LHS, index, inputData, outputData);
        Value *r = PerformOperation(builder, context, binop->RHS, index, inputData, outputData);
        v = builder.CreateMul(l, r);
    } else if (op->Type() == OPTYPE_unop) {
        v = NULL;
    } else {
        printf("Unrecognized operation!");
        return NULL;
    }
    if (op->result_object != NULL) {
        for(auto it = outputData->objects.begin(); it != outputData->objects.end(); it++) {
            if (it->operation == op) {
                Type *column_type = getLLVMType(context, it->type);
                AllocaInst *address = (AllocaInst*) it->alloca_address;
                LoadInst *result_address = builder.CreateLoad(address);
                Value *resptr = builder.CreateGEP(column_type, result_address, index, "result[i]");
                builder.CreateStore(v, resptr, "result[i] = value");
                return v;
            }
        }
        printf("Result object specified, but not found in outputData");
        return NULL;
    }
    return v;
}

JITFunction* 
CompilePipeline(Pipeline *pipeline, Thread *thread) {
    size_t i = 0;
    size_t size = 0;
    std::unique_ptr<Module> owner = make_unique<Module>("PipelineFunction", thread->context);
    Module *module = owner.get();
    std::string fname = std::string("PipelineFunction") + std::to_string(thread->functions++);
    size_t input_count = pipeline->inputData->objects.size();
    size_t output_count = pipeline->outputData->objects.size();
    size_t arg_count = input_count + output_count + 2;
    size_t start_addr = input_count + output_count;
    size_t end_addr = start_addr + 1;
    IRBuilder<> *builder = &thread->builder;

    Type *int64_tpe = Type::getInt64Ty(thread->context);
    Type *void_tpe = Type::getVoidTy(thread->context);

    // arguments of the function
    // the arguments are (void *input_1, ..., void *input_n, void *output_1, ..., void *output_n, size_t start, size_t end);
    // note that we don't actually use void*, we use the type of the input, but this type can vary
    std::vector<Type*> arguments(arg_count);
    i = 0;
    for(auto inputs = pipeline->inputData->objects.begin(); inputs != pipeline->inputData->objects.end(); inputs++, i++) {
        arguments[i] = PointerType::get(getLLVMType(thread->context, inputs->type), 0);
    }
    for(auto outputs = pipeline->outputData->objects.begin(); outputs != pipeline->outputData->objects.end(); outputs++, i++) {
        arguments[i] = PointerType::get(getLLVMType(thread->context, outputs->type), 0);
    }
    arguments[i++] = int64_tpe;    // size_t start
    arguments[i++] = int64_tpe;    // size_t end
    assert(i == arg_count);

    // create the LLVM function
    FunctionType *prototype = FunctionType::get(void_tpe, arguments, false);
    Function *function = Function::Create(prototype, GlobalValue::ExternalLinkage, fname, module);
    function->setCallingConv(CallingConv::C);

    // create the basic blocks
    BasicBlock *loop_entry = BasicBlock::Create(thread->context, "entry", function, 0);
    BasicBlock *loop_cond  = BasicBlock::Create(thread->context, "for.cond", function, 0);
    BasicBlock *loop_body  = BasicBlock::Create(thread->context, "for.body", function, 0);
    BasicBlock *loop_inc   = BasicBlock::Create(thread->context, "for.inc", function, 0);
    BasicBlock *loop_end   = BasicBlock::Create(thread->context, "for.end", function, 0);

#ifndef _NOTDEBUG
    // argument names (for debug purposes only)
    std::vector<std::string> argument_names(arg_count);
    i = 0;
    for(auto inputs = pipeline->inputData->objects.begin(); inputs != pipeline->inputData->objects.end(); inputs++, i++) {
        argument_names[i] = std::string("inputs") + std::to_string(i);
    }
    for(auto outputs = pipeline->outputData->objects.begin(); outputs != pipeline->outputData->objects.end(); outputs++, i++) {
        argument_names[i] = std::string("outputs") + std::to_string(i - input_count);
    }
    argument_names[i++] = "start";
    argument_names[i++] = "end";
#endif

    std::vector<AllocaInst*> argument_addresses(arg_count);
    builder->SetInsertPoint(loop_entry);
    {
        // allocate space for the arguments
        i = 0;
        for(auto args = function->arg_begin(); args != function->arg_end(); args++, i++) {
            argument_addresses[i] = builder->CreateAlloca(arguments[i], nullptr, argument_names[i]);
            builder->CreateStore(&*args, argument_addresses[i]);
        }
        i = 0;
        for(auto inputs = pipeline->inputData->objects.begin(); inputs != pipeline->inputData->objects.end(); inputs++, i++) {
            inputs->alloca_address = (void*) argument_addresses[i];
            if (size == 0) {
                size = inputs->size;
            }
            assert(size == inputs->size || inputs->size == 1);
        }
        for(auto outputs = pipeline->outputData->objects.begin(); outputs != pipeline->outputData->objects.end(); outputs++, i++) {
            outputs->alloca_address = (void*) argument_addresses[i];
            if (size == 0) {
                size = outputs->size;
            }
            assert(size == outputs->size || outputs->size == 1);
        }
        builder->CreateBr(loop_cond);
    }

    // for loop condition: index < end
    builder->SetInsertPoint(loop_cond);
    {
        LoadInst *index = builder->CreateLoad(argument_addresses[start_addr], "index");
        LoadInst *end = builder->CreateLoad(argument_addresses[end_addr], "end");
        Value *condition = builder->CreateICmpSLT(index, end, "index < end");
        builder->CreateCondBr(condition, loop_body, loop_end);
    }

    // loop body: perform the computation
    builder->SetInsertPoint(loop_body);
    {
         LoadInst *index = builder->CreateLoad(argument_addresses[start_addr], "index");
        // perform the computation over the given index
        // we don't use the return value because the final assignment has already taken place
        (void) PerformOperation(thread->builder, thread->context, pipeline->operation, index, pipeline->inputData, pipeline->outputData);

        builder->CreateBr(loop_inc);
    }

    // loop increment: index++
    builder->SetInsertPoint(loop_inc);
    {
        LoadInst *index = builder->CreateLoad(argument_addresses[start_addr], "index");
        Value *incremented_index = builder->CreateAdd(index, ConstantInt::get(int64_tpe, 1, true), "index++");
        builder->CreateStore(incremented_index, argument_addresses[start_addr]);

        builder->CreateBr(loop_cond);
    }

    // loop end: return; (nothing happens here because we have no return value)
    builder->SetInsertPoint(loop_end);
    {
        builder->CreateRetVoid();
    }

    // dump generated LLVM code
    module->dump();
    
    std::string msg;
    ExecutionEngine *engine = EngineBuilder(std::move(owner)).setErrorStr(&msg).create();
    if (!engine) {
        std::cout << "Error: " << msg;
        return NULL;
    }
    engine->finalizeObject();

    JITFunction *jf = CreateJITFunction(thread, pipeline);
    jf->size = size;
    jf->execution_engine = engine;
    jf->function = function;
    assert(jf->function);
    return jf;
}

void initialize_compiler(void) {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    import_array();
    import_umath();
}
