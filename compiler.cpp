
#include "compiler.hpp"
#include "initializers.hpp"
#include "llvmjit.hpp"
#include "optimizer.hpp"

#include <iostream>

using namespace llvm;

static JITFunction* 
CreateJITFunction(Thread *thread, Pipeline *pipeline) {
    JITFunction *jf = (JITFunction*) calloc(1, sizeof(JITFunction));
    jf->pipeline = pipeline;
    return jf;
}

void 
JITFunctionDECREF(JITFunction *f) {
    // TODO: This needs to be thread safe
    printf("DECREF JIT function.\n");
    f->references--;
    if (f->references <= 0) {
        printf("Successfully executed pipeline: %s\n", f->pipeline->name);
        printf("Destroying JIT function (%p).\n", f->pipeline);
        assert(f->pipeline);
        // after the function is evaluated, all output thunks have been evaluated
        f->pipeline->evaluated = true;
        int i = 0;
        for (auto it = f->pipeline->outputData->objects.begin(); it != f->pipeline->outputData->objects.end(); it++) {
            char *array_data = (char*) PyArray_DATA(it->source->object->storage);
            for(int j = 1; j < f->thread_count; j++) {
                if (f->output_ends[i][j - 1] != f->output_starts[j]) {
                    ssize_t element_size = sizeof(long long); // FIXME
                    // the start and end of the different thread outputs do not align
                    // so we have to memmove them
                    memmove(array_data + f->output_ends[i][j - 1] * element_size, 
                        array_data + f->output_starts[j] * element_size, 
                        (f->output_ends[i][j] - f->output_starts[j]) * element_size);
                    f->output_ends[i][j] = f->output_ends[i][j - 1] + (f->output_ends[i][j] - f->output_starts[j]);
                }
            }
            PyArray_DIMS(it->source->object->storage)[0] = f->output_ends[i][f->thread_count - 1];
            it->source->object->evaluated = true;
            printf("Evaluated object at %p\n", it->source->object);
            free(f->output_ends[i]);
            i++;
        }
        if (f->output_starts) {
            free(f->output_starts);
            f->output_starts = NULL;
        }
        if (f->output_ends) {
            free(f->output_ends);
            f->output_ends = NULL;
        }
        // if anyone was waiting for the pipeline to be evaluated, notify them
        if (f->pipeline->semaphore) {
            semaphore_increment(&f->pipeline->semaphore);
        }
        if (f->pipeline->parent) {
            // now schedule the parent of this pipeline, if it has any
            ScheduleExecution(f->pipeline->parent);
        }

        // cleanup
        if (f->inputs) free(f->inputs);
        if (f->outputs) free(f->outputs);
        if (f->jit) f->jit->removeModule(f->handle);
        DestroyPipeline(f->pipeline);
        free(f);
    }
}

void 
ExecuteFunction(JITFunction *f, size_t start, size_t end, int thread_nr) {
    printf("Execute function %zu-%zu\n", start, end);
    if (f->function) {
        long long *output_sizes = (long long*) malloc(f->pipeline->outputData->objects.size() * sizeof(long long));
        // compilable function
        f->function(f->outputs, f->inputs, start, end, output_sizes);
        f->output_starts[thread_nr] = start;
        for(int i = 0; i < f->pipeline->outputData->objects.size(); i++) {
            f->output_ends[i][thread_nr] = output_sizes[i];
        }
        free(output_sizes);
    } else {
        PyArrayObject *result = NULL;
        assert(f->base);
        switch(f->pipeline->inputData->objects.size()) {
            case 0:
                result = ((base_function_nullary) f->base)();
                break;
            case 1: {
                DataElement *element = &f->pipeline->children->child->outputData->objects[0];
                printf("--%s Inputs--\n[ %p ]\n", f->pipeline->name, element->source->data);
                result = ((base_function_unary) f->base)(element->source->object->storage);
                break;
            }
            case 2: {
                DataElement *a1 = &f->pipeline->children->child->outputData->objects[0];                
                DataElement *a2 = &f->pipeline->children->child->outputData->objects[1];
                result = ((base_function_binary) f->base)(a1->source->object->storage, a2->source->object->storage);
                break;
            }
            default:
                fprintf(stderr, "Expected 0, 1 or 2 parameters.\n");
                assert(0);
                return;
        }
        assert(result);
        f->pipeline->outputData->objects[0].source->object->storage = result;
        f->pipeline->outputData->objects[0].source->size = PyArray_SIZE(result);
        f->pipeline->outputData->objects[0].source->data = PyArray_DATA(result);
        f->pipeline->outputData->objects[0].source->type = PyArray_TYPE(result);
    }
}

static Value*
getLLVMConstant(LLVMContext& context, DataElement *entry, Type *type) {
    switch(entry->source->type) {
        case NPY_BOOL:
        case NPY_INT8:
            return ConstantInt::get(type, *(char*) entry->source->data, true);
        case NPY_INT16:
            return ConstantInt::get(type, *(short*) entry->source->data, true);
        case NPY_INT32:
            return ConstantInt::get(type, *(int*) entry->source->data, true);
        case NPY_INT64:
            return ConstantInt::get(type, *(long long*) entry->source->data, true);
        case NPY_UINT8:
            return ConstantInt::get(type, *(unsigned char*) entry->source->data, false);
        case NPY_UINT16:
            return ConstantInt::get(type, *(unsigned short*) entry->source->data, false);
        case NPY_UINT32:
            return ConstantInt::get(type, *(unsigned int*) entry->source->data, false);
        case NPY_UINT64:
            return ConstantInt::get(type, *(unsigned long long*) entry->source->data, false);
        case NPY_FLOAT16:
            return NULL; //todo
        case NPY_FLOAT32:
            return ConstantFP::get(type, *(float*) entry->source->data);
        case NPY_FLOAT64:
            return ConstantFP::get(type, *(double*) entry->source->data);
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

static void
PerformInitialization(JITInformation& info, Operation *op) {
    if (op->Type() == OPTYPE_binop) {
        BinaryOperation *binop = (BinaryOperation*) op;
        PerformInitialization(info, binop->LHS);
        PerformInitialization(info, binop->RHS);
        if (binop->operation->gencode.initialize) {
            ((initialize_gencode)binop->operation->gencode.initialize)(info, op);
        }
    } else if (op->Type() == OPTYPE_unop) {
        UnaryOperation *unop = (UnaryOperation*) op;
        PerformInitialization(info, unop->LHS);
        if (unop->operation->gencode.initialize) {
            ((initialize_gencode)unop->operation->gencode.initialize)(info, op);
        }
    }
}

static Value*
PerformOperation(JITInformation& info, IRBuilder<>& builder, LLVMContext& context, Operation *op, DataBlock *inputData, DataBlock *outputData) {
    Value *v = NULL;
    if (op->Type() == OPTYPE_obj || op->Type() == OPTYPE_pipeline) {
        bool found = false;
        for(auto it = inputData->objects.begin(); it != inputData->objects.end(); it++) {
            if (it->operation == op) {
                Type *column_type = getLLVMType(context, it->source->type);
                if (it->source->size == 1) {
                    v = getLLVMConstant(context, &*it, column_type);
                } else {
                    AllocaInst *address = (AllocaInst*) it->alloca_address;
                    LoadInst *value_address = builder.CreateLoad(address);
                    Value *colptr = builder.CreateGEP(column_type, value_address, info.index, "column[i]");
                    v = builder.CreateLoad(colptr);
                }
                found = true;
                break;
            }
        }
        if (!found) {
            printf("Column/Pipeline operation found, but not found in inputData.\n");
            return NULL;
        }
    } else if (op->Type() == OPTYPE_binop) {
        BinaryOperation *binop = (BinaryOperation*) op;
        Value *l = PerformOperation(info, builder, context, binop->LHS, inputData, outputData);
        Value *r = PerformOperation(info, builder, context, binop->RHS, inputData, outputData);
        if (l == NULL || r == NULL) {
            return NULL;
        }
        assert(binop->operation->gencode.gencode);
        v = ((binary_gencode) binop->operation->gencode.gencode)(info, op, l, r);
    } else if (op->Type() == OPTYPE_unop) {
        UnaryOperation *unop = (UnaryOperation*) op;
        Value *l = PerformOperation(info, builder, context, unop->LHS, inputData, outputData);
        if (l == NULL) {
            return NULL;
        }
        assert(unop->operation->gencode.gencode);
        v = ((unary_gencode) unop->operation->gencode.gencode)(info, op, l);
    } else {
        printf("Unrecognized operation!");
        return NULL;
    }
    if (op->materialize) {
        for(auto it = outputData->objects.begin(); it != outputData->objects.end(); it++) {
            if (it->operation == op) {
                Type *column_type = getLLVMType(context, it->source->type);
                AllocaInst *address = (AllocaInst*) it->alloca_address;
                LoadInst *result_address = builder.CreateLoad(address);
                Value *resptr = builder.CreateGEP(column_type, result_address, info.index, "result[i]");
                Value *stored_value = v;
                it->index_addr = info.index_addr;
                if (v->getType() != column_type) {
                    // types do not match, have to cast
                    // FIXME floating point casts
                    stored_value = builder.CreateIntCast(v, column_type, true);
                }
                builder.CreateStore(stored_value, resptr, "result[i] = value");
                return v;
            }
        }
        fprintf(stderr, "Result of operation has to be materialized, but not found in outputData");
        return NULL;
    }
    return v;
}

bool
CompilableOperation(Operation *operation) {
    switch(operation->Type()) {
        case OPTYPE_nullop:
            return ((NullaryOperation*)operation)->operation->gencode.gencode != NULL;
        case OPTYPE_unop:
            return ((UnaryOperation*)operation)->operation->gencode.gencode != NULL;
        case OPTYPE_binop:
            return ((BinaryOperation*)operation)->operation->gencode.gencode != NULL;
        default:
            return true;
    }
}

JITFunction*
GenerateBaseFunction(Pipeline *pipeline, Thread *thread) {
    JITFunction *jf = CreateJITFunction(thread, pipeline);
    Operation *operation = pipeline->operation;
    jf->size = 0;
    jf->function = NULL;
    jf->jit = NULL;

    switch(operation->Type()) {
        case OPTYPE_nullop:
            jf->base = (base_function) ((NullaryOperation*)operation)->operation->gencode.base;
            break;
        case OPTYPE_unop:
            jf->base = (base_function) ((UnaryOperation*)operation)->operation->gencode.base;
            break;
        case OPTYPE_binop:
            jf->base = (base_function) ((BinaryOperation*)operation)->operation->gencode.base;
            break;
        default:
            return NULL;
    }
    return jf;
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
    size_t function_arg_count = 5;
    size_t arg_count = input_count + output_count + 3;
    size_t start_addr = input_count + output_count;
    size_t end_addr = start_addr + 1;
    size_t result_sizes_addr = end_addr + 1;
    IRBuilder<> *builder = &thread->builder;
    auto passmanager = CreatePassManager(module, thread->jit.get());

    module->setDataLayout(thread->jit->getTargetMachine().createDataLayout());

    Type *int8_tpe = Type::getInt8Ty(thread->context);
    Type *int8ptr_tpe = PointerType::get(int8_tpe, 0);
    Type *int8ptrptr_tpe = PointerType::get(int8ptr_tpe, 0);
    Type *int64_tpe = Type::getInt64Ty(thread->context);
    Type *int64ptr_tpe = PointerType::get(int64_tpe, 0);

    JITInformation info;

    // arguments of the function
    // the arguments are (void **result, void** inputs, size_t start, size_t end);
    // note that we don't actually use void**, we use int8**, because LLVM does not support void pointers
    std::vector<Type*> arguments(function_arg_count);
    i = 0;
    arguments[i++] = int8ptrptr_tpe;  // void** results
    arguments[i++] = int8ptrptr_tpe;  // void** inputs
    arguments[i++] = int64_tpe;       // size_t start
    arguments[i++] = int64_tpe;       // size_t end
    arguments[i++] = int64ptr_tpe;  // size_t* result_sizes
    assert(i == function_arg_count);

    /*for(auto inputs = pipeline->inputData->objects.begin(); inputs != pipeline->inputData->objects.end(); inputs++, i++) {
        arguments[i] = PointerType::get(getLLVMType(thread->context, inputs->type), 0);
    }
    for(auto outputs = pipeline->outputData->objects.begin(); outputs != pipeline->outputData->objects.end(); outputs++, i++) {
        arguments[i] = PointerType::get(getLLVMType(thread->context, outputs->type), 0);
    }*/

    // create the LLVM function
    FunctionType *prototype = FunctionType::get(int64_tpe, arguments, false);
    Function *function = Function::Create(prototype, GlobalValue::ExternalLinkage, fname, module);
    function->setCallingConv(CallingConv::C);

    // create the basic blocks
    BasicBlock *loop_entry = BasicBlock::Create(thread->context, "entry", function, 0);
    BasicBlock *loop_cond  = BasicBlock::Create(thread->context, "for.cond", function, 0);
    BasicBlock *loop_body  = BasicBlock::Create(thread->context, "for.body", function, 0);
    BasicBlock *loop_inc   = BasicBlock::Create(thread->context, "for.inc", function, 0);
    BasicBlock *loop_end   = BasicBlock::Create(thread->context, "for.end", function, 0);

    info.builder = &thread->builder;
    info.context = &thread->context;
    info.function = function;
    info.loop_entry = loop_entry;
    info.loop_cond = loop_cond;
    info.loop_body = loop_body;
    info.loop_inc = loop_inc;
    info.loop_end = loop_end;
    info.current = loop_body;

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
    argument_names[i++] = "result_sizes";
#endif

    std::vector<AllocaInst*> argument_addresses(arg_count);
    builder->SetInsertPoint(loop_entry);
    {
        // allocate space for the arguments
        auto args = function->arg_begin();
        i = 0;
        for(auto outputs = pipeline->outputData->objects.begin(); outputs != pipeline->outputData->objects.end(); outputs++, i++) {
            Type *column_type = PointerType::get(getLLVMType(thread->context, outputs->source->type), 0);
            Value *voidptrptr = builder->CreateGEP(int8ptr_tpe, &*args, ConstantInt::get(int64_tpe, i, true));
            Value *voidptr = builder->CreateLoad(voidptrptr, "voidptr");
            Value *columnptr = builder->CreatePointerCast(voidptr, column_type);
            argument_addresses[i] = builder->CreateAlloca(column_type, nullptr, argument_names[i]);
            builder->CreateStore(columnptr, argument_addresses[i]);
            outputs->alloca_address = (void*) argument_addresses[i];
            if (size == 0 || size == 1) {
                assert(outputs->source->size >= 0);
                size = outputs->source->size;
            }
            assert(size == outputs->source->size || outputs->source->size == 1);
        }
        args++;
        for(auto inputs = pipeline->inputData->objects.begin(); inputs != pipeline->inputData->objects.end(); inputs++, i++) {
            Type *column_type = PointerType::get(getLLVMType(thread->context, inputs->source->type), 0);
            Value *voidptrptr = builder->CreateGEP(int8ptr_tpe, &*args, ConstantInt::get(int64_tpe, i - output_count, true));
            Value *voidptr = builder->CreateLoad(voidptrptr, "voidptr");
            Value *columnptr = builder->CreatePointerCast(voidptr, column_type);
            argument_addresses[i] = builder->CreateAlloca(column_type, nullptr, argument_names[i]);
            builder->CreateStore(columnptr, argument_addresses[i]);
            inputs->alloca_address = (void*) argument_addresses[i];
            if (size == 0 || size == 1) {
                assert(inputs->source->size >= 0);
                size = inputs->source->size;
            }
            assert(size == inputs->source->size || inputs->source->size == 1);
        }
        args++;
        argument_addresses[i] = builder->CreateAlloca(arguments[2], nullptr, argument_names[i]);
        builder->CreateStore(&*args, argument_addresses[i]);
        args++; i++;
        argument_addresses[i] = builder->CreateAlloca(arguments[3], nullptr, argument_names[i]);
        builder->CreateStore(&*args, argument_addresses[i]);
        args++; i++;
        argument_addresses[i] = builder->CreateAlloca(arguments[4], nullptr, argument_names[i]);
        builder->CreateStore(&*args, argument_addresses[i]);
        args++; i++;
        assert(args == function->arg_end());
        assert(i == arg_count);
        info.index_addr = argument_addresses[start_addr];

        PerformInitialization(info, pipeline->operation);

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
        info.index = index;
        info.index_addr = argument_addresses[start_addr];
        // perform the computation over the given index
        // we don't use the return value because the final assignment has already taken place
        Value *v = PerformOperation(info, thread->builder, thread->context, pipeline->operation, pipeline->inputData, pipeline->outputData);
        if (v == NULL) {
            // failed to perform operation
            printf("Failed to compile pipeline %s\n", pipeline->name);
            return NULL;
        }

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
        // return the output size of each of the columns
        int i = 0;
        Value *result_sizes = builder->CreateLoad(argument_addresses[result_sizes_addr], "result_sizes[]");
        for(auto it = pipeline->outputData->objects.begin(); it != pipeline->outputData->objects.end(); it++) {
            Value* output_count;
            if (it->index_addr) {
                output_count = builder->CreateLoad((Value*) it->index_addr, "count");
            } else {
                output_count = ConstantInt::get(int64_tpe, 1, true);
            }
            Value *output_addr = builder->CreateGEP(int64_tpe, result_sizes, ConstantInt::get(int64_tpe, i, true));
            builder->CreateStore(output_count, output_addr);
            i++;
        }

        builder->CreateRet(ConstantInt::get(int64_tpe, 0, true));
    }

#ifndef _NOTDEBUG
    verifyFunction(*function);
    verifyModule(*module);
#endif

    //printf("LLVM for pipeline %s\n", pipeline->name);
    module->dump();
    passmanager->run(*function);
    // dump generated LLVM code
    //module->dump();

    auto handle = thread->jit->addModule(std::move(owner));

    jit_function compiled_function = (jit_function) thread->jit->findSymbol(fname).getAddress();
    if (!compiled_function) {
        printf("Error creating function.\n");
        return NULL;
    }

    JITFunction *jf = CreateJITFunction(thread, pipeline);
    jf->size = size;
    jf->function = compiled_function;
    jf->jit = thread->jit.get();
    jf->handle = handle;

    assert(jf->function);
    return jf;
}

void 
initialize_compiler(void) {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    import_array();
    import_umath();
}
