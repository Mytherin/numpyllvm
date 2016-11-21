
#include "thunk.hpp"
#include "initializers.hpp"
#include <limits.h>

#include "compiler.hpp"

using namespace llvm;

static PyObject *
_thunk_evaluate(PyThunkObject *self, PyObject *args) {
    (void) args;
	if (PyThunk_Evaluate(self) == NULL) {
		return NULL;
	}
    PyThunk_Evaluate(self);
    Py_RETURN_NONE;
}

static PyObject *
_thunk_isevaluated(PyThunkObject *self, PyObject *args) {
    (void) args;
    if (PyThunk_IsEvaluated(self)) {
    	Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
_thunk_asarray(PyThunkObject *self, PyObject *args) {
    (void) args;
    PyObject *arr = PyThunk_AsArray((PyObject*) self);
    Py_INCREF(arr);
    return arr;
}

static PyArrayObject* 
_thunk_inline_sort(PyArrayObject* input) {
    PyArrayObject *output = (PyArrayObject*) PyArray_FromArray(input, PyArray_DESCR(input), NPY_ARRAY_ENSURECOPY);
    PyArray_Sort(output, 0, NPY_QUICKSORT);
    return output;
}

static PyObject *
_thunk_sort(PyThunkObject *self, PyObject *unused) {
    (void) unused;
    if (!PyThunk_CheckExact(self)) {
        PyErr_SetString(PyExc_TypeError, "Expected a thunk object as parameter.");
        return NULL;
    }
    PyThunkObject *left = (PyThunkObject*) self;
    ThunkOperation *op = ThunkOperation_FromUnary(
        (PyObject*) self, 
        optype_fullbreaker, 
        NULL, 
        (base_function_unary) _thunk_inline_sort, 
        strdup("sort"));
    return PyThunk_FromOperation(op, 
        default_cardinality_function, 
        cardinality_exact, 
        left->type);
}


static ssize_t
aggregate_cardinality_function(ssize_t *inputs) {
    return 1;
}

Value *llvm_generate_max(JITInformation& inp, Operation *op, Value *l) {
    assert(((UnaryOperation*) op)->extra);
    BasicBlock *conditional_branch = BasicBlock::Create(*inp.context, "branch", inp.function, 0);
    LoadInst *current_max = inp.builder->CreateLoad((Value*) ((UnaryOperation*) op)->extra, "current_max");
    Value *greater_than = inp.builder->CreateICmpSGT(l, current_max, "l >= current_max");
    inp.builder->CreateCondBr(greater_than, conditional_branch, inp.loop_inc);
    inp.builder->SetInsertPoint(conditional_branch);
    inp.current = conditional_branch;
    inp.builder->CreateStore(l, (Value*) ((UnaryOperation*) op)->extra);
    inp.index = inp.builder->CreateLoad(inp.thread_addr);
    inp.index_addr = inp.thread_addr;
    return inp.builder->CreateLoad((Value*) ((UnaryOperation*) op)->extra, "current_max");
}

void llvm_initialize_max(JITInformation& info, Operation *op) {
    Type *int64_tpe = Type::getInt64Ty(*info.context);
    // create the max
    AllocaInst *max_value = info.builder->CreateAlloca(int64_tpe, nullptr);
    // we start at the minimal possible value
    info.builder->CreateStore(ConstantInt::get(int64_tpe, LLONG_MIN, true), max_value);
    ((UnaryOperation*)op)->extra = (void*) max_value;
}

static PyObject *
_thunk_max(PyThunkObject *self, PyObject *unused) {
    (void) unused;
    if (!PyThunk_CheckExact(self)) {
        PyErr_SetString(PyExc_TypeError, "Expected a thunk object as parameter.");
        return NULL;
    }
    ThunkOperation *op = ThunkOperation_FromUnary(
        (PyObject*) self, 
        optype_parallelbreaker, 
        (void*) llvm_generate_max, 
        NULL, 
        strdup("max"));
    op->gencode.initialize = (void*) llvm_initialize_max;
    return PyThunk_FromOperation(op, 
        aggregate_cardinality_function, 
        cardinality_exact, 
        self->type);
}


Value *llvm_generate_sum(JITInformation& inp, Operation *op, Value *l) {
    assert(((UnaryOperation*) op)->extra);
    LoadInst *current_sum = inp.builder->CreateLoad((Value*) ((UnaryOperation*) op)->extra, "current_max");
    Value *updated_sum = inp.builder->CreateAdd(current_sum, l, "sum");
    inp.builder->CreateStore(updated_sum, (Value*) ((UnaryOperation*) op)->extra);
    inp.index = inp.builder->CreateLoad(inp.thread_addr);
    inp.index_addr = inp.thread_addr;
    return updated_sum;
}

void llvm_initialize_sum(JITInformation& info, Operation *op) {
    // create the max
    Type *int64_tpe = Type::getInt64Ty(*info.context);
    AllocaInst *max_value = info.builder->CreateAlloca(int64_tpe, nullptr);
    // we start at the minimal possible value
    info.builder->CreateStore(ConstantInt::get(int64_tpe, 0, true), max_value);
    ((UnaryOperation*)op)->extra = (void*) max_value;
}

void llvm_finalize_sum_data(JITFunction *f, DataSource *source, int thread_count) {
    // to finalize after all the threads are done we sum over 
    PyArrayObject *storage = source->object->storage;      
    npy_intp elements[] = { (npy_intp) 1 };  
    source->object->storage = (PyArrayObject*) PyArray_ZEROS(1, elements, source->type, 0);
    char *source_data = (char*) PyArray_DATA(storage);
    char *dest_data = (char*) PyArray_DATA(source->object->storage);
    for(int j = 0; j < f->thread_count; j++) {
        switch(source->type) {
            case NPY_INT64:
                ((npy_int64*) dest_data)[0] += ((npy_int64*) source_data)[j];
                break;
            default:
                assert(0); //FIXME
        }
    }
    source->data = dest_data;
    Py_DECREF(storage);
}

void llvm_initialize_sum_data(DataSource *source, int thread_count) {
    if (source->object->storage != NULL) {
        Py_XDECREF(source->object->storage);
    }
    // we create an array with one entry per thread (to store the intermediate results)
    npy_intp elements[] = { (npy_intp) thread_count };
    // FIXME: set to PyArray_EMPTY
    source->object->storage = (PyArrayObject*) PyArray_ZEROS(1, elements, source->type, 0);
    source->data = PyArray_DATA(source->object->storage);
}

static PyObject *
_thunk_sum(PyThunkObject *self, PyObject *unused) {
    (void) unused;
    if (!PyThunk_CheckExact(self)) {
        PyErr_SetString(PyExc_TypeError, "Expected a thunk object as parameter.");
        return NULL;
    }
    ThunkOperation *op = ThunkOperation_FromUnary(
        (PyObject*) self, 
        optype_parallelbreaker, 
        (void*) llvm_generate_sum, 
        NULL, 
        strdup("sum"));
    op->gencode.initialize = (void*) llvm_initialize_sum;
    op->gencode.initialize_data = (initialize_data_function) llvm_initialize_sum_data;
    op->gencode.finalize_data = (finalize_data_function) llvm_finalize_sum_data;
    return PyThunk_FromOperation(op, 
        aggregate_cardinality_function, 
        cardinality_exact, 
        self->type);
}

struct PyMethodDef thunk_methods[] = {
    {"evaluate", (PyCFunction)_thunk_evaluate, METH_NOARGS,"evaluate() => "},
    {"isevaluated", (PyCFunction)_thunk_isevaluated, METH_NOARGS,"isevaluated() => "},
    {"asnumpyarray", (PyCFunction)_thunk_asarray, METH_NOARGS,"asnumpyarray() => "},
    {"sort", (PyCFunction)_thunk_sort, METH_NOARGS,"sort() => "},
    {"max", (PyCFunction)_thunk_max, METH_NOARGS,"max() => "},
    {"sum", (PyCFunction)_thunk_sum, METH_NOARGS,"sum() => "},
    {NULL}  /* Sentinel */
};

void initialize_thunk_methods(void) {
    import_array();
    import_umath();
}
