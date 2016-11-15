
#include "thunk.hpp"
#include "debug_printer.hpp"

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

using namespace llvm;

Value *llvm_generate_gt(IRBuilder<>& builder, Value *l, Value *r) {
    return builder.CreateICmpSGT(l, r);
}

PyObject*
PyThunk_LazyRichCompare(PyThunkObject *self, PyObject *other, int cmp_op) {
    PyThunkObject *other_thunk = (PyThunkObject*) other;
    if (!PyThunk_CheckExact(self)) {
        PyErr_SetString(PyExc_ValueError, "Expected a thunk object.");
        return NULL;
    }
    if (cmp_op != Py_GE) {
        // FIXME
        PyErr_SetString(PyExc_ValueError, "Only >= supported currently.");
        return NULL;
    }
    if (!PyThunk_CheckExact(other)) {
        other_thunk = (PyThunkObject*) PyThunk_FromArray(NULL, PyArray_FromAny(other, NULL, 0, 0, 0, NULL));
    }

    ThunkOperation *op = ThunkOperation_FromBinary(
        (PyObject*) self, 
        (PyObject*) other_thunk, 
        optype_vectorizable, 
        (void*) llvm_generate_gt, 
        NULL, 
        strdup(">="));

    return PyThunk_FromOperation(op, 
        default_binary_cardinality_function, 
        cardinality_exact, 
        NPY_BOOL /* returns a boolean array */
        );
}

Value *llvm_generate_multiply(IRBuilder<>& builder, Value *l, Value *r) {
    return builder.CreateMul(l, r);
}

static PyObject*
thunk_lazymultiply(PyObject *v, PyObject *w) {
    if (!PyThunk_CheckExact(v) || !PyThunk_CheckExact(w)) {
        PyErr_SetString(PyExc_TypeError, "Expected two thunk objects as parameters.");
        return NULL;
    }
    PyThunkObject *left = (PyThunkObject*) v;
    PyThunkObject *right = (PyThunkObject*) w;
    if (left->cardinality > 0 && right->cardinality > 0 && 
        left->cardinality == cardinality_exact && right->cardinality == cardinality_exact && 
        left->cardinality != right->cardinality) {
        PyErr_SetString(PyExc_TypeError, "Incompatible cardinality.");
        return NULL;
    }
    ThunkOperation *op = ThunkOperation_FromBinary(
        (PyObject*) left, 
        (PyObject*) right, 
        optype_vectorizable, 
        (void*) llvm_generate_multiply, 
        NULL, 
        strdup("*"));

    return PyThunk_FromOperation(op, 
        default_binary_cardinality_function, 
        cardinality_exact, 
        left->type /* todo: correct type */
        );
}


PyNumberMethods thunk_as_number = {
    0,   /*nb_add*/
    0,         /*nb_subtract*/
    (binaryfunc)thunk_lazymultiply,         /*nb_multiply*/
    0,         /*nb_divide*/
    0,         /*nb_remainder*/
    0,         /*nb_divmod*/
    0,         /*nb_power*/
    0,         /*nb_negative*/
    0,         /*nb_positive*/
    0,         /*nb_absolute*/
    0,         /*nb_nonzero*/
    0,         /*nb_invert*/
    0,         /*nb_lshift*/
    0,         /*nb_rshift*/
    0,         /*nb_and*/
    0,         /*nb_xor*/
    0,         /*nb_or*/
    0,         /*nb_coerce*/
    0,         /*nb_int*/
    0,         /*nb_long*/
    0,         /*nb_float*/
    0,         /*nb_oct*/
    0,         /*nb_hex*/
    0,                           /*nb_inplace_add*/
    0,                           /*nb_inplace_subtract*/
    0,                           /*nb_inplace_multiply*/
    0,                           /*nb_inplace_divide*/
    0,                           /*nb_inplace_remainder*/
    0,                           /*nb_inplace_power*/
    0,                           /*nb_inplace_lshift*/
    0,                           /*nb_inplace_rshift*/
    0,                           /*nb_inplace_and*/
    0,                           /*nb_inplace_xor*/
    0,                           /*nb_inplace_or*/
    0,         /* nb_floor_divide */
    0, /* nb_true_divide */
    0,                           /* nb_inplace_floor_divide */
    0,                           /* nb_inplace_true_divide */
    0,          /* nb_index */
};

void initialize_thunk_as_number(void) {
    import_array();
    import_umath();
}
