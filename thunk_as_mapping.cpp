
#include "thunk.hpp"
#include "debug_printer.hpp"

#include "compiler.hpp"

using namespace llvm;

Value *llvm_generate_subscript(JITInformation& inp, Operation *op, Value *l, Value *r) {
    BasicBlock *conditional_branch = BasicBlock::Create(*inp.context, "branch", inp.function, 0);
    inp.builder->CreateCondBr(r, conditional_branch, inp.loop_inc);
    inp.builder->SetInsertPoint(conditional_branch);
    inp.current = conditional_branch;
    assert(((BinaryOperation*) op)->extra);
    LoadInst *index = inp.builder->CreateLoad((Value*) ((BinaryOperation*) op)->extra, "index_branch");
    Value *incremented_index = inp.builder->CreateAdd(index, ConstantInt::get(Type::getInt64Ty(*inp.context), 1, true), "index++");
    inp.builder->CreateStore(incremented_index, (Value*) ((BinaryOperation*) op)->extra);
    inp.index = index;
    inp.index_addr = (Value*) ((BinaryOperation*) op)->extra;
    return l;
}

void llvm_initialize_subscript(JITInformation& info, Operation *op) {
	// create the extra index
	Type *int64_tpe = Type::getInt64Ty(*info.context);
	AllocaInst *index = info.builder->CreateAlloca(int64_tpe, nullptr);
	// we start at -1 because we first increment and then use the index
    info.builder->CreateStore(info.builder->CreateLoad(info.index_addr), index);
	((BinaryOperation*)op)->extra = (void*) index;
}

ssize_t
subscript_cardinality_function(ssize_t *inputs) {
	return inputs[1];
}

static PyObject*
thunk_subscript(PyThunkObject *self, PyObject *other) {
	ThunkOperation *op;
	void *llvm_operation = NULL;
	cardinality_type cardinality_tpe;
	PyThunkObject *other_thunk;
	if (PyThunk_CheckExact(other)) {
		// index by a thunk object
		other_thunk = (PyThunkObject*) other;
		if (other_thunk->type == NPY_BOOL) {
			// boolean type access
			// cardinality depends on the amount of TRUE in the array
			llvm_operation = (void*) llvm_generate_subscript;
			cardinality_tpe = cardinality_upper_bound;
		} else if (other_thunk->type == NPY_INT64) {
			// integer type
			// cardinality is equal to the amount of elements in the other array
			llvm_operation = NULL;
			cardinality_tpe = cardinality_exact;
		} else {
			PyErr_SetString(PyExc_IndexError, "arrays used as indices must be of integer (or boolean) type");
			return NULL;
		}
	} else {
		PyErr_SetString(PyExc_IndexError, "FIXME: other types of array access");
		return NULL;
	}

    op = ThunkOperation_FromBinary(
        (PyObject*) self, 
        (PyObject*) other_thunk, 
        optype_vectorizable, 
        (void*) llvm_operation, 
        NULL, 
        strdup("[]"));
    op->gencode.initialize = (void*) llvm_initialize_subscript;
    return PyThunk_FromOperation(op, 
        subscript_cardinality_function, 
        cardinality_tpe, 
        self->type
        );
}

static int
thunk_assign_subscript(PyThunkObject *self, PyObject *ind, PyObject *op) {
	PyThunk_EvaluateChildren(self);
	PyThunk_Evaluate(self);
	self->storage->ob_type->tp_as_mapping->mp_ass_subscript((PyObject*) self->storage, ind, op);
	return 0;
}

PyMappingMethods thunk_as_mapping = {
    (lenfunc)NULL,              /*mp_length*/
    (binaryfunc) thunk_subscript,        /*mp_subscript*/
    (objobjargproc) thunk_assign_subscript,       /*mp_ass_subscript*/
};

void initialize_thunk_as_mapping(void) {
    import_array();
    import_umath();
}
