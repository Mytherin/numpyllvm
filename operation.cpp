
#include "operation.hpp"

ThunkOperation *ThunkOperation_FromUnary(PyObject *arg, operator_type type, void *gencode_function, base_function_unary base, char *opname) {
    ThunkOperation *op = (ThunkOperation*) malloc(sizeof(ThunkOperation));
    op->type = type;
    op->gencode.type = gentype_unary;
    op->gencode.gencode = gencode_function;
    op->gencode.initialize = NULL;
    op->gencode.base = (void*) base;
    op->gencode.parameter[0] = arg;
    op->opname = strdup(opname);
    Py_INCREF(arg);
    return op;
}

ThunkOperation *ThunkOperation_FromBinary(PyObject *left, PyObject *right, operator_type type, void *gencode_function, base_function_binary base, char *opname) {
	ThunkOperation *op = (ThunkOperation*) malloc(sizeof(ThunkOperation));
	op->type = type;
	op->gencode.type = gentype_binary;
	op->gencode.gencode = gencode_function;
    op->gencode.initialize = NULL;
    op->gencode.base = (void*) base;
	op->gencode.parameter[0] = left;
	op->gencode.parameter[1] = right;
    op->opname = strdup(opname);
    Py_INCREF(left);
    Py_INCREF(right);
	return op;
}

void ThunkOperation_Destroy(ThunkOperation *operation) {
    if (!operation) return;
    for(int i = 0; i < operation->gencode.type; i++) {
        Py_DECREF(operation->gencode.parameter[i]);
    }
    free(operation);
}

void initialize_operation(void) {
    import_array();
    import_umath();
}