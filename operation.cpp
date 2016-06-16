
#include "operation.hpp"

ThunkOperation *ThunkOperation_FromBinary(PyObject *left, PyObject *right, operator_type type, void *gencode_function, char *opname) {
	ThunkOperation *op = (ThunkOperation*) malloc(sizeof(ThunkOperation));
	op->type = type;
	op->gencode.type = GENTYPE_BINARY;
	op->gencode.gencode_func = gencode_function;
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