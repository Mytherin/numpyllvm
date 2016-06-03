
#include "operation.hpp"

ThunkOperation *ThunkOperation_FromBinary(PyObject *left, PyObject *right, operator_type type, void *gencode_function) {
	ThunkOperation *op = (ThunkOperation*) malloc(sizeof(ThunkOperation));
	op->type = type;
	op->gencode.type = GENTYPE_BINARY;
	op->gencode.gencode_func = gencode_function;
	op->gencode.parameter[0] = left;
	op->gencode.parameter[1] = right;
	return op;
}

void initialize_operation(void) {
    import_array();
    import_umath();
}