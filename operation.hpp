
#ifndef Py_OPERATION_H
#define Py_OPERATION_H

#include "gencode.hpp"

struct ThunkOperation {
	operator_type type;
	GenCodeInfo gencode;
    char *opname;
};

ThunkOperation *ThunkOperation_FromUnary(PyObject *arg, operator_type type, void *gencode_function, char *opname);
ThunkOperation *ThunkOperation_FromBinary(PyObject *left, PyObject *right, operator_type type, void *gencode_function, char *opname);
void ThunkOperation_Destroy(ThunkOperation *operation);

#endif /*Py_OPERATION_H*/
