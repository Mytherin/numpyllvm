
#ifndef Py_OPERATION_H
#define Py_OPERATION_H

#include "gencode.hpp"

struct ThunkOperation {
	operator_type type;
	GenCodeInfo gencode;
};

ThunkOperation *ThunkOperation_FromBinary(PyObject *left, PyObject *right, operator_type type, void *gencode_function);


#endif /*Py_OPERATION_H*/
