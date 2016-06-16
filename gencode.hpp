
#ifndef Py_GENCODE_H
#define Py_GENCODE_H

#include "config.hpp"

typedef unsigned char gencode_type;

#define GENTYPE_UNKNOWN 255
#define GENTYPE_NOARGS 0
#define GENTYPE_UNARY 1
#define GENTYPE_BINARY 2

struct GenCodeInfo {
	gencode_type type;
	void *gencode_func;
	PyObject *parameter[2];
};

typedef unsigned char operator_type;

#define OPTYPE_UNKNOWN 0
#define OPTYPE_VECTORIZABLE 1
#define OPTYPE_FULLBREAKER 2
#define OPTYPE_PARALLELBREAKER 3
#define OPTYPE_CONSTANT 4
#define OPTYPE_NPYARRAY 5

#endif /*Py_GENCODE_H*/
