


#ifndef Py_THUNK_H
#define Py_THUNK_H

#include "operation.hpp"

typedef ssize_t (*cardinality_function)(ssize_t *inputs);

ssize_t default_cardinality_function(ssize_t *inputs);
ssize_t default_binary_cardinality_function(ssize_t *inputs);

typedef enum {
	cardinality_unknown = 255,
	cardinality_exact = 1,
	cardinality_upper_bound = 2
} cardinality_type;

struct PyThunkObject {
	PyObject_HEAD
	// underlying NumPy array that stores the data
	PyArrayObject *storage;
	// flag that indicates whether or not the array is evaluated
	bool evaluated;
	// the operation that defines how to compute this array
	ThunkOperation *operation;
	// cardinality of thunk
	ssize_t cardinality;
	// cardinality type of the thunk
	cardinality_type cardinality_type;
	// function to compute cardinality of function given a set of inputs
	cardinality_function cardinality_function;
	// type of thunk
	int type;
	// name of thunk
	char *name;
};

PyAPI_DATA(PyTypeObject) PyThunk_Type;

#define PyThunk_Check(op) ((op)->ob_type == &PyThunk_Type)
#define PyThunk_CheckExact(op) ((op)->ob_type == &PyThunk_Type)

PyObject *PyThunk_FromArray(PyObject *, PyObject*);
PyObject *PyThunk_FromOperation(ThunkOperation *operation, cardinality_function cardinality_func, cardinality_type cardinality_tpe, int type);

void PyThunk_Init(void);

PyObject* PyThunk_Evaluate(PyThunkObject *thunk);
bool PyThunk_IsEvaluated(PyThunkObject *thunk);
PyObject *PyThunk_AsArray(PyObject*);

extern PyObject* PyThunk_LazyRichCompare(PyThunkObject *self, PyObject *other, int cmp_op);
extern PyNumberMethods thunk_as_number;
extern PySequenceMethods thunk_as_sequence;
extern PyMappingMethods thunk_as_mapping;
extern struct PyMethodDef thunk_methods[];

#endif /*Py_THUNK_H*/
