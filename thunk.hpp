


#ifndef Py_THUNK_H
#define Py_THUNK_H

#include "operation.hpp"

struct PyThunkObject {
	PyObject_HEAD;
	// underlying NumPy array that stores the data
	PyArrayObject *storage;
	// flag that indicates whether or not the array is evaluated
	bool evaluated;
	// the operation that defines how to compute this array
	ThunkOperation *operation;
	// cardinality of thunk
	size_t cardinality;
	// type of thunk
	int type;
	// name of thunk
	char *name;
};

PyAPI_DATA(PyTypeObject) PyThunk_Type;

#define PyThunk_Check(op) ((op)->ob_type == &PyThunk_Type)
#define PyThunk_CheckExact(op) ((op)->ob_type == &PyThunk_Type)

PyObject *PyThunk_FromArray(PyObject *, PyObject*);
PyObject *PyThunk_FromOperation(ThunkOperation *operation, ssize_t cardinality, int cardinality_type, int type);

void PyThunk_Init(void);

PyObject* PyThunk_Evaluate(PyThunkObject *thunk);
bool PyThunk_IsEvaluated(PyThunkObject *thunk);
PyObject *PyThunk_AsArray(PyObject*);

extern PyNumberMethods thunk_as_number;
extern struct PyMethodDef thunk_methods[];

#endif /*Py_THUNK_H*/
