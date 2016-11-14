
#include "thunk.hpp"
#include "initializers.hpp"


static PyObject *
_thunk_evaluate(PyThunkObject *self, PyObject *args) {
    (void) args;
	if (PyThunk_Evaluate(self) == NULL) {
		return NULL;
	}
    PyThunk_Evaluate(self);
    Py_RETURN_NONE;
}

static PyObject *
_thunk_isevaluated(PyThunkObject *self, PyObject *args) {
    (void) args;
    if (PyThunk_IsEvaluated(self)) {
    	Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
_thunk_asarray(PyThunkObject *self, PyObject *args) {
    (void) args;
    PyObject *arr = PyThunk_AsArray((PyObject*) self);
    Py_INCREF(arr);
    return arr;
}

static PyArrayObject* 
_thunk_inline_sort(PyArrayObject* input) {
    PyArrayObject *output = (PyArrayObject*) PyArray_FromArray(input, PyArray_DESCR(input), NPY_ARRAY_ENSURECOPY);
    PyArray_Sort(output, 0, NPY_QUICKSORT);
    return output;
}

static PyObject *
_thunk_sort(PyThunkObject *self, PyObject *unused) {
    (void) unused;
    if (!PyThunk_CheckExact(self)) {
        PyErr_SetString(PyExc_TypeError, "Expected a thunk object as parameters.");
        return NULL;
    }
    PyThunkObject *left = (PyThunkObject*) self;
    ThunkOperation *op = ThunkOperation_FromUnary(
        (PyObject*) self, 
        optype_fullbreaker, 
        NULL, 
        (base_function_unary) _thunk_inline_sort, 
        strdup("sort"));
    return PyThunk_FromOperation(op, 
        default_cardinality_function, 
        cardinality_exact, 
        left->type);
}

struct PyMethodDef thunk_methods[] = {
    {"evaluate", (PyCFunction)_thunk_evaluate, METH_NOARGS,"evaluate() => "},
    {"isevaluated", (PyCFunction)_thunk_isevaluated, METH_NOARGS,"isevaluated() => "},
    {"asnumpyarray", (PyCFunction)_thunk_asarray, METH_NOARGS,"asnumpyarray() => "},
    {"sort", (PyCFunction)_thunk_sort, METH_NOARGS,"sort() => "},
    {NULL}  /* Sentinel */
};

void initialize_thunk_methods(void) {
    import_array();
    import_umath();
}
