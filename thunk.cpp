
#include "thunk.hpp"
#include "parser.hpp"
#include "scheduler.hpp"

#include <string>
#include "debug_printer.hpp"

ssize_t
default_cardinality_function(ssize_t *inputs) {
    return inputs[0];
}

ssize_t
default_binary_cardinality_function(ssize_t *inputs) {
    // if any of the two inputs is a constant, the other size is used
    if (inputs[0] == 1) return inputs[1];
    if (inputs[1] == 1) return inputs[0];
    // if neither are constants and the size doesn't match we throw an error
    if (inputs[0] != inputs[1]) {
        return -1;
    }
    return inputs[0];
}

PyObject*
PyThunk_FromArray(PyObject *unused, PyObject* input) {
    PyThunkObject *thunk;
    (void) unused;
    input = PyArray_FromAny(input, NULL, 0, 0, NPY_ARRAY_ENSURECOPY, NULL);
    if (input == NULL || !PyArray_CheckExact(input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array as parameter.");
        return NULL;
    }
    thunk = (PyThunkObject *)PyObject_MALLOC(sizeof(PyThunkObject));
    if (thunk == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)thunk, &PyThunk_Type);
    thunk->storage = (PyArrayObject*) input;
    thunk->evaluated = true;
    thunk->operation = NULL;
    thunk->cardinality =  PyArray_SIZE(thunk->storage);
    thunk->cardinality_type = cardinality_exact;
    thunk->cardinality_function = NULL;
    thunk->type = PyArray_TYPE(thunk->storage);
    thunk->name = getname("A");
    return (PyObject*)thunk;
}

PyObject*
PyThunk_FromOperation(ThunkOperation *operation, cardinality_function cardinality_func, cardinality_type cardinality_tpe, int type) {
    PyThunkObject *thunk;
    thunk = (PyThunkObject *)PyObject_MALLOC(sizeof(PyThunkObject));
    if (thunk == NULL)
        return PyErr_NoMemory();
    PyObject_Init((PyObject*)thunk, &PyThunk_Type);
    thunk->storage = NULL;;
    thunk->evaluated = false;
    thunk->operation = operation;
    thunk->cardinality = -1;
    thunk->cardinality_type = cardinality_tpe;
    thunk->cardinality_function = cardinality_func;
    thunk->type = type;
    thunk->name = getname("operation");
    return (PyObject*)thunk;
}

void PyThunk_Init(void) {
    import_array();
    import_umath();
    if (PyType_Ready(&PyThunk_Type) < 0)
        return;
}

PyObject*
PyThunk_Evaluate(PyThunkObject *thunk) {
	if (PyThunk_IsEvaluated(thunk)) Py_RETURN_NONE;
    // parse the separate pipelines from the thunk 
    // (i.e. split the set of operations on blocking operations so we have a set of standalone pipelines)
    // each separate pipeline will be compiled into a single function
    Pipeline *pipeline = ParsePipeline(thunk);
    
    // initialize the semaphore of this pipeline, as we have to wait for its completion
    semaphore_init(&pipeline->semaphore, 0);

    PrintPipeline(pipeline);
    // schedule the pipeline for compilation
    ScheduleCompilation(pipeline);

    printf("Waiting for object at %p (Pipeline %p)\n", pipeline->outputData->objects[0].source->object, pipeline);
    semaphore_wait(&pipeline->semaphore);
    printf("Finished waiting for object at %p\n", pipeline->outputData->objects[0].source->object);
    if (!thunk->evaluated) {
        PyErr_SetString(PyExc_ValueError, "Something went wrong evaluating a Thunk.");
        return NULL;
    }
	Py_RETURN_NONE;
}

bool PyThunk_IsEvaluated(PyThunkObject *thunk) {
	return thunk->evaluated;
}

PyObject *PyThunk_AsArray(PyObject* thunk) {
	if (PyThunk_CheckExact(thunk) && PyThunk_Evaluate((PyThunkObject*) thunk) != NULL) {
		return (PyObject*) ((PyThunkObject*)thunk)->storage;
	}
	return NULL;
}

static PyObject *
thunk_str(PyThunkObject *self)
{
	if (PyThunk_Evaluate(self) == NULL) {
		return NULL;
	}
    return PyArray_Type.tp_str((PyObject*)self->storage);
}

static void
PyThunk_dealloc(PyThunkObject* self)
{
    Py_XDECREF(self->storage);
    self->ob_type->tp_free((PyObject*)self);
}

PyTypeObject PyThunk_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "thunk",
    sizeof(PyThunkObject),
    0,
    (destructor)PyThunk_dealloc,                /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    (reprfunc)0,                   /* tp_repr */
    &thunk_as_number,                       /* tp_as_number */
    &thunk_as_sequence,                                          /* tp_as_sequence */
    &thunk_as_mapping,                                          /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)thunk_str,                    /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
#if !defined(NPY_PY3K)
     | Py_TPFLAGS_CHECKTYPES
     | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
     | Py_TPFLAGS_BASETYPE),                    /* tp_flags */
    "Thunk.",                        /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)PyThunk_LazyRichCompare,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    thunk_methods,                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    PyType_GenericAlloc,                        /* tp_alloc */
    PyType_GenericNew,                          /* tp_new */
    PyObject_Del,                               /* tp_free */
    0,
    0,
    0,
    0,
    0,
    0, 
    0,
    0
};
