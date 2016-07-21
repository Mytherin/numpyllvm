
#include "thunk.hpp"
#include "initializers.hpp"

static char module_docstring[] =
    "This module provides THUNKS.";
static char thunk_docstring[] =
    "thunk.thunk(array) => Creates a thunk array from a numpy array.";

static PyMethodDef module_methods[] = {
    {"thunk", PyThunk_FromArray, METH_O, thunk_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initjit(void);
PyMODINIT_FUNC initjit(void)
{   
    PyThunk_Init();
    initialize_thunk_methods();
    initialize_operation();
    initialize_thunk_as_number();
    initialize_parser();
    initialize_compiler();
    initialize_scheduler();

    //initialize module
    PyObject *m = Py_InitModule3("jit", module_methods, module_docstring);
    if (m == NULL)
        return;

   	create_threads();
}

void unused_function() {
    import_array();
    import_umath();
}
