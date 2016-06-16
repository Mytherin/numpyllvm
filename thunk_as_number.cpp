
#include "thunk.hpp"
#include "debug_printer.hpp"

static
PyObject *thunk_lazymultiply(PyObject *v, PyObject *w) {
    if (!PyThunk_CheckExact(v) || !PyThunk_CheckExact(w)) {
        PyErr_SetString(PyExc_TypeError, "Expected two thunk objects as parameters.");
        return NULL;
    }
    PyThunkObject *left = (PyThunkObject*) v;
    PyThunkObject *right = (PyThunkObject*) w;
    if (left->cardinality != right->cardinality) {
        PyErr_SetString(PyExc_TypeError, "Incompatible cardinality.");
        return NULL;
    }
    size_t cardinality = left->cardinality;
    ThunkOperation *op = ThunkOperation_FromBinary((PyObject*) left, (PyObject*) right, OPTYPE_VECTORIZABLE, NULL, "*");
    return PyThunk_FromOperation(op, cardinality, 0, left->type);
}


PyNumberMethods thunk_as_number = {
    0,   /*nb_add*/
    0,         /*nb_subtract*/
    (binaryfunc)thunk_lazymultiply,         /*nb_multiply*/
    0,         /*nb_divide*/
    0,         /*nb_remainder*/
    0,         /*nb_divmod*/
    0,         /*nb_power*/
    0,         /*nb_negative*/
    0,         /*nb_positive*/
    0,         /*nb_absolute*/
    0,         /*nb_nonzero*/
    0,         /*nb_invert*/
    0,         /*nb_lshift*/
    0,         /*nb_rshift*/
    0,         /*nb_and*/
    0,         /*nb_xor*/
    0,         /*nb_or*/
    0,         /*nb_coerce*/
    0,         /*nb_int*/
    0,         /*nb_long*/
    0,         /*nb_float*/
    0,         /*nb_oct*/
    0,         /*nb_hex*/
    0,                           /*nb_inplace_add*/
    0,                           /*nb_inplace_subtract*/
    0,                           /*nb_inplace_multiply*/
    0,                           /*nb_inplace_divide*/
    0,                           /*nb_inplace_remainder*/
    0,                           /*nb_inplace_power*/
    0,                           /*nb_inplace_lshift*/
    0,                           /*nb_inplace_rshift*/
    0,                           /*nb_inplace_and*/
    0,                           /*nb_inplace_xor*/
    0,                           /*nb_inplace_or*/
    0,         /* nb_floor_divide */
    0, /* nb_true_divide */
    0,                           /* nb_inplace_floor_divide */
    0,                           /* nb_inplace_true_divide */
    0,          /* nb_index */
};

void initialize_thunk_as_number(void) {
    import_array();
    import_umath();
}
