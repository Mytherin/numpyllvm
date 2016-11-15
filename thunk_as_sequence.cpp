
#include "thunk.hpp"
#include "debug_printer.hpp"

PySequenceMethods thunk_as_sequence = {
    (lenfunc) NULL,                  /*sq_length*/
    (binaryfunc )NULL,                       /*sq_concat is handled by nb_add*/
    (ssizeargfunc)NULL,
    (ssizeargfunc)NULL,                /* sq_item */
    (ssizessizeargfunc)NULL,
    (ssizeobjargproc)NULL,        /*sq_ass_item*/
    (ssizessizeobjargproc)NULL,  /*sq_ass_slice*/
    (objobjproc) NULL,            /*sq_contains */
    (binaryfunc) NULL,                      /*sg_inplace_concat */
    (ssizeargfunc)NULL,
};

void initialize_thunk_as_sequence(void) {
    import_array();
    import_umath();
}
