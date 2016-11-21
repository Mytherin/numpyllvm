
#ifndef Py_GENCODE_H
#define Py_GENCODE_H

#include "config.hpp"

typedef void* (*base_function)();
typedef PyArrayObject* (*base_function_nullary)(void);
typedef PyArrayObject* (*base_function_unary)(PyArrayObject*);
typedef PyArrayObject* (*base_function_binary)(PyArrayObject*,PyArrayObject*);

typedef void (*initialize_data_function)(void *data_source, int thread_count);
typedef void (*finalize_data_function)(void *jit_function, void *data_source, int thread_count);

typedef enum {
	gentype_unknown = 255,
	gentype_noargs = 0,
	gentype_unary = 1,
	gentype_binary = 2
} gencode_type;

struct GenCodeInfo {
	gencode_type type;
	void *initialize;            /* the function called to generate the LLVM code for the initialization */
	void *gencode;               /* the function called to generate the LLVM code for this operation in the loop */
	initialize_data_function initialize_data;
	finalize_data_function finalize_data;
	void *base;                  /* the base NumPy function to call (for when there is no gencode or for when compiling is too expensive) */
	PyObject *parameter[2];
};

typedef enum {
	optype_unknown,              /* unknown, should not be used */
	optype_vectorizable,         /* vectorizable: executed within a loop statement */
	optype_fullbreaker,          /* full pipeline breaker: everything is materialized
									and an external (e.g. Numpy) function is called on the materialized array */
	optype_parallelbreaker,      /* parallel breaker: breaks pipelines,
	                                but this function is still computed within the LLVM JIT (no external function)*/
	optype_constant,             /* constant numeric value */
	optype_npyarray              /* materialized numpy array */
} operator_type;

#endif /*Py_GENCODE_H*/
