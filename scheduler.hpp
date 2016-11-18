



#ifndef Py_SCHEDULER_H
#define Py_SCHEDULER_H

#include "parser.hpp"

class JITFunction;

typedef enum {
	tasktype_unknown = 255,
	tasktype_compile = 0,
	tasktype_execute = 1
} task_type;

#define Task_HEAD \
    task_type type;

struct Task {
    Task_HEAD
};

struct CompileTask {
    Task_HEAD
    Pipeline *pipeline;
};

struct ExecuteTask {
    Task_HEAD
    JITFunction *function;
    size_t start;
    size_t end;
    int thread_nr;
};

class Thread;

// Schedule the compilation of a pipeline; this recursively schedules compilation of all children as well
// This should only be called once on a Pipeline
void ScheduleCompilation(Pipeline *pipeline);
// Schedule a pipeline for execution; this function is called when 
//   (1) a function finishes compiling and 
//   (2) when children of a function finish executing
// the function checks if the pipeline is ready for execution (compiled and all children have finished)
// the pipeline will only be scheduled if all these conditions are met
void ScheduleExecution(Pipeline *pipeline);


// Adds a task to the task queue
void ScheduleTask(Task *task);
void DestroyTask(Task *task);
void RunThread(Thread *thread);

#endif /* Py_SCHEDULER_H */
