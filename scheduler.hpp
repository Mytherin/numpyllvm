



#ifndef Py_SCHEDULER_H
#define Py_SCHEDULER_H

#include "parser.hpp"

class JITFunction;

typedef unsigned char task_type;

#define TASKTYPE_UNKNOWN 255
#define TASKTYPE_COMPILE 0
#define TASKTYPE_EXECUTE 1

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
};

class Thread;

// Adds a task to the task queue
void ScheduleTask(Task *task);
void DestroyTask(Task *task);
void SchedulePipeline(Pipeline *pipeline);
void RunThread(Thread *thread);

#endif /* Py_SCHEDULER_H */
