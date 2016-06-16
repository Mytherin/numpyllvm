



#ifndef Py_SCHEDULER_H
#define Py_SCHEDULER_H

#include "parser.hpp"

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
    // todo
};

// Adds a task to the task queue
void ScheduleTask(Task *task);
void SchedulePipeline(Pipeline *pipeline);
void RunThread(void);
#endif /* Py_SCHEDULER_H */
