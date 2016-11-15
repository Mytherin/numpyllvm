
#ifndef Py_MUTEX_H
#define Py_MUTEX_H

#ifdef __APPLE__
#include <dispatch/dispatch.h>
typedef dispatch_semaphore_t semaphore_struct;
#else
#include <semaphore.h>
typedef sem_t semaphore_struct;
#endif

void semaphore_init(semaphore_struct *semaphore, int val);
void semaphore_wait(semaphore_struct *semaphore);
void semaphore_increment(semaphore_struct *semaphore);
void semaphore_destroy(semaphore_struct *semaphore);


#endif /* Py_MUTEX_H */
