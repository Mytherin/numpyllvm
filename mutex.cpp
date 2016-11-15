
#include "mutex.hpp"


#ifdef __APPLE__
void semaphore_init(semaphore_struct *semaphore, int val) {
	*semaphore = dispatch_semaphore_create(val);
}

void semaphore_wait(semaphore_struct *semaphore) {
	dispatch_semaphore_wait(*semaphore, DISPATCH_TIME_FOREVER);
}

void semaphore_increment(semaphore_struct *semaphore) {
    dispatch_semaphore_signal(*semaphore);
}

void semaphore_destroy(semaphore_struct *semaphore) {
	(void) semaphore;
}
#else
void semaphore_init(semaphore_struct *semaphore, int val) {
	if (sem_init(semaphore, 0, val) < 0) {
		printf("Failed to initialize semaphore: %s", strerror());
		errno = 0;
	}
}

void semaphore_wait(semaphore_struct *semaphore) {
	if (sem_wait(semaphore) < 0) {
		printf("Failed to wait for semaphore: %s", strerror());
		errno = 0;
	}
}

void semaphore_increment(semaphore_struct *semaphore) {
	if (sem_post(semaphore) < 0) {
		printf("Failed to increment semaphore: %s", strerror());
		errno = 0;
	}
}

void semaphore_destroy(semaphore_struct *semaphore) {
	if (sem_destroy(semaphore) < 0) {
		printf("Failed to destroy semaphore: %s", strerror());
		errno = 0;
	}
}
#endif
