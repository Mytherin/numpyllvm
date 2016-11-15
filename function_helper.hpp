


#ifndef Py_FUNCTIONHELPER_H
#define Py_FUNCTIONHELPER_H

int FindMethodDef(PyMethodDef[] array, char *name) {
	int i = 0;
	while(array[i] != NULL) {
		if (strcasecmp(array[i], name) == 0) {
			return i;
		}
	}
	return -1;
}

#endif /* Py_FUNCTIONHELPER_H */
