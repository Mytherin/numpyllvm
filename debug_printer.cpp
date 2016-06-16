
#include "debug_printer.hpp"
#include <iostream>
#include <iomanip>

#define MAX_OPERATION_LINES 100
#define CONSOLE_WIDTH = 800

const char* GetOperationData(Operation *op) {
    switch(op->Type()) {
        case OPTYPE_nullop:
            return ((NullaryOperation*)op)->operation->opname;
        case OPTYPE_unop:
            return ((UnaryOperation*)op)->operation->opname;
        case OPTYPE_binop:
            return ((BinaryOperation*)op)->operation->opname;
        case OPTYPE_pipeline:
            return "Pipe";
        case OPTYPE_obj:
            return ((ObjectOperation*)op)->thunk->name;
        default:
            return "?";
    }
}


struct LocationNode {
    Operation *op;
    double location;
    LocationNode *next;
};

bool AddLocationNode(Operation *op, int line, double location, LocationNode ** nodes) {
    LocationNode* node = (LocationNode*) malloc(sizeof(LocationNode));
    node->op = op;
    node->location = location;
    node->next = NULL;
    if (nodes[line] == NULL) {
        nodes[line] = node;
    } else {
        LocationNode *prev = NULL;
        LocationNode *current = nodes[line];
        while(current && current->location <= location) {
            prev = current;
            current = current->next;
        }
        if (prev == NULL) {
            node->next = nodes[line];
            nodes[line] = node;
        } else {
            node->next = prev->next;
            prev->next = node;
        }
    }
}

size_t max(size_t a, size_t b) {
    return (a > b) ? a : b;
}

size_t GatherOperation(Operation *op, int line, double location, LocationNode ** nodes, double width = 0.25, size_t size = 20) {
    AddLocationNode(op, line, location, nodes);
    if (op->Type() == OPTYPE_binop) {
        return max(GatherOperation(((BinaryOperation*)op)->LHS, line + 1, location - width, nodes, width / 2, size + 20),
                GatherOperation(((BinaryOperation*)op)->RHS, line + 1, location + width, nodes, width / 2, size + 20));
    } else if (op->Type() == OPTYPE_unop) {
        return GatherOperation(((UnaryOperation*)op)->LHS, line + 1, location, nodes, size);
    }
    return size;
}

void PrintLocationNodes(LocationNode ** nodes, ssize_t size, ssize_t width) {
    for(int i = 0; i < MAX_OPERATION_LINES; i++) {
        bool binary = false;
        LocationNode *current = nodes[i];
        if (current == NULL) break;
        std::string line = "";
        std::string nextLine = "";
        while (current) {
            std::string opdata = std::string(GetOperationData(current->op));
            if (current->op->Type() == OPTYPE_binop) {
                binary = true;
                ssize_t start = (current->location * size) - width / 2;
                size_t midpoint = current->location * size - opdata.size();
                assert(start > line.size());
                if (start > (ssize_t) line.size()) line += std::string(start - line.size(), ' ');
                line += std::string(midpoint - start, '_');
                line += opdata;
                line += std::string(midpoint - start, '_');
                assert(start > nextLine.size());
                if (start > (ssize_t) nextLine.size() + 1) nextLine += std::string(start - nextLine.size() - 1, ' ');
                nextLine += std::string("/");
                nextLine += std::string(2 * (midpoint - start) + opdata.size(), ' ');
                nextLine += std::string("\\");
            } else if (current->op->Type() == OPTYPE_unop) {
                ssize_t start = current->location * size - opdata.size() / 2;
                assert(start > line.size());
                if (start > (ssize_t)  line.size()) line += std::string(start - line.size(), ' ');
                line += opdata;
                assert(start > nextLine.size());
                if (start > (ssize_t) nextLine.size() + 1) nextLine += std::string(start - nextLine.size() - 1, ' ');
                nextLine += std::string("|");
            } else {
                ssize_t start = current->location * size - opdata.size() / 2;
                assert(start > line.size());
                if (start > (ssize_t) line.size()) line += std::string(start - line.size(), ' ');
                line += opdata;
            }
            current = current->next;
        }
        std::cout << line << std::endl << nextLine << std::endl;
        if (binary) width /= 2;
    }
}

void 
PrintPipeline(Pipeline *pipeline) {
    LocationNode *locations[MAX_OPERATION_LINES];
    for(int i = 0; i < MAX_OPERATION_LINES; i++) {
        locations[i] = NULL;
    }
    size_t size = GatherOperation(pipeline->operation, 0, 0.5, locations);
    /*for(int i = 0; i < MAX_OPERATION_LINES; i++) {
        LocationNode *current = locations[i];
        if (!current) break;
        while(current) {
            printf("%g-", current->location);
            current = current->next;
        }
        printf("\n");
    }*/
    PrintLocationNodes(locations, size, size / 2);
}


std::string _PrintThunk(PyThunkObject *thunk, size_t recursive_depth = 0) {
    if (recursive_depth > 50) return "ERROR";
    if (thunk->operation == NULL) {
        return std::string(thunk->name);
    } else {
        std::string left = _PrintThunk((PyThunkObject*) thunk->operation->gencode.parameter[0], recursive_depth + 1);
        std::string right = _PrintThunk((PyThunkObject*) thunk->operation->gencode.parameter[1], recursive_depth + 1);
        return std::string("(") + left + std::string(" ") + std::string(thunk->operation->opname) + std::string(" ") + right + std::string(")");
    }

}

void PrintThunk(PyThunkObject *thunk) {
    std::cout << _PrintThunk(thunk) << std::endl;
}