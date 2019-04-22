#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                        \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}                                                                            \

// reference:
// https://devblogs.nvidia.com/unified-memory-in-cuda-6/


/**********class definition**********/

// Managed Base Class
// inherit from this to automatically allocate objects in Unified Memory
class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        CHECK(cudaMallocManaged(&ptr, len));
        CHECK(cudaDeviceSynchronize());
        return ptr;
    }

    void operator delete(void *ptr) {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(ptr));
    }
};

// String Class for Managed Memory
class String : public Managed {
private:
    int length;
    char *data;

public:
    String() : length(0), data(0) {}
    
    // Constructor for C-string initializer
    String(const char *s) : length(0), data(0) { 
        _realloc(strlen(s));
        strcpy(data, s);
    }

    // Copy constructor
    String(const String& s) : length(0), data(0) {
        _realloc(s.length);
        strcpy(data, s.data);
    }
    
    ~String() { cudaFree(data); }

    // Assignment operator
    String& operator = (const char* s) {
        _realloc(strlen(s));
        strcpy(data, s);
        return *this;
    }

    // Element access (from host or device)
    __host__ __device__
    char& operator[](int pos) { return data[pos]; }

    // C-string access
    __host__ __device__
    const char* c_str() const { return data; }

private: 
    void _realloc(int len) {
        cudaFree(data);
        length = len;
        CHECK(cudaMallocManaged(&data, length+1));
    }
};

class DataElement : public Managed {
public:
    String name;
    int value;
};

/**********CUDA kernels**********/

__global__ void Kernel_by_pointer(DataElement *elem) {
    printf("On device by pointer:       name=%s, value=%d\n", elem->name.c_str(), elem->value);
    elem->name[0] = 'p';
    elem->value++;
}

__global__ void Kernel_by_ref(DataElement &elem) {
    printf("On device by ref:           name=%s, value=%d\n", elem.name.c_str(), elem.value);
    elem.name[0] = 'r';
    elem.value++;
}

__global__ void Kernel_by_value(DataElement elem) {
    printf("On device by value:         name=%s, value=%d\n", elem.name.c_str(), elem.value);
    elem.name[0] = 'v';
    elem.value++;
}

/**********main function**********/

int main(int argc, char **argv) {
    DataElement *e = new DataElement;
  
    e->value = 10;
    e->name = "hello";
    
    Kernel_by_pointer<<< 1, 1 >>>(e);
    CHECK(cudaDeviceSynchronize());
    printf("On host (after by-pointer): name=%s, value=%d\n\n", e->name.c_str(), e->value);

    Kernel_by_ref<<< 1, 1 >>>(*e);
    CHECK(cudaDeviceSynchronize());
    printf("On host (after by-ref):     name=%s, value=%d\n\n", e->name.c_str(), e->value);

    Kernel_by_value<<< 1, 1 >>>(*e);
    CHECK(cudaDeviceSynchronize());
    printf("On host (after by-value):   name=%s, value=%d\n\n", e->name.c_str(), e->value);

    delete e;
    CHECK(cudaDeviceReset());
    return 0;
}