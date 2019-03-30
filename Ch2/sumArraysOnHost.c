#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void initialData(float *ip, int rows, int cols);
void sumArraysOnHost(float *A, float *B, float *C, const int rows, const int cols);

int main(int argc, char **argv) {
    int rows = 10000, cols = 10000;
    size_t nBytes = rows * cols * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    h_C = (float *) malloc(nBytes);

    initialData(h_A, rows, cols);
    initialData(h_B, rows, cols);

    sumArraysOnHost(h_A, h_B, h_C, rows, cols);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

void initialData(float *ip, int rows, int cols) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            ip[idx] = (float) (rand() & 0xFF) / 10.f;
        }
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}