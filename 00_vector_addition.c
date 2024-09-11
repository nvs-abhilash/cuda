
#include <stdio.h>
#include <stdlib.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int n)
{
    for (int i = 0; i < n; i++)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main()
{
    int n = 10; // Set the size of the arrays
    float *A_h, *B_h, *C_h;

    // Allocate memory for the arrays
    A_h = (float *)malloc(n * sizeof(float));
    B_h = (float *)malloc(n * sizeof(float));
    C_h = (float *)malloc(n * sizeof(float));

    // Check if memory allocation was successful
    if (A_h == NULL || B_h == NULL || C_h == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize the arrays
    for (int i = 0; i < n; i++)
    {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    // Perform the vector addition
    vecAdd(A_h, B_h, C_h, n);

    // Print the result
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", C_h[i]);
    }

    printf("\n");

    // Free the memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}