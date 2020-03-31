#ifndef H_Orthogonalization_H
#define H_Orthogonalization_H


void modifiedGramSchmidt_singleWavefunction(double **V, double *U, double *W, int targetWavefunction, int numPoints, int numWavefunctions);

double local_dot_product(double *A, double *B, double *W, int N);

void subtract_projection(double *A, double *B, double r, int N);

void normalize(double *A, double norm, int N);

#define free_vector(v)  do { free(v); v = NULL; } while (0)

#define free_matrix(a) do {                                    \
    if (a != NULL) {                                           \
        size_t make_matrix_loop_counter;                       \
        for (make_matrix_loop_counter = 0;                     \
                (a)[make_matrix_loop_counter] != NULL;         \
                make_matrix_loop_counter++)                    \
            free_vector((a)[make_matrix_loop_counter]);        \
        free_vector(a);                                        \
        a = NULL;                                              \
    }                                                          \
} while (0)


#endif /* H_Orthogonalization_H */
