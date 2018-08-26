#include "wrapper_common.h"
#include "blas.h"

#if __cplusplus
extern "C" {
#endif
DLLEXPORT void s_axpy(const blas_int n, const float alpha, const float x[], float y[]){
	cblas_saxpy(n, alpha, x, 1, y, 1);
}

DLLEXPORT void d_axpy(const blas_int n, const double alpha, const double x[], double y[]){
	cblas_daxpy(n, alpha, x, 1, y, 1);
}

DLLEXPORT void c_axpy(const blas_int n, const blas_complex_float alpha, const blas_complex_float x[], blas_complex_float y[]){
	cblas_caxpy(n, (float*)&alpha, (float*)x, 1, (float*)y, 1);
}

DLLEXPORT void z_axpy(const blas_int n, const blas_complex_double alpha, const blas_complex_double x[], blas_complex_double y[]){
	cblas_zaxpy(n, (double*)&alpha, (double*)x, 1, (double*)y, 1);
}

DLLEXPORT void s_scale(const blas_int n, const float alpha, float x[]){
	cblas_sscal(n, alpha, x, 1);
}

DLLEXPORT void d_scale(const blas_int n, const double alpha, double x[]){
	cblas_dscal(n, alpha, x, 1);
}

DLLEXPORT void c_scale(const blas_int n, const blas_complex_float alpha, blas_complex_float x[]){
	cblas_cscal(n, (float*)&alpha, (float*)x, 1);
}

DLLEXPORT void z_scale(const blas_int n, const blas_complex_double alpha, blas_complex_double x[]){
	cblas_zscal(n, (double*)&alpha, (double*)x, 1);
}

DLLEXPORT float s_dot_product(const blas_int n, const float x[], const float y[]){
	return cblas_sdot(n, x, 1, y, 1);
}

DLLEXPORT double d_dot_product(const blas_int n, const double x[], const double y[]){
	return cblas_ddot(n, x, 1, y, 1);
}

DLLEXPORT blas_complex_float c_dot_product(const blas_int n, const blas_complex_float x[], const blas_complex_float y[]){
	blas_complex_float ret;
	cblas_cdotu_sub(n, (float*)x, 1, (float*)y, 1, &ret);
	return ret;
}

DLLEXPORT blas_complex_double z_dot_product(const blas_int n, const blas_complex_double x[], const blas_complex_double y[]){
	blas_complex_double ret;
	cblas_zdotu_sub(n, (double*)x, 1, (double*)y, 1, &ret);
	return ret;
}

DLLEXPORT void s_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const blas_int m, const blas_int n, const blas_int k, const float alpha, const float x[], const float y[], const float beta, float c[]){
	blas_int lda = transA == CblasNoTrans ? m : k;
	blas_int ldb = transB == CblasNoTrans ? k : n;

	cblas_sgemm(CblasColMajor, transA, transB, m, n, k, alpha, x, lda, y, ldb, beta, c, m);
}

DLLEXPORT void d_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const blas_int m, const blas_int n, const blas_int k, const double alpha, const double x[], const double y[], const double beta, double c[]){
	blas_int lda = transA == CblasNoTrans ? m : k;
	blas_int ldb = transB == CblasNoTrans ? k : n;

	cblas_dgemm(CblasColMajor, transA, transB, m, n, k, alpha, x, lda, y, ldb, beta, c, m);
}

DLLEXPORT void c_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const blas_int m, const blas_int n, const blas_int k, const blas_complex_float alpha, const blas_complex_float x[], const blas_complex_float y[], const blas_complex_float beta, blas_complex_float c[]){
	blas_int lda = transA == CblasNoTrans ? m : k;
	blas_int ldb = transB == CblasNoTrans ? k : n;

	cblas_cgemm(CblasColMajor, transA, transB, m, n, k, (float*)&alpha, (float*)x, lda, (float*)y, ldb, (float*)&beta, (float*)c, m);
}

DLLEXPORT void z_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const blas_int m, const blas_int n, const blas_int k, const blas_complex_double alpha, const blas_complex_double x[], const blas_complex_double y[], const blas_complex_double beta, blas_complex_double c[]){
	blas_int lda = transA == CblasNoTrans ? m : k;
	blas_int ldb = transB == CblasNoTrans ? k : n;

	cblas_zgemm(CblasColMajor, transA, transB, m, n, k, (double*)&alpha, (double*)x, lda, (double*)y, ldb, (double*)&beta, (double*)c, m);
}

DLLEXPORT blas_int d_sparse_matrix_create_csr(
    sparse_matrix_t* A,
    const blas_int rows,
    const blas_int cols,
    blas_int rows_start[],
    blas_int rows_end[],
    blas_int col_indx[],
    double values[])
{
    /*
    sparse_status_t mkl_sparse_d_create_csr(
        sparse_matrix_t *A,
        sparse_index_base_t indexing,
        MKL_INT rows,
        MKL_INT cols,
        MKL_INT *rows_start,
        MKL_INT *rows_end,
        MKL_INT *col_indx,
        double *values);
        */
    return (blas_int)mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, rows, cols, (MKL_INT*)rows_start, (MKL_INT*)rows_end, (MKL_INT*)col_indx, (double*)values);
}

DLLEXPORT blas_int d_sparse_matrix_destroy(sparse_matrix_t A)
{
    return (blas_int)mkl_sparse_destroy(A);
}

DLLEXPORT blas_int d_sparse_set_hint(sparse_matrix_t A, CBLAS_TRANSPOSE transA, const blas_int dense_matrix_columns, const blas_int expected_calls)
{
    /*
    parse_status_t mkl_sparse_set_mm_hint (
        sparse_matrix_t A,
        sparse_operation_t operation,
        struct matrix_descr descr,
        sparse_layout_t layout,
        MKL_INT dense_matrix_size,
        MKL_INT expected_calls);
    */
    sparse_operation_t op;
    switch (transA)
    {
    case CblasNoTrans:
        op = SPARSE_OPERATION_NON_TRANSPOSE;
        break;
    case CblasTrans:
        op = SPARSE_OPERATION_TRANSPOSE;
        break;
    case CblasConjTrans:
        op = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        break;

    default:
        return -1;
    }
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    return (blas_int)mkl_sparse_set_mm_hint(A, op, descr, SPARSE_LAYOUT_ROW_MAJOR, dense_matrix_columns, expected_calls);
}

DLLEXPORT blas_int d_sparse_matrix_optimize(sparse_matrix_t A)
{
    return (blas_int)mkl_sparse_optimize(A);
}

DLLEXPORT blas_int d_sparse_matrix_multiply(
    CBLAS_TRANSPOSE transA,
    double alpha,
    sparse_matrix_t A,
    const double x[],
    const blas_int columns, // n
    const blas_int ldx, // k
    double beta,
    double y[],
    const blas_int ldy) // m
{
    // if not transposed
    // a := m x k
    // x := k x n
    // y := m x n

    // if transposed
    // a := k x m
    // x := k x n
    // y := m x n
    /*
    sparse_status_t mkl_sparse_d_mm (
        sparse_operation_t operation ,
        double alpha ,
        const sparse_matrix_t A ,
        struct matrix_descr descr ,
        sparse_layout_t layout ,
        const double *x ,
        MKL_INT columns ,
        MKL_INT ldx ,
        double beta ,
        double *y ,
        MKL_INT ldy );
    */

    sparse_operation_t operation;
    switch (transA) {
    case CblasNoTrans:
        operation = SPARSE_OPERATION_NON_TRANSPOSE;
        break;
    case CblasTrans:
        operation = SPARSE_OPERATION_TRANSPOSE;
        break;
    case CblasConjTrans:
        operation = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        break;
    default:
        return -1;
    }

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    return (blas_int) mkl_sparse_d_mm(
        operation,
        alpha,
        A,
        descr,
        SPARSE_LAYOUT_ROW_MAJOR,
        x,
        columns,
        ldx,
        beta,
        y,
        ldy);
}

#if __cplusplus
}
#endif
