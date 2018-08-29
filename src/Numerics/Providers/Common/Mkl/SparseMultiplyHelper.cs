#if NATIVE
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;
using MklMethods = MathNet.Numerics.Providers.Common.Mkl.SafeNativeMethods;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using System.Runtime.InteropServices;

namespace MathNet.Numerics.Providers.Common.Mkl
{
    public class SparseMultiplyHelper : IDisposable
    {
        public SparseMultiplyHelper(SparseMatrix a)
        {
            origMatrix = a;
            var storage = a.Storage as SparseCompressedRowMatrixStorage<double>;
            h1 = GCHandle.Alloc(storage.RowPointers, GCHandleType.Pinned);
            h2 = GCHandle.Alloc(storage.ColumnIndices, GCHandleType.Pinned);
            h3 = GCHandle.Alloc(storage.Values, GCHandleType.Pinned);
            int retval = MklMethods.d_sparse_matrix_create_csr_3(out sparseHandle, a.RowCount, a.ColumnCount, storage.RowPointers, storage.ColumnIndices, storage.Values);
            if (retval != 0)
                throw new Exception($"Failed to create MKL spblas sparse matrix handle: {retval}");
        }

        public void Multiply(Matrix<double> b, Matrix<double> c)
        {
            if (!(b.Storage is DenseColumnMajorMatrixStorage<double>) ||
                !(c.Storage is DenseColumnMajorMatrixStorage<double>))
            {
                origMatrix.Multiply(b, c);
                return;
            }

            _Optimize();

            var bStor = b.Storage as DenseColumnMajorMatrixStorage<double>;
            var cStor = c.Storage as DenseColumnMajorMatrixStorage<double>;

            int n1 = c.RowCount;
            int n2 = b.RowCount;
            int n3 = c.ColumnCount;
            MklMethods.d_sparse_matrix_multiply(Transpose.DontTranspose, 1.0, sparseHandle, bStor.Data, n3, n2, 0.0, cStor.Data, n1);
        }

        public void TransposeThisAndMultiply(Matrix<double> b, Matrix<double> c)
        {
            if (!(b.Storage is DenseColumnMajorMatrixStorage<double>) ||
                !(c.Storage is DenseColumnMajorMatrixStorage<double>))
            {
                origMatrix.TransposeThisAndMultiply(b, c);
                return;
            }

            _Optimize();

            var bStor = b.Storage as DenseColumnMajorMatrixStorage<double>;
            var cStor = c.Storage as DenseColumnMajorMatrixStorage<double>;

            int n1 = c.RowCount;
            int n2 = b.RowCount;
            int n3 = c.ColumnCount;
            var bpin = GCHandle.Alloc(bStor.Data, GCHandleType.Pinned);
            var cpin = GCHandle.Alloc(cStor.Data, GCHandleType.Pinned);
            MklMethods.d_sparse_matrix_multiply(Transpose.Transpose, 1.0, sparseHandle, bStor.Data, n3, n2, 0.0, cStor.Data, n1);
            bpin.Free();
            cpin.Free();
        }

        internal void _Optimize()
        {
            if (!needToOptimize) return;

            MklMethods.d_sparse_matrix_optimize(sparseHandle);
            needToOptimize = false;
        }

        public void Hint(Transpose op, int finalColumns, int repCount)
        {
            MklMethods.d_sparse_set_hint(sparseHandle, op, finalColumns, repCount);
            needToOptimize = true;
        }

        internal SparseMatrix origMatrix;
        internal IntPtr sparseHandle = IntPtr.Zero;
        internal bool needToOptimize = false;
        internal GCHandle h1, h2, h3;

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                if (sparseHandle != IntPtr.Zero)
                {
                    MklMethods.d_sparse_matrix_destroy(sparseHandle);
                    sparseHandle = IntPtr.Zero;
                    h1.Free();
                    h2.Free();
                    h3.Free();
                }

                disposedValue = true;
            }
        }

        ~SparseMultiplyHelper() {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(false);
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
#endif
