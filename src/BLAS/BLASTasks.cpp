#include <BLAS.hpp>

template <typename scalar_t>
class L1Task {
    private:
    public:
        long long n; // Size of vectors
        scalar_t alpha; // Scale
        scalar_t *x, *y; // Actual vectors
        long long incx, incy // Element Steps
    
        L1Task();
        L1Task(long long n, scalar_t alpha, scalar_t* x, long long incx, scalar_t* y, long long incy);
}