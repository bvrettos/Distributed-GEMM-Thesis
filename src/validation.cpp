#include "validation.hpp"

template <typename T>
T abs(T x) {
    if (x < 0) return -x;
    else return x;
}

template <typename T>
inline T Terror(T a, T b) {
    if (a == 0) return abs<T>(a - b);
    return abs(a-b)/a;
}

template <typename T>
long int vectorDifference(T* a, T* b, long long size, double eps) {
    int failed = 0;

    for (long long i = 0; i < size; i++) {
        if (Terror(a[i], b[i]) > eps) {
            failed++;
        }
    }

    return failed;
}

template <typename T>
short testEquality()
{

}

short Dtest_equality(double* C_comp, double* C, long long size) {
  long int acc = 8, failed;
  double eps = 1e-8;
  failed = Dvec_diff(C_comp, C, size, eps);
  while (eps > DBL_MIN && !failed && acc < 30) {
    eps *= 0.1;
    acc++;
    failed = Dvec_diff(C_comp, C, size, eps);
  }
  if (8==acc) {
  	fprintf(stderr, "Test failed %zu times\n", failed);
  	int ctr = 0;
    long long itt = 0;
  	while (ctr < 10 & itt < size){
  		if (Derror(C_comp[itt], C[itt]) > eps){
  			fprintf(stderr, "Baseline vs Tested(adr = %p, itt = %lld): %.15lf vs %.15lf\n", &C[itt], itt, C_comp[itt], C[itt]);
  			ctr++;
  		}
  		itt++;
  	}
  return 0;
  }
  else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  return (short) acc;
}

long int Dvec_diff(double* a, double* b, long long size, double eps) {
	long int failed = 0;
	//#pragma omp parallel for
	for (long long i = 0; i < size; i++)
		if (Derror(a[i], b[i]) > eps){
			//#pragma omp atomic
			failed++;
		}
	return failed;
}

inline double Derror(double a, double b) {
  if (a == 0) return dabs(a - b);
  return dabs(a - b)/a;
}

double dabs(double x) {
	if (x < 0) return -x;
	else return x;
}

template double abs<double>(double x);
template float abs<float>(float x);