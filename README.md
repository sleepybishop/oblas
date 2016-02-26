# oblas

these implement just enough blas-like routines to implement a solver over a finite field of octets.

my hope was that with some prodding i could get gcc to auto vectorize them, and it does for the simpler ones, however GEMM and AXPY do not vectorize very well and are the most expensive.

TODO:
try to get more speed by directly calling intrinsics for xor/load ops, which will probably require some work to guarantee proper alignments.

if youre looking for something robust, you probably want https://github.com/ceph/gf-complete or https://github.com/linbox-team/fflas-ffpack
