# oblas

these implement just enough blas-like routines to implement a solver over a finite field of octets.

my hope was that with some prodding i could get gcc to auto vectorize them, and it does for the simpler ones, however the most commonly used one, AXPY does not vectorize very well.

[the simd branch](https://github.com/sleepybishop/oblas/tree/simd) has manually vectorized code via intrinsics for axpy and addrow

if youre looking for something robust, you probably want https://github.com/ceph/gf-complete or https://github.com/linbox-team/fflas-ffpack
