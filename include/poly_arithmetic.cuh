#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "uintarith_bfv.h"

#include "uint128.cuh"
#include "ntt_60bit.cuh"

#define poly_block 1024
#define POLY_MAX_THREADS 1024
#define small_block 256
#define POLY_MIN_BLOCKS 1

struct uint128_t2
{
    uint128_tt x;
    uint128_tt y;
};
struct double_t2
{
    double x;
    double y;
};

__global__ 
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void barrett_batch_kernel(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    register uint32_tt index = blockIdx.y;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt ra = a[i + idx_a * n];
    register uint64_tt rb = b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    a[i + idx_a * n]=rc.low;
}

__host__ void barrett_batch_device(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 barrett_dim(n / poly_block, mod_num);
    barrett_batch_kernel <<< barrett_dim, poly_block >>> (a, b, n, idx_a, idx_b, idx_mod, mod_num);
}

__global__ 
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void barrett_2batch_kernel(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num, int q_num)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + idx_in_pq * n;

    register uint64_tt ra = a[i + idx_a * n];
    register uint64_tt rb = b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);
    a[i + idx_a * n]=rc.low;

    ra = a[i + idx_a * n + q_num*n];
    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    a[i + idx_a * n + q_num*n]=rc.low;
}

__host__ void barrett_2batch_device(uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num, int q_num)
{
    dim3 barrett_dim(n / poly_block, mod_num);
    barrett_2batch_kernel <<< barrett_dim, poly_block >>> (a, b, n, idx_a, idx_b, idx_mod, mod_num, q_num);
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void barrett_batch_3param_kernel(uint64_tt c[], uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    register uint32_tt index = blockIdx.y;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt ra = a[i + idx_a * n];
    register uint64_tt rb = b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    c[i + idx_c * n]=rc.low;
}

__host__ void barrett_batch_3param_device(uint64_tt c[], uint64_tt a[], const uint64_tt b[], uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 barrett_dim(n / poly_block, mod_num);
    barrett_batch_3param_kernel <<< barrett_dim, poly_block >>> (c, a, b, n, idx_c, idx_a, idx_b, idx_mod, mod_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_add_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_poly = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n + idx_in_poly * blockDim.y * n;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[i + idx_a * n] + device_b[i + idx_b * n];
    csub_q(ra, q);
    device_a[i + idx_a * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + (idx_in_q + idx_in_cipher * q_num) * n;

    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];
    register uint64_tt ra = device_a[i] + device_b[i];
    csub_q(ra, q);
    device_a[i] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_axbx_batch_device_kernel(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + idx_in_q * n;
    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];

    register uint64_tt ra = cipher_device[i] + ax_device[i];
    csub_q(ra, q);
    cipher_device[i] = ra;

    ra = cipher_device[i + q_num*n] + bx_device[i];
    csub_q(ra, q);
    cipher_device[i + q_num*n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_add_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[i + idx_a * n] + device_b[i + idx_b * n];
    csub_q(ra, q);
    device_c[i + idx_c * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_add_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + (idx_in_q + idx_in_cipher * q_num) * n;

    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];
    register uint64_tt ra = device_a[i] + device_b[i];
    csub_q(ra, q);
    device_c[i] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_add_const_batch_device_kernel(uint64_tt* device_a, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_a, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_a) * n + idx_in_poly] + add_const_real_buffer[index];
    csub_q(ra, q);
    device_a[(index + idx_a) * n + idx_in_poly] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void cipher_negate_batch_device_kernel(uint64_tt* device_a, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly] = q - ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void cipher_negate_3param_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    device_b[(index + idx_in_cipher*q_num) * n + idx_in_poly] = q - ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_add_complex_const_batch_device_kernel(uint64_tt* device_a, uint64_tt* add_const_buffer, uint32_tt n, uint64_tt* psi_powers, uint64_tt* psi_powers_shoup, int idx_a, int L, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint128_tt mu_q = {pqt_mu_cons_high[idx_mod + index], pqt_mu_cons_low[idx_mod + index]};
    register uint64_tt ra = device_a[(index + idx_a) * n + idx_in_poly];
    register uint64_tt Nth_root = psi_powers[index * n + 1];
    register uint64_tt Nth_root_shoup = psi_powers_shoup[index * n + 1];

    register uint64_tt temp = mulMod_shoup(add_const_buffer[index + (L+1)], Nth_root, Nth_root_shoup, q);

    if(idx_in_poly < (n >> 1))
        ra = ra + add_const_buffer[index] + temp;
    else
        ra = ra + add_const_buffer[index] + q - temp;
    csub_q(ra, q);
    device_a[(index + idx_a) * n + idx_in_poly] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_const_batch_device_kernel(uint64_tt* device_a, uint64_tt* const_real, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    register uint64_tt rb = const_real[index];
    register uint64_tt rb_shoup = const_real[index + q_num*2];

    device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly] = mulMod_shoup(ra, rb, rb_shoup, q);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_const_batch_andAdd_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* const_real, uint32_tt n, int q_num, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_cipher = blockIdx.z;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt rb = device_b[(index + idx_in_cipher*q_num) * n + idx_in_poly];

    register uint64_tt rc = const_real[index];
    register uint64_tt rc_shoup = const_real[index + q_num*2];

    register uint64_tt ra = mulMod_shoup(rb, rc, rc_shoup, q);

    ra += device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly];
    csub_q(ra, q);
    device_a[(index + idx_in_cipher*q_num) * n + idx_in_poly] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_sub_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint64_tt ra = q + device_a[i + idx_a * n] - device_b[i + idx_b * n];
    csub_q(ra, q);
    device_a[i + idx_a * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_sub2_batch_device_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint64_tt ra = q + device_a[i + idx_a * n] - device_b[i + idx_b * n];
    csub_q(ra, q);
    device_b[i + idx_a * n] = ra;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_sub_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[idx_mod + index];
    register uint64_tt ra = q + device_a[i + idx_a * n] - device_b[i + idx_b * n];
    csub_q(ra, q);
    device_c[i + idx_c * n] = ra;
}

// a = a + b
__host__ void poly_add_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 add_dim(n / poly_block , mod_num);
    poly_add_batch_device_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

// a = a + b
__host__ void cipher_add_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 add_dim(n / poly_block , mod_num, 2);
    cipher_add_batch_device_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_mod, q_num);
}

// a = a + b
__host__ void cipher_add_axbx_batch_device(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 add_dim(n / poly_block, mod_num);
    cipher_add_axbx_batch_device_kernel<<< add_dim, poly_block >>>(cipher_device, ax_device, bx_device, n, idx_mod, q_num);
}

// c = a + b
__host__ void poly_add_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 add_dim(n / poly_block , mod_num);
    poly_add_3param_batch_device_kernel<<< add_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c, idx_a, idx_b, idx_mod);
}

// a = a + b
__host__ void cipher_add_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 add_dim(n / poly_block , mod_num, 2);
    cipher_add_3param_batch_device_kernel<<< add_dim, poly_block >>>(device_c, device_a, device_b, n, idx_mod, q_num);
}

__host__ void poly_add_real_const_batch_device(uint64_tt* device_a, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_a, int idx_mod, int mod_num)
{
    dim3 add_dim(n / poly_block , mod_num);
    poly_add_const_batch_device_kernel<<< add_dim, poly_block >>>(device_a, add_const_real_buffer, n, idx_a, idx_mod);
}

__host__ void cipher_negate_batch_device(uint64_tt* device_a, uint32_tt n, int q_num, int idx_mod, int mod_num)
{
    dim3 negate_dim(n / poly_block , mod_num, 2);
    cipher_negate_batch_device_kernel<<< negate_dim, poly_block >>>(device_a, n, q_num, idx_mod);
}

__host__ void cipher_negate_3param_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int q_num, int idx_mod, int mod_num)
{
    dim3 negate_dim(n / poly_block , mod_num, 2);
    cipher_negate_3param_batch_device_kernel<<< negate_dim, poly_block >>>(device_a, device_b, n, q_num, idx_mod);
}

// a = a - b
__host__ void poly_sub_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 sub_dim(n / poly_block , mod_num);
    poly_sub_batch_device_kernel<<< sub_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

// b = a - b
__host__ void poly_sub2_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 sub_dim(n / poly_block , mod_num);
    poly_sub2_batch_device_kernel<<< sub_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

// c = a - b
__host__ void poly_sub_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num)
{
    dim3 sub_dim(n / poly_block , mod_num);
    poly_sub_3param_batch_device_kernel<<< sub_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c, idx_a, idx_b, idx_mod);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void poly_add_axbx_double_add_cnst_batch_kernel(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_mod, int q_num)
{
    register uint32_tt idx_in_q = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + idx_in_q * n;
    register uint64_tt q = pqt_cons[idx_mod + idx_in_q];

    register uint64_tt ra = cipher_device[i] + ax_device[i];
    csub_q(ra, q);
    ra += ra;
    csub_q(ra, q);
    cipher_device[i] = ra;

    ra = cipher_device[i + q_num*n] + bx_device[i];
    csub_q(ra, q);
    ra += ra;
    csub_q(ra, q);
    ra += add_const_real_buffer[idx_in_q];
    csub_q(ra, q);
    cipher_device[i + q_num*n] = ra;
}

__host__ void poly_add_axbx_double_add_cnst_batch_device(uint64_tt* cipher_device, uint64_tt* ax_device, uint64_tt* bx_device, uint64_tt* add_const_real_buffer, uint32_tt n, int idx_mod, int mod_num, int q_num)
{
    dim3 this_block(n / poly_block, mod_num);
    poly_add_axbx_double_add_cnst_batch_kernel <<< this_block, poly_block >>> (cipher_device, ax_device, bx_device, add_const_real_buffer, n, idx_mod, q_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sxsx_mul_P_3param_kernel(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);
    register uint64_tt ra = device_a[i + idx_a * n];

    register uint128_tt rc;

    mul64(ra, Pmodqi_cons[index + idx_mod - K], rc);
    singleBarrett_new(rc, q, mu);

    device_c[i + idx_c * n] = rc.low;
}

__host__ void sxsx_mul_P_3param(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int mod_num)
{
    dim3 mul_dim(n / poly_block , mod_num);
    sxsx_mul_P_3param_kernel <<< mul_dim, poly_block >>> (device_c, device_a, n, idx_c, idx_a, idx_mod, K);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_LeftRot_kernel(uint64_tt* device_a, uint64_tt* device_b,  uint64_tt*rotGroup, uint32_tt n, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
	register long pow = rotGroup[rot_num];
	register uint64_tt* ai = device_a + (idx_a + idx_in_pq + idx_in_cipher * q_num) * n;
	register uint64_tt* bi = device_b + (idx_b + idx_in_pq + idx_in_cipher * q_num) * n;
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
	register long shift = (global_tid * pow) & (2 * n - 1);
	if(shift < n)
        ai[shift] = bi[global_tid];
	else 
		ai[shift - n] = pqt_cons[idx_in_pq + p_num] - bi[global_tid];
}

// __global__
// __launch_bounds__(
//     POLY_MAX_THREADS, 
//     POLY_MIN_BLOCKS) 
// void sk_and_poly_LeftRot_kernel(uint64_tt* device_a, uint64_tt* device_b,  uint64_tt*rotGroup, uint32_tt n, uint32_tt K, uint32_tt l, uint32_tt rot_num, int idx_a, int idx_b)
// {
//     register uint32_tt index = blockIdx.y;
// 	register long pow = rotGroup[rot_num];
// 	register uint64_tt* ai = device_a + idx_a * n + (index * n);
// 	register uint64_tt* bi = device_b + idx_b * n + (index * n);
// 	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
// 	register long npow = global_tid * pow;
// 	register long shift = npow % (2 * n);
// 	if(shift < n)
//         ai[shift] = bi[global_tid];
// 	else 
// 		ai[shift - n] = pq_cons[index + K] - bi[global_tid];
// }

// __host__ void sk_and_poly_LeftRot(uint64_tt* device_a, uint64_tt* device_b, uint64_tt*rotGroup, uint32_tt n, uint32_tt K, uint32_tt L, uint32_tt rot_num, int idx_a, int idx_b, int mod_num)
// {
//     dim3 leftrot_dim(n / poly_block , mod_num);
//     sk_and_poly_LeftRot_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, K, L, rot_num, idx_a, idx_b);
// }

__host__ void sk_and_poly_LeftRot_double(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* rotGroup, uint32_tt n, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b, int mod_num)
{
    dim3 leftrot_dim(n / poly_block , mod_num, 2);
    sk_and_poly_LeftRot_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, p_num, q_num, rot_num, idx_a, idx_b);
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_LeftRot_inv_kernel(uint64_tt* device_a, uint64_tt* device_b,  uint64_tt*rotGroup, uint32_tt n, uint32_tt K, uint32_tt rot_num, int idx_a, int idx_b)
{
    register uint32_tt index = blockIdx.y;
	register long pow = rotGroup[(n>>1) - rot_num];
	register uint64_tt* ai = device_a + idx_a * n + (index * n);
	register uint64_tt* bi = device_b + idx_b * n + (index * n);
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
	register long npow = global_tid * pow;
	register long shift = npow & (2 * n - 1);
	if(shift < n)
        ai[shift] = bi[global_tid];
	else 
		ai[shift - n] = pqt_cons[index + K] - bi[global_tid];
}

__host__ void sk_and_poly_LeftRot_inv(uint64_tt* device_a, uint64_tt* device_b, uint64_tt*rotGroup, uint32_tt n, uint32_tt K, uint32_tt rot_num, int idx_a, int idx_b, int mod_num)
{
    dim3 leftrot_dim(n / poly_block , mod_num);
    sk_and_poly_LeftRot_inv_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, K, rot_num, idx_a, idx_b);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_conjugate_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, uint32_tt q_num, int idx_a, int idx_b)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    // register uint32_tt idx_in_cipher = blockIdx.z;
	// register uint64_tt* ai = device_a + (idx_a + idx_in_pq + idx_in_cipher * q_num) * n;
	// register uint64_tt* bi = device_b + (idx_b + idx_in_pq + idx_in_cipher * q_num) * n;
	register uint64_tt* ai = device_a + (idx_a + idx_in_pq) * n;
	register uint64_tt* bi = device_b + (idx_b + idx_in_pq) * n;

	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
	ai[global_tid] = bi[n - 1 - global_tid];
}
__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_conjugate_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b)
{
    register uint32_tt index = blockIdx.y;
	register uint64_tt* ai = device_a + idx_a * n + (index * n);
	register uint64_tt* bi = device_b + idx_b * n + (index * n);
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
	ai[global_tid] = bi[n - 1 - global_tid];
}

__host__ void sk_and_poly_conjugate(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, uint32_tt q_num, int idx_a, int idx_b, int mod_num)
{ 
    dim3 conjugate_dim(n / poly_block , mod_num);
    sk_and_poly_conjugate_kernel <<< conjugate_dim, poly_block >>> (device_a, device_b, n, q_num, idx_a, idx_b);
}
__host__ void sk_and_poly_conjugate(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n,int idx_a, int idx_b, int mod_num)
{ 
    dim3 conjugate_dim(n / poly_block , mod_num);
    sk_and_poly_conjugate_kernel <<< conjugate_dim, poly_block >>> (device_a, device_b, n, idx_a, idx_b);
}

// __host__ void sk_and_poly_conjugate_double(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int q_num, int idx_a, int idx_b, int mod_num)
// { 
//     dim3 conjugate_dim(n / poly_block , mod_num, 2);
//     sk_and_poly_conjugate_kernel <<< conjugate_dim, poly_block >>> (device_a, device_b, n, q_num, idx_a, idx_b);
// }

__global__
__launch_bounds__(
    POLY_MAX_THREADS,
    POLY_MIN_BLOCKS)
void divByiAndEqual_kernel(uint64_tt* device_a, uint32_tt n, uint32_tt q_num, int idx_mod, uint64_tt psi_powers[])
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
	
    register uint64_tt q = pqt_cons[idx_mod + idx_in_pq];
    register uint128_tt q_mu = {pqt_mu_cons_high[idx_mod + idx_in_pq], pqt_mu_cons_low[idx_mod + idx_in_pq]};

    register uint64_tt ra = device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly];
    register uint64_tt rb;
    if(idx_in_poly < (n>>1))
        // 4throot of Zq
        rb = q - psi_powers[(idx_in_pq) * n + 1];
    else
        // -4throot of Zq
        rb = psi_powers[(idx_in_pq) * n + 1];

    register uint128_tt temp;
    mul64(ra, rb, temp);
    singleBarrett_new(temp, q, q_mu);

    device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly] = temp.low;
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS,
    POLY_MIN_BLOCKS)
void mulByiAndEqual_kernel(uint64_tt* device_a, uint32_tt n, uint32_tt q_num, int idx_mod, uint64_tt psi_powers[])
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register int idx_in_cipher = blockIdx.z;
	
    register uint64_tt q = pqt_cons[idx_mod + idx_in_pq];
    register uint128_tt q_mu = {pqt_mu_cons_high[idx_mod + idx_in_pq], pqt_mu_cons_low[idx_mod + idx_in_pq]};

    register uint64_tt ra = device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly];
    register uint64_tt rb;
    if(idx_in_poly >= (n>>1))
        // 4throot of Zq
        rb = q - psi_powers[(idx_in_pq) * n + 1];
    else
        // -4throot of Zq
        rb = psi_powers[(idx_in_pq) * n + 1];

    register uint128_tt temp;
    mul64(ra, rb, temp);
    singleBarrett_new(temp, q, q_mu);

    device_a[(idx_in_cipher*q_num + idx_in_pq) * n + idx_in_poly] = temp.low;
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void compute_c0c1c2_kernel(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a1, uint64_tt* a2, uint64_tt* b1, uint64_tt* b2, int n, int idx_poly, int idx_mod)
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register uint64_tt ra1 = a1[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt ra2 = a2[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt rb1 = b1[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt rb2 = b2[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu_q = {pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]};

    register uint128_tt t1, t2, t3;
    mul64(ra1, ra2, t1);    // a1a2
    mul64(rb1, rb2, t2);    // b1b2

    // a1+b1
    ra1 = ra1 + rb1;
    // a2+b2
    ra2 = ra2 + rb2;
    mul64(ra1, ra2, t3);
    // t3 = t3 - t1 - t2;      // a1b2 + a2b1
    sub_uint128_uint128(t3, t1);
    sub_uint128_uint128(t3, t2);

    singleBarrett_new(t1, q, mu_q);
    singleBarrett_new(t2, q, mu_q);
    singleBarrett_new(t3, q, mu_q);

    axax_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t1.low;
    bxbx_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t2.low;
    a1b2a2b1_mul[idx_in_poly + (idx_in_pq + idx_poly) * n] = t3.low;
}

__host__ void compute_c0c1c2(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a1, uint64_tt* a2, uint64_tt* b1, uint64_tt* b2, int n, int idx_poly, int idx_mod, int mod_num)
{
    dim3 compute_c0c1c2_dim(n / poly_block, mod_num);
    compute_c0c1c2_kernel <<< compute_c0c1c2_dim, poly_block >>> (a1b2a2b1_mul, axax_mul, bxbx_mul, a1, a2, b1, b2, n, idx_poly, idx_mod);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void compute_c0c1c2_square_kernel(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a, uint64_tt* b, int n, int idx_poly, int idx_mod)
{
    register int idx_in_poly = blockIdx.x * poly_block + threadIdx.x;
    register int idx_in_pq = blockIdx.y;
    register uint64_tt ra = a[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt rb = b[idx_in_poly + (idx_in_pq + idx_poly) * n];
    register uint64_tt q = pqt_cons[idx_in_pq + idx_mod];
    register uint128_tt mu_q = {pqt_mu_cons_high[idx_in_pq + idx_mod], pqt_mu_cons_low[idx_in_pq + idx_mod]};

    register uint128_tt t1, t2, t3;
    mul64(ra, ra, t1);    // aa
    mul64(rb, rb, t2);    // bb
    mul64(ra, 2*rb, t3);  // 2ab

    singleBarrett_new(t1, q, mu_q);
    singleBarrett_new(t2, q, mu_q);
    singleBarrett_new(t3, q, mu_q);

    axax_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t1.low;
    bxbx_mul[    idx_in_poly + (idx_in_pq + idx_poly) * n] = t2.low;
    a1b2a2b1_mul[idx_in_poly + (idx_in_pq + idx_poly) * n] = t3.low;
}

__host__ void compute_c0c1c2_square(uint64_tt* a1b2a2b1_mul, uint64_tt* axax_mul, uint64_tt* bxbx_mul, uint64_tt* a, uint64_tt* b, int n, int idx_poly, int idx_mod, int mod_num)
{
    dim3 compute_c0c1c2_dim(n / poly_block, mod_num);
    compute_c0c1c2_square_kernel <<< compute_c0c1c2_dim, poly_block >>> (a1b2a2b1_mul, axax_mul, bxbx_mul, a, b, n, idx_poly, idx_mod);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_mul_P_kernel(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K)
{
    register uint32_tt index = blockIdx.y;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);
    register uint64_tt ra = device_a[i + idx_a * n];

    register uint128_tt rc;

    mul64(ra, Pmodqi_cons[index + idx_mod - K], rc);
    singleBarrett_new(rc, q, mu);

    device_c[i + idx_c * n] = rc.low;
}

__host__ void cipher_mul_P(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int mod_num)
{
    dim3 mul_dim(n / poly_block , mod_num);
    cipher_mul_P_kernel <<< mul_dim, poly_block >>> (device_c, device_a, n, idx_c, idx_a, idx_mod, K);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int poly_num)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;

    register uint64_tt ra = device_a[i + idx_a * n + idx_in_cipher * poly_num * n];
    register uint64_tt rb = device_b[i + idx_b * n];

    register uint128_tt rc;

    mul64(ra, rb, rc);
    singleBarrett_new(rc, q, mu);

    device_c[i + idx_c * n + idx_in_cipher * poly_num * n]=rc.low;
}

__host__ void poly_add_batch_device_many_poly(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_a, int idx_b, int idx_mod, int mod_num, int poly_num)
{
    dim3 add_dim(n / poly_block , mod_num, poly_num);
    poly_add_batch_device_kernel<<< add_dim, poly_block >>>(device_a, device_b, n, idx_a, idx_b, idx_mod);
}

__host__ void poly_mul_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num, int poly_num)
{
    dim3 mul_dim(n / poly_block , mod_num, 2);
    poly_mul_3param_batch_device_kernel<<< mul_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c,idx_a, idx_b, idx_mod, poly_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void sk_and_poly_LeftRot_many_poly_kernel(uint64_tt* device_a, uint64_tt* device_b,  uint64_tt*rotGroup, uint32_tt n, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register uint32_tt idx_in_poly = blockIdx.z;
	register long pow = rotGroup[rot_num];
	register uint64_tt* ai = device_a + (idx_a + idx_in_pq + idx_in_poly * blockDim.y) * n;
	register uint64_tt* bi = device_b + (idx_b + idx_in_pq + idx_in_poly * blockDim.y) * n;
	register int global_tid = blockIdx.x * poly_block + threadIdx.x;
	register long shift = (global_tid * pow) & (2 * n - 1);
	if(shift < n)
        ai[shift] = bi[global_tid];
	else 
		ai[shift - n] = pqt_cons[idx_in_pq + p_num + q_num] - bi[global_tid];
}

__host__ void sk_and_poly_LeftRot_many_poly_T(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* rotGroup, uint32_tt n, uint32_tt p_num, uint32_tt q_num, uint32_tt rot_num, int idx_a, int idx_b, int mod_num, int poly_num)
{
    dim3 leftrot_dim(n / poly_block , mod_num, poly_num * 2);
    sk_and_poly_LeftRot_many_poly_kernel <<< leftrot_dim, poly_block >>> (device_a, device_b, rotGroup, n, p_num, q_num, rot_num, idx_a, idx_b);
}


__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS) 
void cipher_mul_P_special_kernel(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int q_num, int target_num)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
    register int j = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n + idx_in_cipher * q_num * n;

    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);
    register uint64_tt ra = device_a[i + idx_a * n];

    register uint128_tt rc;

    mul64(ra, Pmodqi_cons[index + idx_mod - K], rc);
    singleBarrett_new(rc, q, mu);

    device_c[j + K * n + idx_in_cipher * target_num * n] = rc.low;
}

__host__ void cipher_mul_P_special(uint64_tt* device_c, uint64_tt* device_a, uint32_tt n, int idx_c, int idx_a, int idx_mod, int K, int mod_num, int q_num, int target_num)
{
    dim3 mul_dim(n / poly_block , mod_num, 2);
    cipher_mul_P_special_kernel <<< mul_dim, poly_block >>> (device_c, device_a, n, idx_c, idx_a, idx_mod, K, q_num, target_num);
}

__global__
__launch_bounds__(
    POLY_MAX_THREADS, 
    POLY_MIN_BLOCKS)
void poly_mul_add_3param_batch_device_kernel(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int poly_num, int block_size)
{
    register uint32_tt index = blockIdx.y;
    register uint32_tt idx_in_cipher = blockIdx.z;
    register uint64_tt q = pqt_cons[index + idx_mod];
    register uint128_tt mu(pqt_mu_cons_high[index + idx_mod], pqt_mu_cons_low[index + idx_mod]);

    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;
    register uint64_tt rb = device_b[i + idx_b * n];

    for(int k = 0; k < block_size; k++){
        register uint64_tt ra = device_a[i + idx_a * n + idx_in_cipher * poly_num * n + k * blockDim.y * n];
        register uint128_tt rc;
        mul64(ra, rb, rc);
        singleBarrett_new(rc, q, mu);
        register uint64_tt rc_add = device_c[i + idx_c * n + idx_in_cipher * poly_num * n + k * blockDim.y * n];
        device_c[i + idx_c * n + idx_in_cipher * poly_num * n + k * blockDim.y * n] = rc.low + rc_add;
    }

}

//c=a*b+c
__host__ void poly_mul_add_3param_batch_device(uint64_tt* device_c, uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int idx_c, int idx_a, int idx_b, int idx_mod, int mod_num, int poly_num, int block_size)
{
    dim3 mul_dim(n / poly_block , mod_num, 2);
    poly_mul_add_3param_batch_device_kernel<<< mul_dim, poly_block >>>(device_c, device_a, device_b, n, idx_c,idx_a, idx_b, idx_mod, poly_num, block_size);
}


//bfv encrypt
//cipher += m * delta
__global__ void bfv_add_timesQ_overt_kernel(uint64_tt *ct,
                                            const uint64_tt *pt,
                                            uint64_tt negQl_mod_t,
                                            uint64_tt negQl_mod_t_shoup,
                                            const uint64_tt *tInv_mod_q,
                                            const uint64_tt *tInv_mod_q_shoup,
                                            const uint64_tt *modulus_Ql,
                                            uint64_tt t,
                                            uint64_tt n,
                                            uint64_tt size_Ql) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n * size_Ql;
         tid += blockDim.x * gridDim.x)
    {
        size_t twr = tid / n;
        uint64_tt qi = modulus_Ql[twr];
        uint64_tt tInv_mod_qi = tInv_mod_q[twr];
        uint64_tt tInv_mod_qi_shoup = tInv_mod_q_shoup[twr];

        uint64_tt m = pt[tid];
        ct[tid] = 0;
        uint64_tt mQl_mod_t = multiply_and_reduce_shoup(m, negQl_mod_t, negQl_mod_t_shoup, t);
        ct[tid] += multiply_and_reduce_shoup(mQl_mod_t, tInv_mod_qi, tInv_mod_qi_shoup, qi);

        if (ct[tid] >= qi) ct[tid] -= qi;
    }
}
//bfv decrypt
__global__ void hps_decrypt_scale_and_round_kernel_small(uint64_tt *dst, const uint64_tt *src,
    const uint64_tt *t_QHatInv_mod_q_div_q_mod_t,
    const uint64_tt *t_QHatInv_mod_q_div_q_mod_t_shoup,
    const double *t_QHatInv_mod_q_div_q_frac, uint64_tt t,
    size_t n, size_t size_Ql)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x)
    {
        double doubleSum = 0.0;
        uint64_tt intSum = 0;
        uint64_tt tmp;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++)
        {
            tmp = src[i * n + tid];
            doubleSum += static_cast<double>(tmp) * t_QHatInv_mod_q_div_q_frac[i];
            intSum += multiply_and_reduce_shoup(tmp, t_QHatInv_mod_q_div_q_mod_t[i],
            t_QHatInv_mod_q_div_q_mod_t_shoup[i], t);
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        doubleSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_tt>(doubleSum * tInv);
        doubleSum -= static_cast<double>(t * quot);
        // rounding
        dst[tid] = llround(doubleSum);
        
        // uint64_tt res = llround(doubleSum);
        // #pragma unroll
        // for (size_t i = 0; i < size_Ql; i++)
        // {
        //     dst[i * n + tid] = res;
        // }
    }
}

__global__ void hps_decrypt_scale_and_round_kernel_small_lazy(uint64_tt *dst, const uint64_tt *src,
    const uint64_tt *t_QHatInv_mod_q_div_q_mod_t,
    const double *t_QHatInv_mod_q_div_q_frac, uint64_tt t,
    size_t n, size_t size_Ql)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x)
    {
        double doubleSum = 0.0;
        uint64_tt intSum = 0;
        uint64_tt tmp;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++)
        {
            tmp = src[i * n + tid];
            doubleSum += static_cast<double>(tmp) * t_QHatInv_mod_q_div_q_frac[i];
            intSum += tmp * t_QHatInv_mod_q_div_q_mod_t[i];
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        doubleSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_tt>(doubleSum * tInv);
        doubleSum -= static_cast<double>(t * quot);
        // rounding
        uint64_tt res = llround(doubleSum);
        #pragma unroll
        for (size_t i = 0; i < size_Ql; i++)
        {
            dst[i * n + tid] = res;
        }
    }
}

__global__ void hps_decrypt_scale_and_round_kernel_large(
uint64_tt *dst, const uint64_tt *src, const uint64_tt *t_QHatInv_mod_q_div_q_mod_t,
const uint64_tt *t_QHatInv_mod_q_div_q_mod_t_shoup, const double *t_QHatInv_mod_q_div_q_frac,
const uint64_tt *t_QHatInv_mod_q_B_div_q_mod_t, const uint64_tt *t_QHatInv_mod_q_B_div_q_mod_t_shoup,
const double *t_QHatInv_mod_q_B_div_q_frac, uint64_tt t, size_t n, size_t size_Ql, size_t qMSBHf)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x)
    {
        double doubleSum = 0.0;
        uint64_tt intSum = 0;
        uint64_tt tmpLo, tmpHi;
        double tInv = 1. / static_cast<double>(t);

        for (size_t i = 0; i < size_Ql; i++)
        {
            uint64_tt tmp = src[i * n + tid];
            tmpHi = tmp >> qMSBHf;
            tmpLo = tmp & ((1ULL << qMSBHf) - 1);
            doubleSum += static_cast<double>(tmpLo) * t_QHatInv_mod_q_div_q_frac[i];
            doubleSum += static_cast<double>(tmpHi) * t_QHatInv_mod_q_B_div_q_frac[i];
            intSum += multiply_and_reduce_shoup(tmpLo, t_QHatInv_mod_q_div_q_mod_t[i],
            t_QHatInv_mod_q_div_q_mod_t_shoup[i], t);
            intSum += multiply_and_reduce_shoup(tmpHi, t_QHatInv_mod_q_B_div_q_mod_t[i],
            t_QHatInv_mod_q_B_div_q_mod_t_shoup[i], t);
        }
        // compute modulo reduction by finding the quotient using doubles
        // and then subtracting quotient * t
        doubleSum += static_cast<double>(intSum);
        auto quot = static_cast<uint64_tt>(doubleSum * tInv);
        doubleSum -= static_cast<double>(t * quot);
        // rounding
        uint64_tt res = llround(doubleSum);
        #pragma unroll
        for (size_t i = 0; i < size_Ql; i++)
        {
            dst[i * n + tid] = res;
        }
    }
}

__global__ void hps_decrypt_scale_and_round_kernel_large_lazy(uint64_tt *dst, const uint64_tt *src,
    const uint64_tt *t_QHatInv_mod_q_div_q_mod_t,
    const double *t_QHatInv_mod_q_div_q_frac,
    const uint64_tt *t_QHatInv_mod_q_B_div_q_mod_t,
    const double *t_QHatInv_mod_q_B_div_q_frac,
    uint64_tt t, size_t n, size_t size_Ql, size_t qMSBHf) {
for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x)
{
    double doubleSum = 0.0;
    uint64_tt intSum = 0;
    uint64_tt tmpLo, tmpHi;
    double tInv = 1. / static_cast<double>(t);

    for (size_t i = 0; i < size_Ql; i++)
    {
        uint64_tt tmp = src[i * n + tid];
        tmpHi = tmp >> qMSBHf;
        tmpLo = tmp & ((1ULL << qMSBHf) - 1);
        doubleSum += static_cast<double>(tmpLo) * t_QHatInv_mod_q_div_q_frac[i];
        doubleSum += static_cast<double>(tmpHi) * t_QHatInv_mod_q_B_div_q_frac[i];
        intSum += tmpLo * t_QHatInv_mod_q_div_q_mod_t[i];
        intSum += tmpHi * t_QHatInv_mod_q_B_div_q_mod_t[i];
    }
    // compute modulo reduction by finding the quotient using doubles
    // and then subtracting quotient * t
    doubleSum += static_cast<double>(intSum);
    auto quot = static_cast<uint64_tt>(doubleSum * tInv);
    doubleSum -= static_cast<double>(t * quot);
    // rounding
    uint64_tt res = llround(doubleSum);
    #pragma unroll
    for (size_t i = 0; i < size_Ql; i++)
    {
        dst[i * n + tid] = res;
    }
    }
}

__global__ void bconv_mult_unroll2_kernel(uint64_tt *dst, const uint64_tt *src, const uint64_tt *scale,
                                          const uint64_tt *scale_shoup, const uint64_tt *base, size_t base_size,
                                          size_t n)
{
    constexpr const int unroll_factor = 2;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < base_size * n / unroll_factor;
         tid += blockDim.x * gridDim.x) {
        size_t i = tid / (n / unroll_factor);
        size_t coeff_idx = tid * unroll_factor;
        auto modulus = base[i];
        auto scale_factor = scale[i];
        auto scale_factor_shoup = scale_shoup[i];
        uint64_tt in_x, in_y;
        uint64_tt out_x, out_y;

        ld_two_uint64(in_x, in_y, src + coeff_idx);
        out_x = multiply_and_reduce_shoup(in_x, scale_factor, scale_factor_shoup, modulus);
        out_y = multiply_and_reduce_shoup(in_y, scale_factor, scale_factor_shoup, modulus);
        st_two_uint64(dst + coeff_idx, out_x, out_y);
    }
}

__forceinline__ __device__ auto base_convert_acc_unroll2(const uint64_tt *ptr, const uint64_tt *QHatModp,
                                                         size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                         size_t degree_idx)
{
    uint128_t2 accum{0};
    for (int i = 0; i < ibase_size; i++)
    {
        const uint64_tt op2 = QHatModp[out_prime_idx * ibase_size + i];
        uint128_t2 out{};

        uint64_tt op1_x, op1_y;
        ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = multiply_uint64_uint64(op1_x, op2);
        // add_uint128_uint128(out.x, accum.x, accum.x);
        add_uint128_uint128(accum.x, out.x);
        out.y = multiply_uint64_uint64(op1_y, op2);
        // add_uint128_uint128(out.y, accum.y, accum.y);
        add_uint128_uint128(accum.y, out.y);
    }
    return accum;
}
__forceinline__ __device__ auto base_convert_acc_frac_unroll2(const uint64_tt *ptr, const double *qiInv,
                                                              size_t degree, size_t ibase_size,
                                                              size_t degree_idx)
{
    double_t2 accum{0};
    for (int i = 0; i < ibase_size; i++)
    {
        const double op2 = qiInv[i];

        uint64_tt op1_x, op1_y;
        ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;
    }
    return accum;
}

__global__ static void base_convert_matmul_hps_unroll2_kernel(uint64_tt *dst, const uint64_tt *xi_qiHatInv_mod_qi,
                                                              const uint64_tt *qiHat_mod_pj, const uint64_tt *v_Q_mod_pj,
                                                              const double *qiInv, 
                                                              const uint64_tt *obaseMu_high, const uint64_tt *obaseMu_low,
                                                              const uint64_tt *ibase, 
                                                              size_t ibase_size, const uint64_tt *obase,
                                                              size_t obase_size, size_t n)
{
    constexpr const int unroll_number = 2;
    extern __shared__ uint64_tt s_qiHat_mod_pj[];
    for (size_t i = threadIdx.x; i < obase_size * ibase_size; i += blockDim.x)
    {
        s_qiHat_mod_pj[i] = qiHat_mod_pj[i];
    }
    __syncthreads();

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n / unroll_number;
         tid += blockDim.x * gridDim.x)
    {
        const size_t degree_idx = unroll_number * (tid / obase_size);
        const size_t out_prime_idx = tid % obase_size;

        uint128_t2 accum =
                base_convert_acc_unroll2(xi_qiHatInv_mod_qi, s_qiHat_mod_pj, out_prime_idx, n, ibase_size, degree_idx);

        double_t2 accum_frac = base_convert_acc_frac_unroll2(xi_qiHatInv_mod_qi, qiInv, n, ibase_size, degree_idx);

        uint64_tt obase_value = obase[out_prime_idx];
        // uint64_tt obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};
        uint64_tt obase_ratio[2] = {obaseMu_low[out_prime_idx], obaseMu_high[out_prime_idx]};

        uint64_tt out = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
        uint64_tt out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
        uint64_tt vQ_mod_pj = v_Q_mod_pj[llround(accum_frac.x) * obase_size + out_prime_idx];
        uint64_tt vQ_mod_pj2 = v_Q_mod_pj[llround(accum_frac.y) * obase_size + out_prime_idx];
        out = sub_uint64_uint64_mod(out, vQ_mod_pj, obase_value);
        out2 = sub_uint64_uint64_mod(out2, vQ_mod_pj2, obase_value);
        st_two_uint64(dst + out_prime_idx * n + degree_idx, out, out2);
    }
}

__global__ void tensor_prod_2x2_rns_poly(const uint64_tt *operand1,
                                         const uint64_tt *operand2,
                                         const uint64_tt *modulus,
                                         const uint64_tt *Mu_high,
                                         const uint64_tt *Mu_low, 
                                         uint64_tt *result,
                                         size_t poly_degree,
                                         size_t coeff_mod_size)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x)
    {
        size_t twr = tid / poly_degree;
        uint64_tt mod = modulus[twr];
        uint64_tt barrett_mu[2]{Mu_low[twr] , Mu_high[twr]};

        uint64_tt c0_0, c0_1, c1_0, c1_1;
        uint64_tt d0, d1, d2;
        uint64_tt rns_coeff_count = poly_degree * coeff_mod_size;

        c0_0 = operand1[tid];
        c0_1 = operand1[tid + rns_coeff_count];
        c1_0 = operand2[tid];
        c1_1 = operand2[tid + rns_coeff_count];

        // d0 <- c0 * c'0
        d0 = multiply_and_barrett_reduce_uint64(c0_0, c1_0, mod, barrett_mu);
        // d2 <- c1 * c'1
        d2 = multiply_and_barrett_reduce_uint64(c0_1, c1_1, mod, barrett_mu);
        // d1 <- (c0 + c1) * (c'0 + c'1) - c0 * c'0 - c1 * c'1
        d1 = multiply_and_barrett_reduce_uint64(c0_0 + c0_1, c1_0 + c1_1, mod, barrett_mu);
        d1 = d1 + 2 * mod - d0 - d2;
        if (d1 >= mod) d1 -= mod;
        if (d1 >= mod) d1 -= mod;

        result[tid] = d0;
        result[tid + rns_coeff_count] = d1;
        result[tid + 2 * rns_coeff_count] = d2;
    }
}
__global__ void tensor_prod_2x2_rns_poly_for_square(const uint64_tt *operand1,
                                         const uint64_tt *modulus,
                                         const uint64_tt *Mu_high,
                                         const uint64_tt *Mu_low, 
                                         uint64_tt *result,
                                         size_t poly_degree,
                                         size_t coeff_mod_size)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < poly_degree * coeff_mod_size;
         tid += blockDim.x * gridDim.x)
    {
        size_t twr = tid / poly_degree;
        uint64_tt mod = modulus[twr];
        uint64_tt barrett_mu[2]{Mu_low[twr] , Mu_high[twr]};

        uint64_tt c0_0, c0_1, c1_0, c1_1;
        uint64_tt d0, d1, d2;
        uint64_tt rns_coeff_count = poly_degree * coeff_mod_size;

        c0_0 = operand1[tid];
        c0_1 = operand1[tid + rns_coeff_count];
        c1_0 = operand1[tid];
        c1_1 = operand1[tid + rns_coeff_count];

        // d0 <- c0 * c'0
        d0 = multiply_and_barrett_reduce_uint64(c0_0, c1_0, mod, barrett_mu);
        // d2 <- c1 * c'1
        d2 = multiply_and_barrett_reduce_uint64(c0_1, c1_1, mod, barrett_mu);
        // d1 <- (c0 + c1) * (c'0 + c'1) - c0 * c'0 - c1 * c'1
        d1 = multiply_and_barrett_reduce_uint64(c0_0 + c0_1, c1_0 + c1_1, mod, barrett_mu);
        d1 = d1 + 2 * mod - d0 - d2;
        if (d1 >= mod) d1 -= mod;
        if (d1 >= mod) d1 -= mod;

        result[tid] = d0;
        result[tid + rns_coeff_count] = d1;
        result[tid + 2 * rns_coeff_count] = d2;
    }
}

// QR -> R
__global__ void scaleAndRound_HPS_QR_R_kernel(uint64_tt *dst, const uint64_tt *src,
                                                const uint64_tt *t_R_SHatInv_mod_s_div_s_mod_r,
                                                const double *t_R_SHatInv_mod_s_div_s_frac, const uint64_tt *base_Rl,
                                                const uint64_tt *RMu_high, const uint64_tt *RMu_low, 
                                                size_t n, size_t size_Ql, size_t size_Rl)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x)
    {
        auto src_Ql = src;
        auto src_Rl = src + size_Ql * n;

        double nu = 0.5;
        for (size_t i = 0; i < size_Ql; i++)
        {
            uint64_tt xi = src_Ql[i * n + tid];
            nu += static_cast<double>(xi) * t_R_SHatInv_mod_s_div_s_frac[i];
        }

        auto alpha = static_cast<uint64_tt>(nu);

        for (size_t j = 0; j < size_Rl; j++)
        {
            uint128_tt curValue = {0, 0};
            auto rj = base_Rl[j];
            uint64_tt rj_ratio[2] = { RMu_low[j], RMu_high[j] };
            // auto rj_ratio = base_Rl[j].const_ratio();
            auto t_R_SHatInv_mod_s_div_s_mod_rj = t_R_SHatInv_mod_s_div_s_mod_r + j * (size_Ql + 1);

            for (size_t i = 0; i < size_Ql; i++)
            {
                uint64_tt xi = src_Ql[i * n + tid];
                uint128_tt temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[i]);
                add_uint128_uint128(temp, curValue, curValue);
            }

            uint64_tt xi = src_Rl[j * n + tid];
            uint128_tt temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[size_Ql]);
            add_uint128_uint128(temp, curValue, curValue);

            uint64_tt curNativeValue = barrett_reduce_uint128_uint64(curValue, rj, rj_ratio);
            // alpha = barrett_reduce_uint64_uint64_sp(alpha, rj, rj_ratio[1]);
            barrett_reduce_uint64_uint64(alpha, rj, rj_ratio[1]);
            dst[j * n + tid] = add_uint64_uint64_mod(curNativeValue, alpha, rj);
        }
    }
}
void scaleAndRound_HPS_QR_R(uint64_tt *dst, const uint64_tt *src, 
        uint64_tt *base_R , uint64_tt *RMu_high, uint64_tt *RMu_low, 
        uint64_tt *tRSHatInvModsDivsModr_, double *tRSHatInvModsDivsFrac_, 
        size_t size_Q , size_t size_R , size_t N , const cudaStream_t &stream)
{
    uint64_tt gridDimGlb = N / blockDimGlb.x;
    scaleAndRound_HPS_QR_R_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            dst, src, tRSHatInvModsDivsModr_, tRSHatInvModsDivsFrac_, base_R, RMu_high , RMu_low , N, size_Q, size_R);
}