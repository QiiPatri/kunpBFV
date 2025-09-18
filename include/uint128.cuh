#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <cinttypes>
#include <string>
#include <math.h>
#include <NTL/ZZ.h>
#include <NTL/RR.h>

typedef unsigned char uint8_tt;
typedef unsigned int uint32_tt;
typedef unsigned long long uint64_tt;

class uint128_tt
{
public:
	
	uint64_tt low;
	uint64_tt high;

	__host__ __device__ __forceinline__ uint128_tt(uint64_tt high, uint64_tt low) : high(high), low(low)
	{}

	__host__ __device__ __forceinline__ uint128_tt()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_tt(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_tt& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}
};

// __host__ __device__ __forceinline__ uint128_tt operator+(const uint128_tt& x, const uint128_tt& y)
// {
// 	uint128_tt z;

// 	z.low = x.low + y.low;
// 	z.high = x.high + y.high + (z.low < x.low);

// 	return z;
// }

// __host__ __device__ __forceinline__ uint128_tt operator+(const uint128_tt& x, const uint64_t& y)
// {
// 	uint128_tt z;

// 	z.low = x.low + y;
// 	z.high = x.high + (z.low < x.low);

// 	return z;
// }

// __host__ __device__ __forceinline__ uint128_tt operator-(const uint128_tt& x, const uint128_tt& y)
// {
// 	uint128_tt z;

// 	z.low = x.low - y.low;
// 	z.high = x.high - y.high - (x.low < y.low);

// 	return z;
	
// }

// __host__ __device__ __forceinline__ void operator-=(uint128_tt& x, const uint128_tt& y)
// {
// 	x.high = x.high - y.high - (x.low < y.low);
// 	x.low = x.low - y.low;
// }

// __host__ __device__ __forceinline__ uint128_tt operator-(const uint128_tt& x, const uint64_t& y)
// {
// 	uint128_tt z;

// 	z.low = x.low - y;
// 	z.high = x.high - (x.low < y);

// 	return z;
// }

// c = a * b
__device__ __forceinline__ void mul64(const uint64_tt& a, const uint64_tt& b, uint128_tt& c)
{
	asm("{\n\t"
		"mad.lo.cc.u64		%1, %2, %3, 0;		\n\t"
		"madc.hi.u64		%0, %2, %3, 0;		\n\t"
		"}"
		: "=l"(c.high), "=l"(c.low)
		: "l"(a), "l"(b));
}

__forceinline__ __device__ void csub_q(uint64_tt& a, uint64_tt q) 
{
	register uint64_tt tmp = a - q;
	a = tmp + (tmp >> 63) * q;
}

// c = a * b \in [0, q)
__device__ __forceinline__ uint64_tt mulMod_shoup(const uint64_tt& a, const uint64_tt& b, const uint64_tt& b_shoup, uint64_tt& mod)
{
	uint64_tt hi = __umul64hi(a, b_shoup);
	uint64_tt ra = a * b - hi * mod;
	csub_q(ra, mod);
	return ra;
}

// from phantom-fhe
__device__ __forceinline__ void singleBarrett_new(uint128_tt& a, uint64_tt& q, uint128_tt& mu)
{
	uint64_tt result;
	asm(
		"{\n\t"
		" .reg .u64 	tmp;\n\t"
		// Multiply input and const_ratio
		// Round 1
		" mul.hi.u64 	tmp, %1, %3;\n\t"
		" mad.lo.cc.u64 tmp, %1, %4, tmp;\n\t"
		" madc.hi.u64 	%0, %1, %4, 0;\n\t"
		// Round 2
		" mad.lo.cc.u64 tmp, %2, %3, tmp;\n\t"
		" madc.hi.u64 	%0, %2, %3, %0;\n\t"
		// This is all we care about
		" mad.lo.u64 	%0, %2, %4, %0;\n\t"
		// Barrett subtraction
		" mul.lo.u64 	%0, %0, %5;\n\t"
		" sub.u64 		%0, %1, %0;\n\t"
		"}"
		: "=l"(result)
		: "l"(a.low), "l"(a.high), "l"(mu.low), "l"(mu.high), "l"(q));
	csub_q(result, q);
	a.high = 0;
	a.low = result;
}

__forceinline__ __device__ void barrett_reduce_uint64_uint64(uint64_tt& a,
															 uint64_tt& q,
															 const uint64_tt& mu_hi)
{
	uint64_tt s = __umul64hi(mu_hi, a);
	a = a - s * q;
	csub_q(a, q);
}

// __forceinline__ __device__ uint64_tt barrett_reduce_uint64_uint64_sp(const uint64_tt& operand,
// 																	const uint64_tt& modulus,
// 																	const uint64_tt& barrett_mu_hi)
// {
// 	uint64_tt s = __umul64hi(barrett_mu_hi, operand);
// 	uint64_tt result_ = operand - s * modulus;
// 	csub_q(result_, modulus);
// 	return result_;
// }

__forceinline__ __device__ uint128_tt multiply_uint64_uint64(const uint64_tt& operand1,
															const uint64_tt& operand2)
{
	uint128_tt result_;
	result_.low = operand1 * operand2;
	result_.high = __umul64hi(operand1, operand2);
	return result_;
}
__forceinline__ __device__ uint64_tt barrett_multiply_and_shift_uint128(const uint128_tt& operand1,
																		const uint128_tt& operand2) 
{
	uint64_tt p0 = __umul64hi(operand1.low, operand2.low);
	// !!!notice: volatile is necessary to avoid the incorrect compiler optimization!!!
	volatile uint128_tt p1 = multiply_uint64_uint64(operand1.low, operand2.high);
	volatile uint128_tt p2 = multiply_uint64_uint64(operand1.high, operand2.low);
	uint64_tt p3 = operand1.high * operand2.high;
	asm("add.cc.u64 %0, %0, %1;" : "+l"(p1.low) : "l"(p0));
	asm("addc.cc.u64 %0, %0, %1;" : "+l"(p2.high) : "l"(p1.high));
	asm("add.cc.u64 %0, %0, %1;" : "+l"(p2.low) : "l"(p1.low));
	asm("addc.cc.u64 %0, %0, %1;" : "+l"(p3) : "l"(p2.high));
	return p3;
}
__forceinline__ __device__ uint64_tt barrett_reduce_uint128_uint64(const uint128_tt& product,
																	const uint64_tt& modulus,
																	const uint64_tt* barrett_mu)
{
	uint64_tt result;
	uint64_tt q = modulus;

	//use CUDA PTX
	uint64_t lo = product.low;
	uint64_t hi = product.high;
	uint64_t ratio0 = barrett_mu[0];
	uint64_t ratio1 = barrett_mu[1];

	asm(
		"{\n\t"
		" .reg .u64 tmp;\n\t"
		// Multiply input and const_ratio
		// Round 1
		" mul.hi.u64 tmp, %1, %3;\n\t"
		" mad.lo.cc.u64 tmp, %1, %4, tmp;\n\t"
		" madc.hi.u64 %0, %1, %4, 0;\n\t"
		// Round 2
		" mad.lo.cc.u64 tmp, %2, %3, tmp;\n\t"
		" madc.hi.u64 %0, %2, %3, %0;\n\t"
		// This is all we care about
		" mad.lo.u64 %0, %2, %4, %0;\n\t"
		// Barrett subtraction
		" mul.lo.u64 %0, %0, %5;\n\t"
		" sub.u64 %0, %1, %0;\n\t"
		"}"
		: "=l"(result)
		: "l"(lo), "l"(hi), "l"(ratio0), "l"(ratio1), "l"(q));

	// uint128_tt barrett_mu_uint128;
	// barrett_mu_uint128.high = barrett_mu[1];
	// barrett_mu_uint128.low = barrett_mu[0];
	// const uint64_tt s = barrett_multiply_and_shift_uint128(product, barrett_mu_uint128);
	// result = product.low - s * modulus;

	csub_q(result, q);
	return result;
}

// #define singleBarrett_qq 0
// __device__ __forceinline__ void singleBarrett_new(uint128_tt &a, uint64_tt q, uint128_tt mu)
// {
//     uint64_tt res;
//     asm("{\n\t"
//         "mul.hi.u64       %0, %2, %3;  		    \n\t"
//         "mad.hi.u64       %0, %1, %4, %0;		\n\t"
//         "mad.lo.u64       %0, %1, %3, %0;   	\n\t"
//         "mul.lo.u64       %0, %0, %5;      		\n\t"
//         "sub.u64          %0, %2, %0;      		\n\t"
//         "}"
//         : "=l"(res)
//         : "l"(a.high), "l"(a.low), "l"(mu.high), "l"(mu.low), "l"(q));
// 	csub_q(res, q<<1);
// 	csub_q(res, q);
//     a.high = 0;
//     a.low = res;
// }

__forceinline__ __device__ uint64_tt sub_uint64_uint64_mod(const uint64_tt& operand1,
															const uint64_tt& operand2,
															const uint64_tt& modulus)
{
	uint64_tt result_ = (operand1 + modulus) - operand2;
	csub_q(result_, modulus);
	return result_;
}

__device__ __forceinline__ void sub_uint128_uint128(uint128_tt& a, const uint128_tt& b)
{
	asm("{\n\t"
		"sub.cc.u64      %1, %3, %5;    \n\t"
		"subc.u64        %0, %2, %4;    \n\t"
		"}"
		: "=l"(a.high), "=l"(a.low)
		: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));
}

__device__ __forceinline__ void add_uint128_uint128(uint128_tt& a, const uint128_tt& b)
{
	asm("{\n\t"
		"add.cc.u64      %1, %3, %5;    \n\t"
		"addc.u64        %0, %2, %4;    \n\t"
		"}"
		: "=l"(a.high), "=l"(a.low)
		: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));
}

__forceinline__ __device__ uint64_tt add_uint64_uint64_mod(const uint64_tt& operand1,
															const uint64_tt& operand2,
															const uint64_tt& modulus)
{
	uint64_tt result_ = operand1 + operand2;
	csub_q(result_, modulus);
	return result_;
}

__forceinline__ __device__ void sub_uint128_uint64(const uint128_tt& operand1,
												   const uint64_tt& operand2,
												   uint128_tt& result)
{
        asm("{\n\t"
            "sub.cc.u64     %1, %3, %4;\n\t"
            "subc.u64    	%0, %2, 0;\n\t"
            "}"
            : "=l"(result.high), "=l"(result.low)
            : "l"(operand1.high), "l"(operand1.low), "l"(operand2));
}

__forceinline__ __device__ void add_uint128_uint128(const uint128_tt& operand1,
													const uint128_tt& operand2,
													uint128_tt& result)
{
	asm("{\n\t"
		"add.cc.u64     %0, %2, %4;\n\t"
		"addc.u64    	%1, %3, %5;\n\t"
		"}"
		: "=l"(result.low), "=l"(result.high)
		: "l"(operand1.low), "l"(operand1.high), "l"(operand2.low), "l"(operand2.high));
}

__forceinline__ __device__ void madc_uint64_uint64_uint128(const uint64_tt& operand1,
														   const uint64_tt& operand2,
														   uint128_tt& result)
{
	asm("{\n\t"
		"mad.lo.cc.u64		%0, %4, %5, %2;\n\t"
		"madc.hi.u64    	%1, %4, %5, %3;\n\t"
		"}"
		: "=l"(result.low), "=l"(result.high)
		: "l"(result.low), "l"(result.high), "l"(operand1), "l"(operand2));
}

#define max_tnum 12
#define max_Riblock_num 16
#define max_Qjblock_num 48
#define max_pqt_num 80
#define max_qr_num 80
#define max_p_num 10
#define max_gamma_num 10
#define max_q_num 20

// pqt_i in constant memory
__constant__ uint64_tt pqt_cons[max_pqt_num];
__constant__ uint64_tt pqt2_cons[max_pqt_num];
// pq_mu_i in constant memory
__constant__ uint64_tt pqt_mu_cons_high[max_pqt_num];
__constant__ uint64_tt pqt_mu_cons_low[max_pqt_num];
// T//2 mod pqt in constant memory
__constant__ uint64_tt halfTmodpqti_cons[max_pqt_num];

__constant__ uint64_tt Pmodqi_cons[max_pqt_num];
__constant__ uint64_tt Pinvmodqi_cons[max_pqt_num];
__constant__ uint64_tt Pinvmodqi_shoup_cons[max_pqt_num];
__constant__ uint64_tt pHatInvVecModp_cons[max_p_num];
__constant__ uint64_tt pHatInvVecModp_shoup_cons[max_p_num];
__constant__ uint64_tt Rimodti_cons[max_tnum * max_Riblock_num];
__constant__ uint64_tt Tmodpqi_cons[max_pqt_num];

__constant__ uint64_tt qr_cons[max_qr_num];
__constant__ uint64_tt qr_mu_cons_high[max_qr_num];
__constant__ uint64_tt qr_mu_cons_low[max_qr_num];