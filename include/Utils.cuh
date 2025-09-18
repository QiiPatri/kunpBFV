#pragma once

#include "uint128.cuh"
#include "cuComplex.h"

#include <algorithm>
#include <stdlib.h>
#include <random>
#include <set>
#include <iostream>

using namespace std;

uint64_tt modpow128(uint64_tt a, uint64_tt b, uint64_tt mod)
{
    uint64_tt res = 1;
    while(b)
    {
        if(b & 1)
        {
            __uint128_t r128 = res;
            r128 *= a;
            res = uint64_tt(r128 % mod);
        }
        __uint128_t t128 = a;
        t128 *= a;
        a = uint64_tt(t128 % mod);
        b >>= 1;
    }
    return res;
}

uint64_tt mulMod128(uint64_tt a, uint64_tt b, uint64_tt q)
{
    __uint128_t c = a;
    c *= b;
    return uint64_tt(c % q);
}

uint64_tt x_Shoup(uint64_tt x, uint64_tt q)
{
    __uint128_t temp = x;
    temp <<= 64;
    return uint64_tt(temp / q);
}

//return a^-1 mod q
uint64_tt modinv128(uint64_tt a, uint64_tt q)
{
    a = a % q;
    uint64_tt ainv = modpow128(a, q - 2, q);
    return ainv;
}

void findPrimeFactors(set<uint64_tt> &s, uint64_tt number) {
	while (number % 2 == 0) {
		s.insert(2);
		number /= 2;
	}
	for (uint64_t i = 3; i < sqrt(number); i++) {
		while (number % i == 0) {
			s.insert(i);
			number /= i;
		}
	}
	if (number > 2) {
		s.insert(number);
	}
}

uint64_tt findPrimitiveRoot(uint64_tt modulus) {
	set<uint64_tt> s;
	uint64_tt phi = modulus - 1;
	findPrimeFactors(s, phi);
	for (uint64_tt r = 2; r <= phi; r++) {
		bool flag = false;
		for (auto it = s.begin(); it != s.end(); it++) {
			if (modpow128(r, phi / (*it), modulus) == 1) {
				flag = true;
				break;
			}
		}
		if (flag == false) {
			return r;
		}
	}
	return -1;
}

uint64_tt findMthRootOfUnity(uint64_tt M, uint64_tt mod) {
    uint64_tt res;
    res = findPrimitiveRoot(mod);
    if((mod - 1) % M == 0) {
        uint64_tt factor = (mod - 1) / M;
        res = modpow128(res, factor, mod);
        return res;
    }
    else {
        return -1;
    }
}

__host__ __device__ uint64_tt bitReverse(uint64_tt a, int bit_length)
{
    uint64_tt res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    return res;
}

void Complex_bitReverse(cuDoubleComplex* vals, const long slots) 
{
	for (long i = 1, j = 0; i < slots; ++i) {
		long bit = slots >> 1;
		for (; j >= bit; bit>>=1) {
			j -= bit;
		}
		j += bit;
		if(i < j) {
            cuDoubleComplex temp = vals[i];
            vals[i] = vals[j];
            vals[j] = temp;
		}
	}
}

std::random_device dev;
std::mt19937_64 rng(dev());

void randomComplexArray(cuDoubleComplex* ComplexArray, long slots, double bound = 1.0)
{
    std::uniform_int_distribution<int> randnum(0, RAND_MAX);

	for (long i = 0; i < slots; ++i) {
		// ComplexArray[i].x = ((double) rand()/(RAND_MAX) - 0.5) * 2 * bound;
        // ComplexArray[i].y = ((double) rand()/(RAND_MAX) - 0.5) * 2 * bound;
        // ComplexArray[i].x = (double) randnum(rng)/(RAND_MAX) * bound;
        // ComplexArray[i].y = (double) randnum(rng)/(RAND_MAX) * bound;

		ComplexArray[i].x = (double) rand()/(RAND_MAX) * bound;
        // ComplexArray[i].y = (double) rand()/(RAND_MAX) * bound;
	}
}

void fillTablePsi128(uint64_tt psi, uint64_tt q, uint64_tt psiinv, uint64_tt psiTable[], uint64_tt psiinvTable[], uint32_tt n)
{
    for (int i = 0; i < n; i++)
    {
        psiTable[i] = modpow128(psi, bitReverse(i, log2(n)), q);
        psiinvTable[i] = modpow128(psiinv, bitReverse(i, log2(n)), q);
    }
}
// void fillTablePsi128(uint64_tt psi, uint64_tt q, uint64_tt psiinv, uint64_tt psiTable[], uint64_tt psiinvTable[], uint32_tt n)
// {
//     psiTable[0] = psiinvTable[0] = 1;
//     for (int i = 1; i < n; i++)
//     {
//         int idx_prev = bitReverse(i-1, log2(n));
//         int idx_next = bitReverse(i, log2(n));
//         psiTable[idx_next] = mulMod128(psi, psiTable[idx_prev], q);
//         psiinvTable[idx_next] = mulMod128(psiinv, psiinvTable[idx_prev], q);
//     }
// }

void fillTablePsi128_special(uint64_tt psi, uint64_tt q, uint64_tt psiinv, uint64_tt psiTable[], uint64_tt psiinvTable[], uint32_tt n, uint64_tt inv_degree)
{
    psiTable[0] = psiinvTable[0] = 1;
    for (int i = 1; i < n; i++)
    {
        int idx_prev = bitReverse(i-1, log2(n));
        int idx_next = bitReverse(i, log2(n));
        psiTable[idx_next] = mulMod128(psi, psiTable[idx_prev], q);
        psiinvTable[idx_next] = mulMod128(psiinv, psiinvTable[idx_prev], q);
    }
    psiinvTable[1] = mulMod128(psiinvTable[1], inv_degree, q);
}

void fillTablePsi_shoup128(uint64_tt psiTable[], uint64_tt q, uint64_tt psiinv_Table[], uint64_tt psi_shoup_table[], uint64_tt psiinv_shoup_table[], uint32_tt n)
{
    for (int i = 0; i < n; i++)
    {
        psi_shoup_table[i] = x_Shoup(psiTable[i], q);
        psiinv_shoup_table[i] = x_Shoup(psiinv_Table[i], q);
    }
}

void bitReverseArray(uint64_tt array[], uint32_tt n)
{
    uint64_tt* temp = (uint64_tt*)malloc(sizeof(uint64_tt) * n);
    uint32_tt log2n = log2(n);
    for(int i = 0; i < n; i++)
    {
        temp[i] = array[bitReverse(i, log2n)];
    }
    memcpy(array, temp, sizeof(uint64_tt) * n);
    free(temp);
}

void randomArray128(uint64_tt a[], int n, uint64_tt q)
{
    std::uniform_int_distribution<uint64_tt> randnum(0, q - 1);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void randomArray64(uint32_tt a[], int n, uint32_tt q)
{
    std::uniform_int_distribution<uint32_tt> randnum(0, q);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void randomArray8(uint8_tt a[], int n, uint8_tt q)
{
    std::uniform_int_distribution<uint8_tt> randnum(0, q);

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

//poly a * poly b on  Zm[x]/(x^n+1)
uint64_tt* refPolyMul128(uint64_tt a[], uint64_tt b[], uint64_tt m, int n)
{
    uint64_tt* c = (uint64_tt*)malloc(sizeof(uint64_tt) * n * 2);
    uint64_tt* d = (uint64_tt*)malloc(sizeof(uint64_tt) * n);

    for (int i = 0; i < (n * 2); i++)
    {
        c[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i + j] = (__uint128_t(a[i]) * b[j] + c[i + j]) % m;
        }
    }

    for (int i = 0; i < n; i++)
    {

        if (c[i] < c[i + n])
            c[i] += m;

        d[i] = (c[i] - c[i + n]) % m;
    }

    free(c);

    return d;
}

uint32_tt* refPolyMul64(uint32_tt a[], uint32_tt b[], uint32_tt m, int n)
{
    uint32_tt* c = (uint32_tt*)malloc(sizeof(uint32_tt) * n * 2);
    uint32_tt* d = (uint32_tt*)malloc(sizeof(uint32_tt) * n);

    for (int i = 0; i < (n * 2); i++)
    {
        c[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i + j] = ((uint64_tt)a[i] * b[j]) % m + c[i + j] % m;
            c[i + j] %= m;
        }
    }

    for (int i = 0; i < n; i++)
    {

        if (c[i] < c[i + n])
            c[i] += m;

        d[i] = (c[i] - c[i + n]) % m;
    }

    free(c);

    return d;
}


__global__ void print(uint64_tt * data, int N)
{
    register int k = blockIdx.x * 256 + threadIdx.x;
    if(k == 0)
    {
        for(int i = 0; i < N; i++) printf("%llu\n", data[i]);
    }
}

void print_device_array(uint64_tt* data, int N, int mod_num, const char* vname)
{
    uint64_tt* array_PQ = new uint64_tt[N * mod_num];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * mod_num, cudaMemcpyDeviceToHost);
    printf("%s = [\n", vname);
    int start = 0;
    for(int i = 0; i < mod_num; i++)
    {
        printf("[");
        for(int t = start; t < start+8; t++)
        {
            printf("%llu, ", array_PQ[i*N + t]);
            // cout << fixed << array_PQ[i*N + t] << " , ";
        }
        printf("],\n");
    }
    printf("]\n");
    free(array_PQ);
}

void print_device_array(uint64_tt* data, int slots, const char* vname)
{
    uint64_tt* array_PQ = new uint64_tt[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    // cout.precision(8);
    int end = min(start + 6, slots);
    for(int i = start; i < end; i++)
    {
        printf("%llu, ", array_PQ[i]);
        // cout << fixed << array_PQ[i] << " , ";
    }
    printf("]\n");
    free(array_PQ);
}

void print_device_array_max(uint64_tt* data, int slots, const char* vname, int max_size)
{
    uint64_tt* array_PQ = new uint64_tt[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    int end = min(max_size, slots);
    for(int i = start; i < end; i++)
    {
        printf("%05llu, ", array_PQ[i]);
    }
    printf("]\n");
    free(array_PQ);
}
void print_check_mul(uint64_tt* mes1, uint64_tt* mes2, uint64_tt* data, int slots , uint64_tt mod)
{
    uint64_tt* array_PQ = new uint64_tt[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToHost);
    for(int i = 0; i < slots; i++)
    {
        if( ( (mes1[i] * mes2[i]) % mod ) != array_PQ[i] ) 
        {
            printf("error in %d\n" , i);
            return;
        }
    }
    free(array_PQ);
}
void print_check_equal(size_t &first_error_idx, size_t &error_idx_num, uint64_tt* data, uint64_tt* data2, int slots)
{
    uint64_tt* array_PQ = new uint64_tt[slots];
    uint64_tt* array_PQ2 = new uint64_tt[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToHost);
    for(int i = 0; i < slots; i++)
    {
        if( array_PQ[i] != array_PQ2[i] ) 
        {
            if( first_error_idx == 0 )
            {
                first_error_idx = i;
            }
            error_idx_num++;

            // printf("error in %d\n" , i);
            // return;
        }
    }
    free(array_PQ);
    free(array_PQ2);
}

void print_device_array_from_symbol(uint64_tt* data, int slots, const char* vname)
{
    uint64_tt* array_PQ = new uint64_tt[slots];
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(array_PQ, data, sizeof(uint64_tt) * slots, 0, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    // cout.precision(8);
    int end = min(start + 6, slots);
    for(int i = start; i < end; i++)
    {
        printf("%llu, ", array_PQ[i]);
        // cout << fixed << array_PQ[i] << " , ";
    }
    printf("]\n");
    free(array_PQ);
}

void print_host_array(uint64_tt* data, int slots, const char* vname)
{
    printf("%s = [", vname);
    int start = 0;
    int end = min(start + 6, slots);
    for(int i = start; i < end; i++)
    {
        printf("%llu, ", data[i]);
    }
    printf("]\n");
}
void print_host__array(uint64_tt* data, uint64_tt* data2, int slots)
{
    printf("mes1 = [ ");
    for( size_t idx = 0 ; idx <= 10 ; ++idx )
    {
        printf( "%lld " , data[idx] );
    }
    printf(" ]\n");
    printf("mes2 = [ ");
    for( size_t idx = 0 ; idx <= 10 ; ++idx )
    {
        printf( "%lld " , data2[idx] );
    }
    printf(" ]\n");
}

void print_device_array(uint8_tt* data, int slots, const char* vname)
{
    uint8_tt* array_PQ = new uint8_tt[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint8_tt) * slots, cudaMemcpyDeviceToHost);
    printf("%s = [", vname);
    int start = 0;
    // cout.precision(8);
    for(int i = start; i < start+11; i++)
    {
        // printf("%lf + i*%lf, ", array_PQ[i].x, array_PQ[i].y);
        printf( "%hhu, " , array_PQ[i] );
    }
    printf("]\n");
    free(array_PQ);
}

int compare_device_array(cuDoubleComplex* data1, cuDoubleComplex* data2, int slots, const char* vname)
{
    cuDoubleComplex* array_PQ1 = new cuDoubleComplex[slots];
    cuDoubleComplex* array_PQ2 = new cuDoubleComplex[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    int cnt = 0;
    for(int i = 0; i < slots; i++)
    {
        if(abs(array_PQ1[i].x - array_PQ2[i].x) < 1e-3 && abs(array_PQ1[i].y - array_PQ2[i].y) < 1e-3) continue;
        // printf("%lf+%lf vs %lf+%lf\n", array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        // printf("%d, %lf+%lfi vs %lf+%lfi\n", i, array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        cnt++;
    }
    // printf("%s error num: %d\n", vname, cnt);
    delete array_PQ1;
    delete array_PQ2;

    return cnt;
}

void compare_device_array(cuDoubleComplex* data1, cuDoubleComplex* data2, cuDoubleComplex* data3, int slots, const char* vname)
{
    cuDoubleComplex* array_PQ1 = new cuDoubleComplex[slots];
    cuDoubleComplex* array_PQ2 = new cuDoubleComplex[slots];
    cuDoubleComplex* array_PQ3 = new cuDoubleComplex[slots];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ3, data3, sizeof(cuDoubleComplex) * slots, cudaMemcpyDeviceToHost);
    int cnt = 0;
    for(int i = 0; i < slots; i++)
    {
        array_PQ3[i] = cuCmul(array_PQ2[i], array_PQ3[i]);
        if(abs(array_PQ1[i].x - array_PQ3[i].x) < 1e-5 && abs(array_PQ1[i].y - array_PQ3[i].y) < 1e-5) continue;
        // printf("%lf+%lf vs %lf+%lf\n", array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        // printf("%d, %lf+%lfi vs %lf+%lfi\n", i, array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
        cnt++;
    }
    printf("%s error num: %d\n", vname, cnt);
    delete array_PQ1;
    delete array_PQ2;
    delete array_PQ3;
}

void compare_device_array(uint64_tt* data1, uint64_tt* data2, int N, int len, string vname)
{
    uint64_tt* array_PQ1 = new uint64_tt[len*N];
    uint64_tt* array_PQ2 = new uint64_tt[len*N];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ1, data1, sizeof(uint64_tt) * len*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PQ2, data2, sizeof(uint64_tt) * len*N, cudaMemcpyDeviceToHost);
    printf("%s error num: ", vname.c_str());
    int cnt = 0;
    for(int idx = 0; idx < len; idx++)
    {
        cnt = 0;
        for(int i = 0; i < N; i++)
        {
            if(array_PQ1[idx*N + i] == array_PQ2[idx*N + i]) continue;
            // else printf("%d %d %llu %llu\n", idx, i, array_PQ1[idx*N + i], array_PQ2[idx*N + i]);
            // printf("%lf+%lf vs %lf+%lf\n", array_PQ1[i].x, array_PQ1[i].y, array_PQ2[i].x, array_PQ2[i].y);
            // printf("%d, %llu vs %llu\n", i, array_PQ1[i], array_PQ2[i]);
            cnt++;
        }
        printf("(%d,%d), ", idx, cnt);
    }
    cout<<endl;
    delete array_PQ1;
    delete array_PQ2;
}

void print_device_array(uint128_tt* data, int N, int K, int L, string vname)
{
    uint128_tt* array_PQ = new uint128_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint128_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);
    printf("%s = [", vname.c_str());
    for(int i = 0; i < (K+L+1); i++)
    {
        printf("[");
        for(int t = 0; t < 4; t++)
        {
            printf("(%llu,%llu), ", array_PQ[i*N + t].high, array_PQ[i*N + t].low);
        }
        printf("],");
    }
    printf("]\n");
    delete array_PQ;
}

void checkHWT(uint64_tt* data, int N, int K, int L)
{
    uint64_tt* array_PQ = new uint64_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);
    
    printf("array_PQ = [\n");
    for(int t = 0; t < N; t++)
    {
        if(array_PQ[t] != 0)
        printf("%llu %d\n", array_PQ[t], t);
    }
    printf("]\n");
}

void count_Zero(uint64_tt* data, int N, int K, int L)
{
    uint64_tt* array_PQ = new uint64_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);

    printf("array_PQ non zero count = ");
    for(int i = 0; i < K+L+1; i++)
    {
        int cnt = 0;
        for(int t = 0; t < N; t++)
        {
            if(array_PQ[i*N + t] != 0) cnt++;
        }
        printf("%d ", cnt);
    }
    printf("\n");
}

void count_ZO(uint64_tt* data, int N, int K, int L)
{
    uint64_tt* array_PQ = new uint64_tt[N * (K+L+1)];
    cudaDeviceSynchronize();
    cudaMemcpy(array_PQ, data, sizeof(uint64_tt) * N * (K+L+1), cudaMemcpyDeviceToHost);

    printf("array_PQ 0 count = ");
    for(int i = 0; i < K+L+1; i++)
    {
        int cnt = 0;
        for(int t = 0; t < N; t++)
        {
            if(array_PQ[i*N + t] == 0) cnt++;
        }
        printf("%d ", cnt);
    }
    printf("\n");

    printf("array_PQ +-1 count = ");
    for(int i = 0; i < K+L+1; i++)
    {
        int cnt = 0;
        for(int t = 0; t < N; t++)
        {
            if(array_PQ[i*N + t] != 0) cnt++;
        }
        printf("%d ", cnt);
    }
    printf("\n");
}

uint64_tt reverse_bits(uint64_tt operand, int bit_count) {
    // 帮助函数：翻转32位整数中的所有位
    auto reverse_bits_u32 = [](uint32_t value) -> uint32_t {
        value = (((value & 0xaaaaaaaa) >> 1) | ((value & 0x55555555) << 1));
        value = (((value & 0xcccccccc) >> 2) | ((value & 0x33333333) << 2));
        value = (((value & 0xf0f0f0f0) >> 4) | ((value & 0x0f0f0f0f) << 4));
        value = (((value & 0xff00ff00) >> 8) | ((value & 0x00ff00ff) << 8));
        return (value >> 16) | (value << 16);
    };
    
    // 翻转64位整数中的所有位
    auto reverse_all_bits = [&reverse_bits_u32](uint64_tt value) -> uint64_tt {
        return static_cast<uint64_tt>(reverse_bits_u32(static_cast<uint32_t>(value >> 32))) |
               (static_cast<uint64_tt>(reverse_bits_u32(static_cast<uint32_t>(value & 0xFFFFFFFF))) << 32);
    };
    
    // 如果bit_count为0，直接返回0
    if (bit_count == 0) {
        return 0;
    }
    
    // 先翻转所有位，然后右移以保留需要的位数
    return reverse_all_bits(operand) >> (64 - static_cast<size_t>(bit_count));
}