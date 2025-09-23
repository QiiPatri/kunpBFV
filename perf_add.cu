// BFV 加法性能测试
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

#include "include/Plaintext.cuh"
#include "include/Ciphertext.cuh"
#include "include/BFVScheme.cuh"
#include "include/BFVcontext.cuh"

int main(int argc, char* argv[]){
    size_t poly_modulus_degree = 32768;
    BFVContext context(poly_modulus_degree);
    BFVScheme scheme(context);
    SecretKey sk(context);
    scheme.mallocMemory(); scheme.addEncKey(sk); scheme.addMultKey_23(sk); scheme.addLeftRotKey_23(sk,1);

    int N = context.N; int L = context.L; int slots = context.slots;

    uint64_tt* mes1 = new uint64_tt[slots]; uint64_tt* mes2 = new uint64_tt[slots];
    for(size_t i=0;i<slots;++i){ mes1[i]=rand()%context.plain_modulus; mes2[i]=rand()%context.plain_modulus; }
    uint64_tt* d_msg1; uint64_tt* d_msg2; cudaMalloc(&d_msg1,sizeof(uint64_tt)*slots); cudaMalloc(&d_msg2,sizeof(uint64_tt)*slots);
    cudaMemcpy(d_msg1, mes1, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice); cudaMemcpy(d_msg2, mes2, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);

    Plaintext plain1(N,L,L,slots); Plaintext plain2(N,L,L,slots);
    Ciphertext c1(N,L,L,slots); Ciphertext c2(N,L,L,slots);

    const int round = 15; const int warmup = 5;
    float enc_time=0, add_time=0, cadd_time=0, dec_time=0, temp=0;
    cudaEvent_t start,end; cudaEventCreate(&start); cudaEventCreate(&end);

    for(int i=0;i<round;++i){
        cudaEventRecord(start); context.encode(d_msg1, plain1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) enc_time+=temp;
        context.encode(d_msg2, plain2);
        cudaEventRecord(start); scheme.encryptMsg(c1, plain1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) enc_time+=temp;
        scheme.encryptMsg(c2, plain2);

        cudaEventRecord(start); scheme.addAndEqual(c1, c2); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) add_time+=temp;

        cudaEventRecord(start); scheme.addConstAndEqual(c1, plain1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) cadd_time+=temp;

        Plaintext plain_dec(N,L,L,slots); cudaEventRecord(start); scheme.decryptMsg(plain_dec, sk, c1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) dec_time+=temp;
    }

    int iters = round - warmup;
    // printf("perf_add: enc(us): %f avg\n", enc_time/iters*1000);
    printf("BFV: 密文-密文加法(us): %f avg\n", add_time/iters*1000);
    printf("BFV: 密文-明文加法(us): %f avg\n", cadd_time/iters*1000);
    // printf("perf_add: dec(us): %f avg\n", dec_time/iters*1000);

    cudaFree(d_msg1); cudaFree(d_msg2); delete[] mes1; delete[] mes2;
    return 0;
}
