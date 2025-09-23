// BFV 乘法性能测试
#include <iostream>
#include <string>
#include <vector>

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

    const int round = 50; const int warmup = 10;
    float hmult=0, cmult=0, dec=0, temp=0;
    cudaEvent_t start,end; cudaEventCreate(&start); cudaEventCreate(&end);

    for(int i=0;i<round;++i){
        context.encode(d_msg1, plain1); context.encode(d_msg2, plain2);
        scheme.encryptMsg(c1, plain1); scheme.encryptMsg(c2, plain2);

        cudaEventRecord(start); scheme.multAndEqual_23(c1, c2); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) hmult+=temp;
        cudaEventRecord(start); scheme.multConstAndEqual(c1, plain1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) cmult+=temp;

        Plaintext plain_dec(N,L,L,slots); cudaEventRecord(start); scheme.decryptMsg(plain_dec, sk, c1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) dec+=temp;
    }

    int iters = round - warmup;
    printf("BFV: 密文-密文乘法(us): %f avg\n", hmult/iters*1000);
    printf("BFV: 密文-明文乘法(us): %f avg\n", cmult/iters*1000);
    // printf("perf_mult: dec(us): %f avg\n", dec/iters*1000);

    cudaFree(d_msg1); cudaFree(d_msg2); delete[] mes1; delete[] mes2;
    return 0;
}
