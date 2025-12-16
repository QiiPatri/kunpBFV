// BFV 编解码/加解密性能测试
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#include "include/Plaintext.cuh"
#include "include/Ciphertext.cuh"
#include "include/BFVScheme.cuh"
#include "include/BFVcontext.cuh"

int main(int argc, char* argv[]){
    size_t poly_modulus_degree = argc > 1 ? 1 << atoi(argv[1]) : 32768;
    BFVContext context(poly_modulus_degree);
    BFVScheme scheme(context);
    SecretKey sk(context);
    scheme.mallocMemory(); scheme.addEncKey(sk);

    int N = context.N; int L = context.L; int slots = context.slots;
    uint64_tt* mes = new uint64_tt[slots]; for(size_t i=0;i<slots;++i) mes[i]=(i+7)%context.plain_modulus;
    uint64_tt* d_msg; cudaMalloc(&d_msg,sizeof(uint64_tt)*slots); cudaMemcpy(d_msg, mes, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);

    Plaintext plain(N,L,L,slots); Ciphertext c(N,L,L,slots);

    const int round = 80000; const int warmup = 0; float ecd=0, enc=0, dec=0, dcd=0, temp=0;
    cudaEvent_t start,end; cudaEventCreate(&start); cudaEventCreate(&end);

    for(int i=0;i<round;++i){
        cudaEventRecord(start); context.encode(d_msg, plain); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) ecd+=temp;
        cudaEventRecord(start); scheme.encryptMsg(c, plain); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) enc+=temp;

        Plaintext plain_dec(N,L,L,slots); cudaEventRecord(start); scheme.decryptMsg(plain_dec, sk, c); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) dec+=temp;
        uint64_tt* d_dec; cudaMalloc(&d_dec,sizeof(uint64_tt)*slots); cudaEventRecord(start); context.decode(plain_dec, d_dec); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp,start,end); if(i>=warmup) dcd+=temp; cudaFree(d_dec);
    }

    int iters = round - warmup;
    // printf("perf_encrypt: encode(us): %f avg\n", ecd/iters*1000);
    // printf("BFV: 加密(us): %f avg\n", (enc + ecd)/iters*1000);
    // printf("BFV: 解密(us): %f avg\n", (dec + dcd)/iters*1000);
    printf("BFV: 加密总耗时(us): %f, 执行次数：%d, 平均耗时(us): %f\n", (enc + ecd)*1000, iters, (enc + ecd)/iters*1000);
    printf("BFV: 解密总耗时(us): %f, 执行次数：%d, 平均耗时(us): %f\n", (dec + dcd)*1000, iters, (dec + dcd)/iters*1000);
    
    // printf("perf_encrypt: decode(us): %f avg\n", dcd/iters*1000);

    cudaFree(d_msg); delete[] mes; return 0;
}
