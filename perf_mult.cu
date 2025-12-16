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
    size_t poly_modulus_degree = argc > 1 ? 1 << atoi(argv[1]) : 32768;
    BFVContext context(poly_modulus_degree);
    BFVScheme scheme(context);
    SecretKey sk(context);
    scheme.mallocMemory(); scheme.addEncKey(sk); scheme.addMultKey_23(sk); scheme.addLeftRotKey_23(sk,1);

    int N = context.N; int L = context.L; int slots = context.slots;
    uint64_tt* mes1 = new uint64_tt[slots]; uint64_tt* mes2 = new uint64_tt[slots];
    for(size_t i=0;i<slots;++i){ mes1[i]=rand()%context.plain_modulus; mes2[i]=rand()%context.plain_modulus; }
    const size_t msgBytes = sizeof(uint64_tt) * slots;
    uint64_tt *d_msg1, *d_msg2, *d_msg1_init, *d_msg2_init;
    cudaMalloc(&d_msg1, msgBytes); cudaMalloc(&d_msg2, msgBytes);
    cudaMalloc(&d_msg1_init, msgBytes); cudaMalloc(&d_msg2_init, msgBytes);
    cudaMemcpy(d_msg1, mes1, msgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg2, mes2, msgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg1_init, mes1, msgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg2_init, mes2, msgBytes, cudaMemcpyHostToDevice);

    Plaintext plain1(N,L,L,slots); Plaintext plain2(N,L,L,slots); Plaintext plain1_ntt(N,L,L,slots);
    Ciphertext c1(N,L,L,slots); Ciphertext c2(N,L,L,slots);

    const int round_hmult = 3000;
    const int round_cmult = 450000;
    const int warmup = 0;
    double hmult = 0.0, cmult = 0.0;
    float temp_ms = 0.0f;
    cudaEvent_t start,end; cudaEventCreate(&start); cudaEventCreate(&end);
    cudaMemcpy(d_msg1, d_msg1_init, msgBytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_msg2, d_msg2_init, msgBytes, cudaMemcpyDeviceToDevice);
    context.encode(d_msg1, plain1); 
    context.encode(d_msg2, plain2); 
    for (int i = 0; i < round_hmult; ++i) {

        scheme.encryptMsg(c1, plain1); 
        scheme.encryptMsg(c2, plain2); 

        cudaEventRecord(start); 
        scheme.multAndEqual_23(c1, c2); 
        cudaEventRecord(end); 
        cudaEventSynchronize(end); 
        cudaEventElapsedTime(&temp_ms, start, end); 
        if(i >= warmup) hmult += static_cast<double>(temp_ms);

        // printf("轮次 %d/%d 完成,时间: %f ms\n", i+1, round_hmult, temp_ms);
        
        // Plaintext plain_dec(N,L,L,slots); cudaEventRecord(start); scheme.decryptMsg(plain_dec, sk, c1); cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&temp_ms,start,end);
    }
    cudaMemcpy(d_msg1, d_msg1_init, msgBytes, cudaMemcpyDeviceToDevice);
    context.encode(d_msg1, plain1);
    cudaMemcpy(d_msg1, d_msg1_init, msgBytes, cudaMemcpyDeviceToDevice);
    context.encode_ntt(d_msg1, plain1_ntt); 
    scheme.encryptMsg(c1, plain1);
    for (int i = 0; i < round_cmult; ++i) {
        cudaEventRecord(start);
            scheme.multConstAndEqual(c1, plain1_ntt);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp_ms, start, end);
        if(i >= warmup) cmult += static_cast<double>(temp_ms);
    }

    int iters_cmult = round_cmult - warmup;
    int iters_hmult = round_hmult - warmup;
    const double total_hmult_us = hmult * 1000.0;
    const double avg_hmult_us = iters_hmult > 0 ? (hmult / iters_hmult * 1000.0) : 0.0;
    printf("BFV: 密文-密文乘法总耗时(us): %f, 执行次数：%d, 平均耗时(us): %f\n", total_hmult_us, iters_hmult, avg_hmult_us);
    if (iters_cmult > 0) {
        const double total_cmult_us = cmult * 1000.0;
        const double avg_cmult_us = cmult / iters_cmult * 1000.0;
        printf("BFV: 密文-明文乘法总耗时(us): %f, 执行次数：%d, 平均耗时(us): %f\n", total_cmult_us, iters_cmult, avg_cmult_us);
    }
    cudaFree(d_msg1); cudaFree(d_msg2); cudaFree(d_msg1_init); cudaFree(d_msg2_init);
    delete[] mes1; delete[] mes2;
    return 0;
}
