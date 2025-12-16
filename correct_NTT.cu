// BFV 数据流图功能性测试
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
    uint64_tt* mes1 = new uint64_tt[slots]; uint64_tt* host_dec = new uint64_tt[slots];
    for(size_t i=0;i<slots;++i){ mes1[i]=rand()%context.plain_modulus; }

    uint64_tt *d_msg1, *d_dec; cudaMalloc(&d_msg1,sizeof(uint64_tt)*slots); cudaMalloc(&d_dec,sizeof(uint64_tt)*slots);
    cudaMemcpy(d_msg1, mes1, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice); cudaMemcpy(d_dec, mes1, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);
    
    //进行INTT
    puts("进行INTT...");
    context.FromNTTInplace_for_Test(d_dec ,0 ,0, 0, 0, 1);

    //进行NTT
    puts("进行NTT...");
    context.ToNTTInplace_for_Test(d_dec ,0 ,0, 0, 0, 1);

    cudaMemcpy(host_dec, d_dec, sizeof(uint64_tt)*slots, cudaMemcpyDeviceToHost);

    printf("预期结果: ");
    for(int i=0;i<8;++i) printf("%llu ", (uint64_tt)((mes1[i])%context.plain_modulus));
    printf("\n实际结果: ");
    for(int i=0;i<8;++i) printf("%llu ", host_dec[i]);
    printf("\n");

    cudaFree(d_msg1); cudaFree(d_dec); delete[] mes1; delete[] host_dec;
    return 0;
}
