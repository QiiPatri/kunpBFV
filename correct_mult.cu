// BFV 乘法正确性测试
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

    uint64_tt *d_msg1, *d_msg2, *d_dec; cudaMalloc(&d_msg1,sizeof(uint64_tt)*slots); cudaMalloc(&d_msg2,sizeof(uint64_tt)*slots); cudaMalloc(&d_dec,sizeof(uint64_tt)*slots);
    cudaMemcpy(d_msg1, mes1, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice); cudaMemcpy(d_msg2, mes2, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);

    Plaintext plain1(N,L,L,slots); Plaintext plain2(N,L,L,slots); Plaintext plain1_ntt(N,L,L,slots);
    Ciphertext c1(N,L,L,slots); Ciphertext c2(N,L,L,slots);
    cout << "进行编码和加密..." << endl;
    context.encode(d_msg1, plain1); context.encode(d_msg2, plain2); context.encode_ntt(d_msg1, plain1_ntt);
    scheme.encryptMsg(c1, plain1); scheme.encryptMsg(c2, plain2);

    // homomorphic multiply
    cout << "进行密文-密文乘法..." << endl;
    scheme.multAndEqual_23(c1, c2);
    // multiply by constant
    cout << "进行密文-明文乘法..." << endl;
    scheme.multConstAndEqual(c1, plain1_ntt);

    Plaintext plain_dec(N,L,L,slots);
    cout << "进行解密和解码..." << endl;
    scheme.decryptMsg(plain_dec, sk, c1);
    context.decode(plain_dec, d_dec);

    uint64_tt* host_dec = new uint64_tt[slots]; cudaMemcpy(host_dec, d_dec, sizeof(uint64_tt)*slots, cudaMemcpyDeviceToHost);

    printf("预期结果: ");
    for(int i=0;i<8;++i) printf("%llu ", (uint64_tt)((mes1[i]*mes2[i]*mes1[i])%context.plain_modulus));
    printf("\n实际结果: ");
    for(int i=0;i<8;++i) printf("%llu ", host_dec[i]);
    printf("\n");

    cudaFree(d_msg1); cudaFree(d_msg2); cudaFree(d_dec); delete[] mes1; delete[] mes2; delete[] host_dec;
    return 0;
}
