// BFV 加法正确性测试（参考 bfv_demo.cu 与 ckks correct_add）
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "include/Plaintext.cuh"
#include "include/Ciphertext.cuh"
#include "include/BFVScheme.cuh"
#include "include/BFVcontext.cuh"
#include "include/Utils.cuh"

int main(int argc, char* argv()){
    size_t poly_modulus_degree = 32768;
    BFVContext context(poly_modulus_degree);
    printf("poly_modulus_degree : %lu\n", context.N);
    printf("plain_modulus : %llu\n", context.plain_modulus);

    BFVScheme scheme(context);
    SecretKey sk(context);
    scheme.mallocMemory();
    scheme.addEncKey(sk);
    scheme.addMultKey_23(sk);
    scheme.addLeftRotKey_23(sk, 1);

    int N = context.N;
    int L = context.L;
    int slots = context.slots;

    // host messages
    uint64_tt* mes1 = new uint64_tt[slots];
    uint64_tt* mes2 = new uint64_tt[slots];
    for(size_t i=0;i<slots;++i){ mes1[i] = rand()%context.plain_modulus; mes2[i] = rand()%context.plain_modulus; }

    // device buffers
    uint64_tt* d_msg1; uint64_tt* d_msg2; uint64_tt* d_dec;
    cudaMalloc(&d_msg1, sizeof(uint64_tt)*slots);
    cudaMalloc(&d_msg2, sizeof(uint64_tt)*slots);
    cudaMalloc(&d_dec, sizeof(uint64_tt)*slots);
    cudaMemcpy(d_msg1, mes1, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg2, mes2, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);

    Plaintext plain1(N, L, L, slots);
    Plaintext plain2(N, L, L, slots);
    Ciphertext c1(N, L, L, slots);
    Ciphertext c2(N, L, L, slots);

    // encode + encrypt
    cout << "进行编码和加密..." << endl;
    context.encode(d_msg1, plain1);
    context.encode(d_msg2, plain2);
    scheme.encryptMsg(c1, plain1);
    scheme.encryptMsg(c2, plain2);

    // homomorphic add
    cout << "进行密文-密文加法..." << endl;
    scheme.addAndEqual(c1, c2);
    // const add
    cout << "进行密文-明文加法..." << endl;
    scheme.addConstAndEqual(c1, plain1);

    // decrypt + decode
    cout << "进行解密和解码..." << endl;
    Plaintext plain_dec(N, L, L, slots);
    scheme.decryptMsg(plain_dec, sk, c1);
    context.decode(plain_dec, d_dec);

    // copy back and print first 20
    uint64_tt* host_dec = new uint64_tt[slots];
    cudaMemcpy(host_dec, d_dec, sizeof(uint64_tt)*slots, cudaMemcpyDeviceToHost);

    printf("预期结果: ");
    for(int i=0;i<8;++i) printf("%llu ", (uint64_tt)((mes1[i]+mes2[i] + mes1[i])%context.plain_modulus));
    printf("\n");

    printf("实际结果: ");
    for(int i=0;i<8;++i) printf("%llu ", host_dec[i]);
    printf("\n");

    // cleanup
    cudaFree(d_msg1); cudaFree(d_msg2); cudaFree(d_dec);
    delete[] mes1; delete[] mes2; delete[] host_dec;
    return 0;
}
