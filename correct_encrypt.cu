// BFV 编解码与加解密正确性测试
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
    scheme.mallocMemory(); scheme.addEncKey(sk);

    int N = context.N; int L = context.L; int slots = context.slots;
    uint64_tt* mes = new uint64_tt[slots];
    for(size_t i=0;i<slots;++i) mes[i] = rand()%context.plain_modulus;

    uint64_tt* d_msg; cudaMalloc(&d_msg,sizeof(uint64_tt)*slots); cudaMemcpy(d_msg, mes, sizeof(uint64_tt)*slots, cudaMemcpyHostToDevice);

    Plaintext plain(N,L,L,slots); Ciphertext c(N,L,L,slots);
    cout << "进行编码和加密..." << endl;
    context.encode(d_msg, plain);
    scheme.encryptMsg(c, plain);

    Plaintext plain_dec(N,L,L,slots); 
    cout << "进行解密和解码..." << endl;

    scheme.decryptMsg(plain_dec, sk, c);
    uint64_tt* d_dec; cudaMalloc(&d_dec,sizeof(uint64_tt)*slots); context.decode(plain_dec, d_dec);

    uint64_tt* host_dec = new uint64_tt[slots]; cudaMemcpy(host_dec, d_dec, sizeof(uint64_tt)*slots, cudaMemcpyDeviceToHost);

    printf("预期结果: "); for(int i=0;i<8;++i) printf("%llu ", mes[i]); printf("\n");
    printf("实际结果: "); for(int i=0;i<8;++i) printf("%llu ", host_dec[i]); printf("\n");

    cudaFree(d_msg); cudaFree(d_dec); delete[] mes; delete[] host_dec;
    return 0;
}
