#pragma once
#include "uint128.cuh"
#include "Utils.cuh"

class Ciphertext {
public:
    uint64_tt *ax_device = nullptr;
    uint64_tt *bx_device = nullptr;
    uint64_tt *cipher_device = nullptr;

    // uint64_tt* ax_host = nullptr;
    // uint64_tt* bx_host = nullptr;
    // uint64_tt* cipher_host = nullptr;

    // Ring dim
    int N;
    // mul level
    int l;
    // to malloc memory
    int L;
    // slots
    int slots;

    Ciphertext(){ax_device = nullptr; bx_device = nullptr;}
    Ciphertext(int N, int L, int l, int slots) : N(N), L(L), l(l), slots(slots)
    {
        cudaMallocManaged(&cipher_device, sizeof(uint64_tt) * N * (L + 1) * 2);
        ax_device = cipher_device;
        bx_device = cipher_device + N * (L + 1);
    }

    Ciphertext(uint64_tt *a, uint64_tt *b, int N, int l, int slots) : N(N), L(L), l(l), slots(slots)
    {
        cudaMallocManaged(&cipher_device, sizeof(uint64_tt) * N * (L + 1) * 2);
        ax_device = cipher_device;
        bx_device = cipher_device + N * (L + 1);

        cudaMemcpy(ax_device, a, sizeof(uint64_tt) * N * (L + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(bx_device, b, sizeof(uint64_tt) * N * (L + 1), cudaMemcpyHostToDevice);
    }

    Ciphertext(const Ciphertext& c) : N(c.N), L(c.L), l(c.l), slots(c.slots)
    {
        if(this->cipher_device == nullptr)
        {
            cudaMallocManaged(&(this->cipher_device), sizeof(uint64_tt) * N * (L + 1) * 2);
        }
        if(c.cipher_device != nullptr)
        {
            cudaMemcpy(this->cipher_device, c.cipher_device, sizeof(uint64_tt) * N * (L + 1) * 2, cudaMemcpyDeviceToDevice);
        }
        this->ax_device = this->cipher_device;
        this->bx_device = this->cipher_device + N * (L + 1);
    }

    Ciphertext& operator = (Ciphertext& c)
    {
        if(this == &c) return *this;
        
        this->N = c.N;
        this->L = c.L;
        this->l = c.l;
        this->slots = c.slots;

        if(this->cipher_device == nullptr)
        {
            cudaMallocManaged(&cipher_device, sizeof(uint64_tt) * N * (L + 1) * 2);
        }

        if(c.cipher_device != nullptr)
        {
            cudaMemcpy(this->cipher_device, c.cipher_device, sizeof(uint64_tt) * N * (L + 1) * 2, cudaMemcpyDeviceToDevice);
        }
        this->ax_device = this->cipher_device;
        this->bx_device = this->cipher_device + N * (L + 1);
        return *this;
    }

    virtual ~Ciphertext()
    {
        if(cipher_device != nullptr)
        {
            cudaFree(cipher_device);
            cipher_device = nullptr;
        }
        ax_device = nullptr;
        bx_device = nullptr;
    }
};
 