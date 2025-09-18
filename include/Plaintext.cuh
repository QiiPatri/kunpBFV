#pragma once
#include "uint128.cuh"
#include "Utils.cuh"


class Plaintext {
public:
    Plaintext(){mx_device = nullptr;}
    Plaintext(int N, int L, int l, int slots) : N(N), L(L), l(l), slots(slots)
    {
        cudaMalloc(&mx_device, sizeof(uint64_tt) * N * (L + 1));
    }

    Plaintext(uint64_tt *m_host, int N, int L,int l, int slots) : N(N), L(L), l(l), slots(slots)
    {
        cudaMalloc(&mx_device, sizeof(uint64_tt) * N * (L + 1));

        cudaMemcpy(mx_device, m_host, sizeof(uint64_tt) * N * (L + 1), cudaMemcpyHostToDevice);
    }

    Plaintext(const Plaintext& c) : N(c.N), L(c.L), l(c.l), slots(c.slots)
    {
        if(this->mx_device == nullptr)
        {
            cudaMalloc(&(this->mx_device), sizeof(uint64_tt) * N * (L + 1));
            this->mx_device = this->mx_device;
        }
        if(c.mx_device != nullptr)
        {
            cudaMemcpy(this->mx_device, c.mx_device, sizeof(uint64_tt) * N * (L + 1), cudaMemcpyDeviceToDevice);
        }
    }

    Plaintext& operator = (const Plaintext& m)
    {
        if(this == &m) return *this;
        
        N = m.N;
        L = m.L;
        l = m.l;
        slots = m.slots;

        if(this->mx_device == nullptr)
        {
            cudaMalloc(&mx_device, sizeof(uint64_tt) * N * (L + 1));
        }
        cudaMemcpy(mx_device, m.mx_device, sizeof(uint64_tt) * N * (L + 1), cudaMemcpyDeviceToDevice);
        return *this;
    }

    virtual ~Plaintext()
    {
        if(mx_device != nullptr)
        {
            cudaFree(mx_device);
            mx_device = nullptr;
        }
    }

    uint64_tt *mx_device = nullptr;
    // Ring dim
    int N;
    // mul level
    int l;
    // to malloc memory
    int L;
    // slots
    int slots;
};

class PlaintextT {
public:
    PlaintextT(){mx_device = nullptr;}
    PlaintextT(int N, int t_num, int slots) : N(N), slots(slots)
    {
        cudaMalloc(&mx_device, sizeof(uint64_tt) * N * t_num);
    }

    PlaintextT(uint64_tt *m_host, int N, int t_num, int slots) : N(N), slots(slots)
    {
        cudaMalloc(&mx_device, sizeof(uint64_tt) * N * t_num);

        cudaMemcpy(mx_device, m_host, sizeof(uint64_tt) * N * t_num, cudaMemcpyHostToDevice);
    }

    PlaintextT(const PlaintextT& c) : N(c.N), t_num(c.t_num), slots(c.slots)
    {
        if(this->mx_device == nullptr)
        {
            cudaMalloc(&(this->mx_device), sizeof(uint64_tt) * N * t_num);
            this->mx_device = this->mx_device;
        }
        if(c.mx_device != nullptr)
        {
            cudaMemcpy(this->mx_device, c.mx_device, sizeof(uint64_tt) * N * t_num, cudaMemcpyDeviceToDevice);
        }
    }

    PlaintextT& operator = (const PlaintextT& m)
    {
        if(this == &m) return *this;
        
        N = m.N;
        t_num = m.t_num;
        slots = m.slots;

        if(this->mx_device == nullptr)
        {
            cudaMalloc(&mx_device, sizeof(uint64_tt) * N * t_num);
        }
        cudaMemcpy(mx_device, m.mx_device, sizeof(uint64_tt) * N * t_num, cudaMemcpyDeviceToDevice);
        return *this;
    }

    virtual ~PlaintextT()
    {
        if(mx_device != nullptr)
        {
            cudaFree(mx_device);
            mx_device = nullptr;
        }
    }

    uint64_tt *mx_device = nullptr;
    // Ring dim
    int N;
    // to malloc memory
    int t_num;
    // slots
    int slots;
};