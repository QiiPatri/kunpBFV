#pragma once

#include "uint128.cuh"
#include "BFVcontext.h"
#include "RNG.cuh"
#include "Sampler.cuh"
#include "ntt_60bit.cuh"

// SecretKey
class SecretKey {
public:

	uint64_tt* sx_host;

	uint64_tt* sx_device;
    // Ring dim
	int N;
	int L, K;

	SecretKey(BFVContext& context, cudaStream_t stream = 0)
	{
		N = context.N;
		L = context.L;
		K = context.K;
		int logN = context.logN;
		sx_host = new uint64_tt[N * (K+L+1)];
        cudaMalloc(&sx_device, sizeof(uint64_tt) * N * (K+L+1));
		
		// print_device_array(context.randomArray_sk_device, N, "sk_device");

        Sampler::HWTSampler(context.randomArray_sk_device, sx_device, logN, context.h, 0, 0, K+L+1);

		// print_device_array(sx_device, N, "sx_device");
		// checkHWT(sx_device, N, K, L);
		//context.forwardNTT_batch(sx_device, 0, 0, 1, K+L+1);
		context.ToNTTInplace(sx_device, 0, 0, 1, K+L+1, K+L+1);

		// print_device_array(sx_device, N, K+L+1, "sx_device");
	}

    SecretKey operator = (SecretKey Secretkey)
    {
        if(this == &Secretkey) return *this;
        
        N = Secretkey.N;
        
        cudaMemcpy(this->sx_device, Secretkey.sx_device, sizeof(uint64_tt) * N, cudaMemcpyDeviceToDevice);

        return *this;
    }

	virtual ~SecretKey()
	{
		delete sx_host;

		cudaFree(sx_device);
	}

	void copyToHost()
	{
		cudaMemcpy(sx_host, sx_device, sizeof(uint64_tt) * N, cudaMemcpyDeviceToHost);
	}
};