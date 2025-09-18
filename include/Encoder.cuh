#pragma once

#include "BFVcontext.h"
#include "cuda.h"

__global__ void encode_BFV_kernel(uint64_tt *out, uint64_tt *in, uint64_tt *index_map, uint64_t mod,
	size_t slots) {
	for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < slots; tid += blockDim.x * gridDim.x) {
	if (tid < slots) {
	const uint64_t temp = in[tid];
	out[index_map[tid]] = temp + (temp >> 63) * mod;
	} else
	out[index_map[tid]] = 0;
	}
}

__global__ void decode_BFV_kernel(uint64_tt *out, uint64_tt *in, uint64_tt *index_map, uint64_tt slots) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < slots; tid += blockDim.x * gridDim.x) {
        out[tid] = in[index_map[tid]];
    }
}

__host__ void BFVContext::encode(uint64_tt* vals, Plaintext& msg)
{
	long mes_slots = msg.slots;
	int level = msg.l;

	// print_device_array(vals, N, "vals");

	// encode_BFV_kernel <<<slots / 128, 128, 0, 0>>> (vals, vals, batch_encode_index_device, plain_modulus, slots);

	FromNTTInplace_for_BFV(vals ,0 ,0, 0, 0, 1);

	// print_device_array(vals, N, "vals");

	buffer_idx ++;
	if(buffer_idx == 2)
		cudaMemcpy(encode_buffer_device, vals, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToDevice);

	for( size_t i = 0 ; i < L + 1 ; ++i )
		cudaMemcpy(msg.mx_device + i * slots, vals, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToDevice);

    // uint64_tt gridDimGlb = N * q_num / blockDimGlb.x;
    // bfv_add_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb, 0>>>(
    //         msg.mx_device, msg.mx_device,
    //         negQl_mod_t,
    //         negQl_mod_t_shoup,
    //         tInv_mod_q,
    //         tInv_mod_q_shoup,
    //         qVec.data(),
    //         plain_modulus, N, q_num);

	// // print_device_array(msg.mx_device, N, L+1, "m * Q / t");

	// // buffer_idx ++;
	// // if(buffer_idx == 2)
	// // 	cudaMemcpy(encode_buffer_device, msg.mx_device, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToDevice);

	// //encode
	// ToNTTInplace(msg.mx_device, 0, K, 1, level+1, L+1);//NTT
	// // print_device_array(msg.mx_device, N, L+1, "encode");
}

__host__ void BFVContext::decode(Plaintext& msg, uint64_tt* vals)
{
	int l = msg.l;
	int L = msg.L;

	//decode
	// cudaMemcpy(decode_buffer, msg.mx_device, sizeof(uint64_tt) * N * (L+1), cudaMemcpyDeviceToDevice);

	// FromNTTInplace(decode_buffer, 0, K, 1, l+1, L+1);

	// // print_device_array(decode_buffer, N , L+1 , "decode_buffer");

	// // buffer_idx--;
	// // if(buffer_idx == 2)
	// // 	cudaMemcpy(decode_buffer_device, decode_buffer, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToDevice);

	// hps_decrypt_scale_and_round(decode_buffer, decode_buffer, 0);

	buffer_idx--;
	if(buffer_idx == 2)
		cudaMemcpy(decode_buffer_device, decode_buffer, sizeof(uint64_tt) * slots, cudaMemcpyDeviceToDevice);

	// print_device_array(decode_buffer, N , L+1 , "decode_buffer");

	cudaMemcpy(vals, msg.mx_device, sizeof(uint64_tt) * N, cudaMemcpyDeviceToDevice);
	// cudaMemcpy(vals, decode_buffer, sizeof(uint64_tt) * N, cudaMemcpyDeviceToDevice);

	// ToNTTInplace(decode_buffer, 0, K, 1, l+1, L+1);
	ToNTTInplace_for_BFV(vals ,0 ,0, 0, 0, 1);
	// print_device_array(vals, N , "decode");

	// decode_BFV_kernel<<<slots / 128, 128, 0, 0>>>(vals, vals, batch_encode_index_device, slots);
}