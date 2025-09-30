#pragma once

#include "BFVScheme.h"
#include "KeySwitch_23.cuh"
#include "poly_arithmetic.cuh"

void BFVScheme::mallocMemory()
{
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    // int gamma = context.gamma;
    int Ri_blockNum = context.Ri_blockNum;
    int Qj_blockNum = context.Qj_blockNum;
    // printf("Ri_blockNum: %d Qj_blockNum: %d\n", Ri_blockNum, Qj_blockNum);

    cudaMalloc(&ex_swk, sizeof(uint64_tt) * N * (K+L+1));
    cudaMalloc(&sxsx, sizeof(uint64_tt) * N * (K+L+1));
    cudaMalloc(&sx_coeff, sizeof(uint64_tt) * N * (K+L+1));

    cudaMalloc(&axbx1_mul, sizeof(uint64_tt) * N * (L+1));
    cudaMalloc(&axbx2_mul, sizeof(uint64_tt) * N * (L+1));
    cudaMalloc(&axax_mul, sizeof(uint64_tt) * N * (L+1));
    cudaMalloc(&bxbx_mul, sizeof(uint64_tt) * N * (L+1));

    cudaMalloc(&temp_mul, sizeof(uint64_tt) * N * (K+L+1));

    cudaMalloc(&vx_enc, sizeof(uint64_tt) * N * (K+L+1));
    cudaMalloc(&ex_enc, sizeof(uint64_tt) * N * (K+L+1));

    cudaMalloc(&modUp_RitoT_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * context.dnum);
    cudaMalloc(&modUp_QjtoT_temp, sizeof(uint64_tt) * N * t_num * Qj_blockNum);
    cudaMalloc(&exProduct_T_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);
	cudaMalloc(&modUp_TtoQj_buffer, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);

    cudaMalloc(&rescale_buffer, sizeof(uint64_tt) * N * (L+1) * 2);

    random_len_for_enc = N * sizeof(uint8_tt) + N * sizeof(uint32_tt) + N * sizeof(uint32_tt);
    cudaMalloc(&in_enc, random_len_for_enc);
    
    rotKey_vec_23 = vector<Key_decomp*>(N, nullptr);
    autoKey_vec_23 = vector<Key_decomp*>(context.logN, nullptr);
    cipher_temp_pool = new Ciphertext(N, L, L, context.slots);

    add_const_copy_vec = vector<uint64_tt>((L+1) * 4);

    cudaMalloc(&add_const_buffer, sizeof(uint64_tt) * (L+1) * 4);

    cudaMalloc(&rotKey_pointer_device, sizeof(uint64_tt*));
    cudaMalloc(&rotSlots_device, sizeof(int));
}

/**
 * generates key for public encryption (key is stored in keyMap)
*/
void BFVScheme::addEncKey(SecretKey& secretKey, cudaStream_t stream)
{
    int N = context.N;
    int L = context.L;
    int K = context.K;

    // pk = (a, -as + e)
    publicKey = new Key(N, L, 0, 1);

    Sampler::uniformSampler_xq(context.randomArray_pk_device, publicKey->ax_device, N, 0, K, L+1);

    uint64_tt* ex;
    cudaMalloc(&ex, sizeof(uint64_tt) * N * (L+1));

    // only NTT on QL
    Sampler::gaussianSampler_xq(context.randomArray_e_pk_device, ex, N, 0, K, L+1);
    // context.forwardNTT_batch(ex, 0, K, 1, L+1);

    // cudaMemset(ex , 0 , sizeof(uint64_tt) * N * (L+1));

    context.ToNTTInplace(ex, 0, K, 1, L+1,L+1);

    barrett_batch_3param_device(publicKey->bx_device, publicKey->ax_device, secretKey.sx_device, N, 0, 0, K, K, L+1);
    poly_sub2_batch_device(ex, publicKey->bx_device, N, 0, 0, K, L+1);

    cudaMemcpy(context.sec_buffer, secretKey.sx_device, sizeof(uint64_tt) * N * (L+K+1), cudaMemcpyDeviceToDevice);

	cudaFree(ex);
}

void BFVScheme::encryptZero(Ciphertext& cipher, int l, int slots, cudaStream_t stream)
{
    int N = context.N;
    int L = context.L;
    int K = context.K;



    RNG::generateRandom_device(in_enc, random_len_for_enc);
    
    Sampler::ZOSampler(in_enc, vx_enc, N, 0.5, 0, K, l+1);
    context.ToNTTInplace(vx_enc, 0, K, 1, l+1, L+1);

    barrett_batch_3param_device(cipher.ax_device, vx_enc, publicKey->ax_device, N, 0, 0, 0, K, l+1);

    Sampler::gaussianSampler_xq(in_enc + N * sizeof(uint8_tt), ex_enc, N, 0, K, l+1);
    context.ToNTTInplace(ex_enc, 0, K, 1, l+1, L+1);

    poly_add_batch_device(cipher.ax_device, ex_enc, N, 0, 0, K, l+1);

    barrett_batch_3param_device(cipher.bx_device, vx_enc, publicKey->bx_device, N, 0, 0, 0, K, l+1);

    Sampler::gaussianSampler_xq(in_enc + N * sizeof(uint8_tt) + N * sizeof(uint32_tt), ex_enc, N, 0, K, l+1);
    context.ToNTTInplace(ex_enc, 0, K, 1, l+1, L+1);

    poly_add_batch_device(cipher.bx_device, ex_enc, N, 0, 0, K, l+1);


}


void BFVScheme::encryptMsg(Ciphertext& cipher, Plaintext& message, cudaStream_t stream)
{
    cipher.l = message.l;
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int q_num = context.q_num;
    int level = message.l;
    
    encryptZero(cipher, message.l, message.slots, stream);

    cudaMemcpy(context.encode_buffer, message.mx_device, sizeof(uint64_tt) * N * (L+1), cudaMemcpyDeviceToDevice);

    uint64_tt gridDimGlb = N * q_num / blockDimGlb.x;
    bfv_add_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb, 0>>>(
            context.encode_buffer, context.encode_buffer,
            context.negQl_mod_t,
            context.negQl_mod_t_shoup,
            context.tInv_mod_q,
            context.tInv_mod_q_shoup,
            context.qVec.data(),
            context.plain_modulus, N, q_num);

	context.ToNTTInplace(context.encode_buffer, 0, K, 1, level+1, L+1);//NTT
	// print_device_array(msg.mx_device, N, L+1, "encode");

    // cudaMemset(cipher.bx_device, 0, sizeof(uint64_tt) * context.N * (cipher.L + 1));
    poly_add_batch_device(cipher.bx_device, context.encode_buffer, context.N, 0, 0, context.K, message.l+1);
}

void BFVScheme::decryptMsg(Plaintext& plain, SecretKey& secretKey, Ciphertext& cipher, cudaStream_t stream)
{
    int N = context.N;
    int l = cipher.l;
    int K = context.K;
    int L = context.L;
    int q_num = context.q_num;
    plain.l = cipher.l;
    plain.slots = cipher.slots;

    // cudaMemset(plain.mx_device + (plain.l+1)*N, 0, sizeof(uint64_tt) * N * (context.L-plain.l));
    cudaMemset(plain.mx_device, 0, sizeof(uint64_tt) * N * (context.L + 1));
    barrett_batch_3param_device(plain.mx_device, cipher.ax_device, secretKey.sx_device, N, 0, 0, K, K, l+1);
    poly_add_batch_device(plain.mx_device, cipher.bx_device, N, 0, 0, K, l+1);

	context.FromNTTInplace(plain.mx_device, 0, K, 1, l+1, L+1);

	hps_decrypt_scale_and_round(plain.mx_device, plain.mx_device, 0);
}

// Homomorphic Addition
void BFVScheme::add(Ciphertext& cipher_res, Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int L = context.L;
    int l = min(cipher1.l, cipher2.l);
    int K = context.K;
    int slots = cipher1.slots;
    cipher_res.l = l;

    cipher_add_3param_batch_device(cipher_res.cipher_device, cipher1.cipher_device, cipher2.cipher_device, N, K, l+1, L+1);
}

void BFVScheme::addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int L = context.L;
    int l = min(cipher1.l, cipher2.l);
    int K = context.K;
    cipher1.l = l;

    cipher_add_batch_device(cipher1.cipher_device, cipher2.cipher_device, N, K, l+1, L+1);
}

void BFVScheme::addConstAndEqual(Ciphertext& cipher, Plaintext& cnst)
{
    int l = min(cipher.l, cnst.l);
    cipher.l = l;

    int N = context.N;
    int L = context.L;
    int K = context.K;
    int q_num = context.q_num;
    int level = cnst.l;

    cudaMemcpy(context.encode_buffer, cnst.mx_device, sizeof(uint64_tt) * N * (L+1), cudaMemcpyDeviceToDevice);

    uint64_tt gridDimGlb = N * q_num / blockDimGlb.x;
    bfv_add_timesQ_overt_kernel<<<gridDimGlb, blockDimGlb, 0>>>(
            context.encode_buffer, context.encode_buffer,
            context.negQl_mod_t,
            context.negQl_mod_t_shoup,
            context.tInv_mod_q,
            context.tInv_mod_q_shoup,
            context.qVec.data(),
            context.plain_modulus, N, q_num);

    context.ToNTTInplace(context.encode_buffer, 0, K, 1, level+1, L+1);//NTT

    poly_add_batch_device(cipher.bx_device, context.encode_buffer, context.N, 0, 0, context.K, cnst.l+1);
}

void BFVScheme::addConstAndEqual(Ciphertext& cipher, uint64_tt cnst)
{
    multConstAndEqual(cipher_const_1, cnst);
    addAndEqual(cipher, cipher_const_1);
}

void BFVScheme::multConstAndEqual(Ciphertext& cipher, Plaintext& cnst)
{
    int L = context.L;
    int N = context.N;
    int K = context.K;
    int level = cnst.l;
    if(cipher.l == 0){
        throw invalid_argument("Ciphertexts are on level 0");
    }

    // cudaMemcpy(context.encode_buffer, cnst.mx_device, sizeof(uint64_tt) * N * (L+1), cudaMemcpyDeviceToDevice);

    // context.ToNTTInplace(context.encode_buffer, 0, K, 1, level+1, L+1);//NTT

    // barrett_2batch_device(cipher.cipher_device, context.encode_buffer, N, 0, 0, K, cipher.l+1, L+1);
    // barrett_batch_device(cipher.ax_device, cnst.mx_device, N, 0, 0, K, cipher.l+1);
    // barrett_batch_device(cipher.bx_device, cnst.mx_device, N, 0, 0, K, cipher.l+1);

    barrett_2batch_device(cipher.cipher_device, cnst.mx_device, N, 0, 0, K, cipher.l+1, L+1);
}


// can't mult negative real number
void BFVScheme::multConstAndEqual(Ciphertext& cipher, uint64_tt cnst)
{
    int L = context.L;
    if(cnst == 1) return;
    if(cnst == -1)
    {
        negateAndEqual(cipher);
        return;
    }
    if(cipher.l == 0){
        throw invalid_argument("Ciphertexts are on level 0");
    }

    // NTL::ZZ scaled_cnst(context.qVec[cipher.l] * cnst);
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = cnst % context.qVec[i];
        add_const_copy_vec[i + 2*(L+1)] = x_Shoup(add_const_copy_vec[i], context.qVec[i]);
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    context.poly_mul_const_batch_device(cipher.ax_device, add_const_buffer, context.K, cipher.l+1);
}

// c1 += c2*cnst
// c1.scale = c2.scale * target_scale
void BFVScheme::multConstAndAddCipherEqual(Ciphertext& c1, Ciphertext& c2, uint64_tt cnst)
{
    if(c2.l == 0){
        throw invalid_argument("Ciphertexts are on level 0");
    }
    int L = context.L;
    c1.l = min(c1.l, c2.l);

    // NTL::ZZ scaled_cnst = to_ZZ(target_scale * cnst);
    for(int i = 0; i < c1.l+1; i++)
    {
        add_const_copy_vec[i] = cnst % context.qVec[i];
        add_const_copy_vec[i + 2*(L+1)] = x_Shoup(add_const_copy_vec[i], context.qVec[i]);
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);
    context.poly_mul_const_add_cipher_batch_device(c1.cipher_device, c2.cipher_device, add_const_buffer, context.K, c1.l+1);
}

// Homomorphic Substraction
void BFVScheme::sub(Ciphertext& cipher_res, Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int L = context.L;
    int l = min(cipher1.l, cipher2.l);
    cipher_res.l = cipher1.l;
    int K = context.K;
    int slots = cipher1.slots;

    poly_sub_3param_batch_device(cipher_res.ax_device, cipher1.ax_device, cipher2.ax_device, N, 0, 0, 0, K, l+1);
    poly_sub_3param_batch_device(cipher_res.bx_device, cipher1.bx_device, cipher2.bx_device, N, 0, 0, 0, K, l+1);
}

void BFVScheme::subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2)
{
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int K = context.K;

    poly_sub_batch_device(cipher1.ax_device, cipher2.ax_device, N, 0, 0, K, l+1);
    poly_sub_batch_device(cipher1.bx_device, cipher2.bx_device, N, 0, 0, K, l+1);
}

// Homomorphic Negation
void BFVScheme::negate(Ciphertext& cipher_res, Ciphertext& cipher)
{
    cipher_negate_3param_batch_device(cipher.cipher_device, cipher_res.cipher_device, context.N, cipher.L+1, context.K, cipher.l+1);
}

void BFVScheme::negateAndEqual(Ciphertext& cipher)
{
    cipher_negate_batch_device(cipher.cipher_device, context.N, cipher.L+1, context.K, cipher.l+1);
}

void BFVScheme::divByiAndEqual(Ciphertext& cipher)
{
    context.divByiAndEqual(cipher.cipher_device, context.K, cipher.l+1);
}

void BFVScheme::mulByiAndEqual(Ciphertext& cipher)
{
    context.mulByiAndEqual(cipher.cipher_device, context.K, cipher.l+1);
}

__global__ void divByPo2_device_kernel(uint64_tt* device_c, uint32_tt n,int q_num)
{
    register uint32_tt index = blockIdx.y;
	register int idx_in_cipher = blockIdx.z;
    register int i = blockIdx.x * poly_block + threadIdx.x + blockIdx.y * n;
    register uint64_tt ra = device_c[i + idx_in_cipher * q_num * n] >> 1;
    device_c[i + idx_in_cipher * q_num * n] = ra;
}

void BFVScheme::divByPo2AndEqual(Ciphertext& cipher)
{
    int N = context.N;
    int K = context.K;
    int l = cipher.l;
    int L = context.L;
    dim3 div_dim(N / poly_block , l, 2);
    divByPo2_device_kernel<<< div_dim, poly_block >>>(cipher.cipher_device, N, L+1);
}

//bfv decrypt
void BFVScheme::hps_decrypt_scale_and_round(uint64_tt *dst, uint64_tt *src, const cudaStream_t &stream) const
{
    uint64_tt t = context.plain_modulus;
    size_t n = context.N;
    size_t q_num = context.q_num;
    size_t qMSB = context.qMSB;
    size_t sizeQMSB = context.sizeQMSB;
    size_t tMSB = context.tMSB;
    uint64_tt gridDimGlb = n / blockDimGlb.x;
    if (qMSB + sizeQMSB < 52)
    {
        if ((qMSB + tMSB + sizeQMSB) < 52)
        {
            hps_decrypt_scale_and_round_kernel_small_lazy<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, context.t_QHatInv_mod_q_div_q_mod_t, context.t_QHatInv_mod_q_div_q_frac, t, n, q_num);
        }else
        {
            hps_decrypt_scale_and_round_kernel_small<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, context.t_QHatInv_mod_q_div_q_mod_t, context.t_QHatInv_mod_q_div_q_mod_t_shoup,
                    context.t_QHatInv_mod_q_div_q_frac, t, n, q_num);
        }
    }else
    {
        // qMSB + sizeQMSB >= 52
        size_t qMSBHf = qMSB >> 1;
        if ((qMSBHf + tMSB + sizeQMSB) < 52)
        {
            hps_decrypt_scale_and_round_kernel_large_lazy<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, context.t_QHatInv_mod_q_div_q_mod_t, context.t_QHatInv_mod_q_div_q_frac,
                    context.t_QHatInv_mod_q_B_div_q_mod_t, context.t_QHatInv_mod_q_B_div_q_frac, t, n, q_num,
                    qMSBHf);
        }else
        {
            hps_decrypt_scale_and_round_kernel_large<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, context.t_QHatInv_mod_q_div_q_mod_t, context.t_QHatInv_mod_q_div_q_mod_t_shoup,
                    context.t_QHatInv_mod_q_div_q_frac, context.t_QHatInv_mod_q_B_div_q_mod_t,
                    context.t_QHatInv_mod_q_B_div_q_mod_t_shoup, context.t_QHatInv_mod_q_B_div_q_frac, t, n, q_num,
                    qMSBHf);
        }
    }
}

//-----------------------------------rescale-----------------------------------//
#define rescale_block 256

__global__ void rescaleAndEqual_kernel(uint64_tt* device_a, uint32_tt n, int p_num, int q_num, int l, uint64_tt* qiInvVecModql_device, uint64_tt* qiInvVecModql_shoup_device)
{
	register int idx_in_pq = blockIdx.y;
	register int idx_in_poly = blockIdx.x * rescale_block + threadIdx.x;
	register int idx_in_cipher = blockIdx.z;
    register int idx = idx_in_poly + (idx_in_cipher * q_num + idx_in_pq) * n;

    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
	register uint128_tt mu_q(pqt_mu_cons_high[p_num + idx_in_pq], pqt_mu_cons_low[p_num + idx_in_pq]);

	register uint64_tt ra = device_a[idx] + 2*q - device_a[(idx_in_cipher*q_num + l) * n + idx_in_poly];

	csub_q(ra, q);
	register uint64_tt ql_inv = qiInvVecModql_device[idx_in_pq];
	register uint64_tt ql_inv_shoup = qiInvVecModql_shoup_device[idx_in_pq];	
	device_a[idx] = mulMod_shoup(ra, ql_inv, ql_inv_shoup, q);
}

__global__ void rescaleAndEqual_new_kernel(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int p_num, int q_num, int l, uint64_tt* qiInvVecModql_device, uint64_tt* qiInvVecModql_shoup_device)
{
	register int idx_in_pq = blockIdx.y;
	register int idx_in_poly = blockIdx.x * rescale_block + threadIdx.x;
	register int idx_in_cipher = blockIdx.z;
    register int idx = idx_in_poly + (idx_in_cipher * q_num + idx_in_pq) * n;

    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
	register uint128_tt mu_q(pqt_mu_cons_high[p_num + idx_in_pq], pqt_mu_cons_low[p_num + idx_in_pq]);

	register uint64_tt ra = device_a[idx] + 2*q - device_b[idx];
	csub_q(ra, q);
	register uint64_tt ql_inv = qiInvVecModql_device[idx_in_pq];
	register uint64_tt ql_inv_shoup = qiInvVecModql_shoup_device[idx_in_pq];	
	device_a[idx] = mulMod_shoup(ra, ql_inv, ql_inv_shoup, q);
}

__global__ void mod_to_qi(uint64_tt* device_a, uint64_tt* device_b, uint32_tt n, int p_num, int q_num, int l)
{
	register int idx_in_pq = blockIdx.y;
	register int idx_in_poly = blockIdx.x * rescale_block + threadIdx.x;
	register int idx_in_cipher = blockIdx.z;
    
    register uint64_tt q = pqt_cons[p_num + idx_in_pq];
    register uint64_tt rb = device_b[idx_in_poly + (idx_in_cipher*q_num + l)*n];
    barrett_reduce_uint64_uint64(rb, q, pqt_mu_cons_high[p_num + idx_in_pq]);
    device_a[idx_in_poly + (idx_in_cipher*q_num + idx_in_pq)*n] = rb;
}

void BFVScheme::rescaleAndEqual(Ciphertext& cipher)
{
    int N = context.N;
    int K = context.K;
    int L = cipher.L;

    context.FromNTTInplace(cipher.cipher_device, cipher.l, K+cipher.l, 2, 1, L+1);
    dim3 resc_dim(N / rescale_block, cipher.l, 2);
    mod_to_qi <<< resc_dim, rescale_block >>> (rescale_buffer, cipher.cipher_device, N, K, L+1, cipher.l);

    context.ToNTTInplace(rescale_buffer, 0, K, 2, cipher.l, L+1);
    rescaleAndEqual_new_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, rescale_buffer, N, K, L+1, cipher.l, context.qiInvVecModql_device + cipher.l*(cipher.l-1)/2, context.qiInvVecModql_shoup_device + cipher.l*(cipher.l-1)/2);
    // cipher.scale = cipher.scale / context.qVec[cipher.l];
    cipher.l -= 1;

    // cudaMemset(cipher.ax_device + (cipher.l+1)*N, 0, sizeof(uint64_tt) * N);
    // cudaMemset(cipher.bx_device + (cipher.l+1)*N, 0, sizeof(uint64_tt) * N);
}

void BFVScheme::rescaleToAndEqual(Ciphertext& cipher, int level)
{
    int N = context.N;
    int K = context.K;
    int L = context.L;
    int last_l = cipher.l;

    context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    while(cipher.l > level)
    {
        int l = cipher.l;
	    dim3 resc_dim(N / rescale_block, l, 2);
	    rescaleAndEqual_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, N, K, L+1, l, context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
        cipher.l -= 1;
        // cipher.scale = cipher.scale / context.qVec[l];
        // cout<<"cipher.level: "<<cipher.l<<"  cipher.scale: "<<cipher.scale<<endl;
    }
    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
}

void BFVScheme::rescaleAndEqual_noNTT(Ciphertext& cipher)
{
    int N = context.N;
    int K = context.K;
    int l = cipher.l;
    int L = cipher.L;

	//context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    dim3 resc_dim(N / rescale_block, l, 2);
	rescaleAndEqual_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, N, K, L+1, l, context.qiInvVecModql_device + l*(l-1)/2, context.qiInvVecModql_shoup_device + l*(l-1)/2);
    cipher.l -= 1;
    // cipher.scale = cipher.scale / context.qVec[l];
    //context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
}