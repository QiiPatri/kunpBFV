#pragma once

#include "BFVScheme.h"
#include "Key_decomp.cuh"

Key_decomp *BFVScheme::addSWKey_23(SecretKey &secretKey, uint64_tt *s2, cudaStream_t stream)
{
    int N = context.N;
    int dnum = context.dnum;
    int gamma = context.gamma;
    int Ri_blockNum = context.Ri_blockNum;

    Key_decomp *swk = new Key_decomp(N, dnum, gamma, Ri_blockNum);

    return swk;
}

/**
 * generates key for multiplication (key is stored in keyMap)
 */
void BFVScheme::addMultKey_23(SecretKey &secretKey, cudaStream_t stream)
{
    if (rlk_23 != nullptr)
        return;
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;

    //rlk = (a' , -a's + e' + P·s^2)
    rlk_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);

    barrett_batch_3param_device(sxsx, secretKey.sx_device, secretKey.sx_device, N, 0, K, K, K, L + 1);

    long randomArray_len = sizeof(uint64_tt) * dnum * N * (L+1+K) + sizeof(uint32_tt) * dnum * N;
    RNG::generateRandom_device(context.randomArray_swk_device, randomArray_len);

    for (int i = 0; i < dnum; i++)
    {
        sxsx_mul_P_3param(temp_mul, sxsx, N, K + i * K, i * K, K + i * K, K, K);

        Sampler::gaussianSampler_xq(context.randomArray_e_swk_device + i * N * sizeof(uint32_tt) / sizeof(uint8_tt), ex_swk, N, 0, 0, K + L + 1);
        context.ToNTTInplace(ex_swk, 0, 0, 1, K + L + 1, K + L + 1);

        poly_add_batch_device(temp_mul, ex_swk, N, 0, 0, 0, K + L + 1);
        Sampler::uniformSampler_xq(context.randomArray_swk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (rlk_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);

        barrett_batch_3param_device((rlk_23->bx_device) + i * N * t_num * Ri_blockNum, (rlk_23->ax_device) + i * N * t_num * Ri_blockNum, secretKey.sx_device, N, 0, 0, 0, 0, K + L + 1);
        poly_sub2_batch_device(temp_mul, (rlk_23->bx_device) + i * N * t_num * Ri_blockNum, N, 0, 0, 0, K + L + 1);
        cudaMemset(temp_mul, 0, sizeof(uint64_tt) * N * (K + L + 1));
    }

    context.FromNTTInplace((rlk_23->ax_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    context.FromNTTInplace((rlk_23->bx_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);

    for (int i = 0; i < dnum; i++)
    {
        cudaMemcpy(modUp_RitoT_temp, (rlk_23->ax_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rlk_23->ax_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
        cudaMemcpy(modUp_RitoT_temp, (rlk_23->bx_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rlk_23->bx_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
    }
    context.ToNTTInplace((rlk_23->cipher_device), 0, K + L + 1, dnum * Ri_blockNum * 2, t_num, t_num);
}

/**
 * generates key for conjugation (key is stored in keyMap)
 */
void BFVScheme::addConjKey_23(SecretKey &secretKey, cudaStream_t steam)
{
    if (ConjKey_23 != nullptr)
        return;
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;
    // ConjKey
    ConjKey_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);
    // sk_conj
    sk_and_poly_conjugate(sxsx, secretKey.sx_device, N, L+1, 0, K, L+1);

    long randomArray_len = sizeof(uint64_tt) * dnum * N * (L+1+K) + sizeof(uint32_tt) * dnum * N;
    RNG::generateRandom_device(context.randomArray_swk_device, randomArray_len);

    for (int i = 0; i < dnum; i++)
    {
        sxsx_mul_P_3param(temp_mul, sxsx, N, K + i * K, i * K, K + i * K, K, K);

        Sampler::gaussianSampler_xq(context.randomArray_e_swk_device + i * N * sizeof(uint32_tt) / sizeof(uint8_tt), ex_swk, N, 0, 0, K + L + 1);
        context.ToNTTInplace(ex_swk, 0, 0, 1, K + L + 1, K + L + 1);

        poly_add_batch_device(temp_mul, ex_swk, N, 0, 0, 0, K + L + 1);
        // Sampler::uniformSampler_xq(context.randomArray_conjk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);
        Sampler::uniformSampler_xq(context.randomArray_swk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);

        barrett_batch_3param_device((ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, secretKey.sx_device, N, 0, 0, 0, 0, K + L + 1);
        poly_sub2_batch_device(temp_mul, (ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, N, 0, 0, 0, K + L + 1);
        cudaMemset(temp_mul, 0, sizeof(uint64_tt) * N * (K + L + 1));
    }

    context.FromNTTInplace((ConjKey_23->ax_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    context.FromNTTInplace((ConjKey_23->bx_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    for (int i = 0; i < dnum; i++)
    {
        cudaMemcpy(modUp_RitoT_temp, (ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((ConjKey_23->ax_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
        cudaMemcpy(modUp_RitoT_temp, (ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((ConjKey_23->bx_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
    }
    context.ToNTTInplace((ConjKey_23->cipher_device), 0, K + L + 1, dnum * Ri_blockNum * 2, t_num, t_num);
}

/**
 * generates key for left rotation <Hoisting Rotation> (key is stored in leftRotKeyMap)
 */
void BFVScheme::addLeftRotKey_23(SecretKey &secretkey, long rot_num, cudaStream_t stream)
{
    if (rotKey_vec_23[rot_num] != nullptr)
        return;
    int N = context.N;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;

    Key_decomp *rotKey_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);

    cudaMemcpy(sx_coeff, secretkey.sx_device, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
    sk_and_poly_LeftRot_inv(sxsx, sx_coeff, context.rotGroups_device, N, 0, rot_num, 0, 0, K+L+1);

    long randomArray_len = sizeof(uint64_tt) * dnum * N * (L+1+K) + sizeof(uint32_tt) * dnum * N;
    RNG::generateRandom_device(context.randomArray_swk_device, randomArray_len);

    for (int i = 0; i < dnum; i++)
    {
        sxsx_mul_P_3param(temp_mul, sx_coeff + N * K, N, K + i * K, i * K, K + i * K, K, K);

        Sampler::gaussianSampler_xq(context.randomArray_e_swk_device + i * N * sizeof(uint32_tt) / sizeof(uint8_tt), ex_swk, N, 0, 0, K + L + 1);
        context.ToNTTInplace(ex_swk, 0, 0, 1, K + L + 1, K + L + 1);

        poly_add_batch_device(temp_mul, ex_swk, N, 0, 0, 0, K + L + 1);
        // Sampler::uniformSampler_xq(context.randomArray_rotk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);
        Sampler::uniformSampler_xq(context.randomArray_swk_device + i * N * (L + K + 1) * sizeof(uint64_tt) / sizeof(uint8_tt), (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, N, 0, 0, K + L + 1);

        barrett_batch_3param_device((rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, sxsx, N, 0, 0, 0, 0, K + L + 1);
        poly_sub2_batch_device(temp_mul, (rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, N, 0, 0, 0, K + L + 1);
        cudaMemset(temp_mul, 0, sizeof(uint64_tt) * N * (K + L + 1));
    }

    context.FromNTTInplace((rotKey_23->ax_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    context.FromNTTInplace((rotKey_23->bx_device), 0, 0, dnum, K + L + 1, t_num * Ri_blockNum);
    for (int i = 0; i < dnum; i++)
    {
        cudaMemcpy(modUp_RitoT_temp, (rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rotKey_23->ax_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
        cudaMemcpy(modUp_RitoT_temp, (rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, sizeof(uint64_tt) * N * (K + L + 1), cudaMemcpyDeviceToDevice);
        context.modUpPQtoT_23((rotKey_23->bx_device) + i * N * t_num * Ri_blockNum, modUp_RitoT_temp, L, 1);
    }
    context.ToNTTInplace((rotKey_23->cipher_device), 0, K + L + 1, dnum * Ri_blockNum * 2, t_num, t_num);
    rotKey_vec_23[rot_num] = rotKey_23;
}

/**
 * generates key for left rotation (key is stored in leftRotKeyMap)
 */
void BFVScheme::addAutoKey_23(SecretKey &secretkey, int d, cudaStream_t stream)
{
    int N = context.N;
    if (d < 0 && d > log2(N))
    {
        throw invalid_argument("autoKey only for i in range(logn, logN)");
    }
    if (autoKey_vec_23[d] != nullptr)
        return;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int dnum = context.dnum;
    int Ri_blockNum = context.Ri_blockNum;

    Key_decomp *autoKey_23 = new Key_decomp(N, dnum, t_num, Ri_blockNum);

    // xxx add code here
    autoKey_vec_23[d] = autoKey_23;
}

void BFVScheme::mult_23(Ciphertext &cipher_res, Ciphertext &cipher1, Ciphertext &cipher2)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher_res.l = l;
    // cipher_res.scale = cipher1.scale * cipher2.scale;

    // compute_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);
    compute_t_QInv_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1, cipher2);

    // cudaMemcpy(cipher_res.bx_device, bxbx_mul, sizeof(uint64_tt) * slots * (L + 1), cudaMemcpyDeviceToDevice);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_res.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher_res.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher_res.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);
    // cudaMemset

    // cudaMemcpy(cipher_res.cipher_device, bxbx_mul, sizeof(uint64_tt) * slots * (L + 1), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(cipher_res.cipher_device, context.mult_scale_buffer, 2 * N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
}

void BFVScheme::multAndEqual_23(Ciphertext &cipher1, Ciphertext &cipher2)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    // cipher1.scale = cipher1.scale * cipher2.scale;

    //axax = a1a2 , bxbx = b1b2 , axbx1 = a1b2 + a2b1
    // compute_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);

    compute_t_QInv_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1, cipher2);

    // print_device_array(axax_mul, N, "axax_mul");
    // print_device_array(axbx1_mul, N, "axbx1_mul");
    // print_device_array(bxbx_mul, N, "bxbx_mul");

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);//intt
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // print_device_array(modUp_QjtoT_temp, N, "modUp_QjtoT_temp");

    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);
    // print_device_array(modUp_QjtoT_temp, N, "modUp_QjtoT_temp");

    // print_device_array(rlk_23->bx_device, N, 3, "rlk_23->cipher_device");
    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    // print_device_array(exProduct_T_temp, N, "exProduct_T_temp");

    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    // print_device_array(exProduct_T_temp, N, "exProduct_T_temp_ax");
    // print_device_array(exProduct_T_temp + sizeof(uint64_tt) * N * t_num * Ri_blockNum, N, "exProduct_T_temp_bx");

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l,2 );
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    // print_device_array(modUp_TtoQj_buffer, N, L+1, "modUp_TtoQj_buffer");
    // print_device_array(cipher1.bx_device, N, L+1, "bx_device");

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher1.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);
    // print_device_array(cipher1.bx_device, N, L+1, "bx_device");

    cudaMemcpyAsync(cipher1.cipher_device, context.mult_scale_buffer, 2 * N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(cipher1.bx_device, bxbx_mul, sizeof(uint64_tt) * slots * (L + 1), cudaMemcpyDeviceToDevice);
}

void BFVScheme::multAndEqual_beforeIP_23(Ciphertext &cipher1, Ciphertext &cipher2, uint64_tt* IP_input, uint64_tt* axbx1_mul_batch, uint64_tt* bxbx_mul_batch)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    // cipher1.scale = cipher1.scale * cipher2.scale;

    compute_c0c1c2(axbx1_mul_batch, axax_mul, bxbx_mul_batch, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(IP_input, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(IP_input, 0, K + L + 1, cipher_blockNum, t_num, t_num);
}

void BFVScheme::multAndEqual_afterIP_23(Ciphertext &cipher1, Ciphertext &cipher2, uint64_tt* IP_output, uint64_tt* axbx1_mul_batch, uint64_tt* bxbx_mul_batch)
{
    if (cipher1.l == 0 || cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }

    int N = context.N;
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);

    context.FromNTTInplace_for_externalProduct(IP_output, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);
    context.modUpTtoPQl_23(modUp_TtoQj_buffer, IP_output, l, 2);
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    // cipher_add_axbx_batch_device(cipher1.cipher_device, axbx1_mul_batch, bxbx_mul_batch, N, K, l+1, L+1);
}

// Homomorphic Squaring
void BFVScheme::square(Ciphertext &cipher1, Ciphertext& cipher2)
{
    if (cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher2.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher2.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher1.l = cipher2.l;
    // cipher1.scale = cipher2.scale * cipher2.scale;

    // compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher2.ax_device, cipher2.bx_device, N, 0, K, l + 1);
    compute_t_QInv_c0c1c2_for_square(axbx1_mul, axax_mul, bxbx_mul, cipher2);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    cipher_add_axbx_batch_device(cipher1.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);

    // cudaMemcpy(cipher1.bx_device, bxbx_mul, sizeof(uint64_tt) * slots * (L + 1), cudaMemcpyDeviceToDevice);
}

void BFVScheme::squareAndEqual(Ciphertext &cipher)
{
    if (cipher.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int slots = cipher.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    // cipher.scale = cipher.scale * cipher.scale;

    // compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher.ax_device, cipher.bx_device, N, 0, K, l + 1);
    compute_t_QInv_c0c1c2_for_square(axbx1_mul, axax_mul, bxbx_mul, cipher);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    // d2*evk.a + (a0b1 + a1b0)
    cipher_add_axbx_batch_device(cipher.cipher_device, axbx1_mul, bxbx_mul, N, K, l+1, L+1);

    // cudaMemcpy(cipher.bx_device, bxbx_mul, sizeof(uint64_tt) * slots * (L + 1), cudaMemcpyDeviceToDevice);
}

void BFVScheme::conjugate_23(Ciphertext& cipher_res, Ciphertext &cipher)
{
    if (ConjKey_23 == nullptr)
    {
        throw invalid_argument("conjKey_23 not exists");
    }

    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int slots = cipher.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher_res.l = cipher.l;
    // cipher_res.scale = cipher.scale;

    // conj---(b,a)
    sk_and_poly_conjugate(bxbx_mul, cipher.bx_device, N, L+1, 0, 0, l + 1);
    sk_and_poly_conjugate(axax_mul, cipher.ax_device, N, L+1, 0, 0, l + 1);
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding(no inverse NTT)
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, ConjKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_res.cipher_device, modUp_TtoQj_buffer, l, 2);
    
    context.ToNTTInplace(cipher_res.cipher_device, 0, K, 2, l + 1, L + 1);

    poly_add_batch_device(cipher_res.bx_device, bxbx_mul, N, 0, 0, K, l + 1);
}


void BFVScheme::conjugateAndEqual_23(Ciphertext &cipher)
{
    if (ConjKey_23 == nullptr)
    {
        throw invalid_argument("conjKey_23 not exists");
    }

    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    // conj---(b,a)
    sk_and_poly_conjugate(bxbx_mul, cipher.bx_device, N, L+1, 0, 0, l + 1);
    sk_and_poly_conjugate(axax_mul, cipher.ax_device, N, L+1, 0, 0, l + 1);
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // a_conj zeroPadding(no inverse NTT)
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, ConjKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l, 2);
    
    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    poly_add_batch_device(cipher.bx_device, bxbx_mul, N, 0, 0, K, l + 1);
}

// Homomorphic Rotate <Hoisting Rotation>
void BFVScheme::leftRotateAndEqual_23(Ciphertext &cipher, long rotSlots)
{
    if (rotKey_vec_23[rotSlots] == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    Key_decomp *rotKey_23 = rotKey_vec_23[rotSlots];

    context.FromNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);
    // a_conj zeroPadding
    context.modUpQjtoT_23(modUp_QjtoT_temp, cipher.ax_device, l, 1);
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rotKey_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher_temp_pool->cipher_device, modUp_TtoQj_buffer, l, 2);
    
    poly_add_batch_device(cipher_temp_pool->bx_device, cipher.bx_device, N, 0, 0, K, l+1);
    sk_and_poly_LeftRot_double(cipher.cipher_device, cipher_temp_pool->cipher_device, context.rotGroups_device, N, K, L+1, rotSlots, 0, 0, l+1);
    
    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l+1, L+1);
    // context.ToNTTInplace(cipher_temp_pool->cipher_device, 0, K, 2, l+1, L+1);
    // sk_and_poly_LeftRot_ntt_double(cipher.cipher_device, cipher_temp_pool->cipher_device, context.rotGroups_device, N, K, L+1, rotSlots, 0, 0, l+1);
}

void BFVScheme::rightRotateAndEqual_23(Ciphertext &cipher, long rotSlots)
{
    long rotslots = context.Nh - (1 << rotSlots); // Convert to left shift
    leftRotateAndEqual_23(cipher, rotslots);
}

// f(X) -> f(X) + f(X^(2^d+1))
void BFVScheme::automorphismAndAdd(Ciphertext &cipher, int d)
{
}


void BFVScheme::square_uint64_add_const_rescale(Ciphertext& cipher1, Ciphertext &cipher2, uint64_tt cnst)
{
    if (cipher2.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher2.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int gamma = context.gamma;
    int slots = cipher2.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    cipher1.l = cipher2.l;
    // cipher1.scale = cipher2.scale * cipher2.scale;

    compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher2.ax_device, cipher2.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher1.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher1.cipher_device, 0, K, 2, l + 1, L + 1);

    // NTL::ZZ scaled_real = to_ZZ(round(cipher1.scale * cnst));
    for(int i = 0; i < cipher1.l+1; i++)
    {
        add_const_copy_vec[i] = cnst % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);

    poly_add_axbx_double_add_cnst_batch_device(cipher1.cipher_device, axbx1_mul, bxbx_mul, add_const_buffer, N, K, l+1, L+1);

    rescaleAndEqual(cipher1);
}

void BFVScheme::squareAndEqual_uint64_add_const_rescale(Ciphertext& cipher, uint64_tt cnst)
{
    if (cipher.l == 0)
    {
        throw invalid_argument("Ciphertexts are on level 0");
    }
    if (rlk_23 == nullptr)
    {
        throw invalid_argument("rotKey_23 not exists");
    }
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int slots = cipher.slots;
    int Ri_blockNum = context.Ri_blockNum;
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);
    int cipher_blockNum = ceil(double(l + 1) / K);

    // cipher.scale = cipher.scale * cipher.scale;

    compute_c0c1c2_square(axbx1_mul, axax_mul, bxbx_mul, cipher.ax_device, cipher.bx_device, N, 0, K, l + 1);

    // a0a1 on Q to coeff
    context.FromNTTInplace(axax_mul, 0, K, 1, l + 1, L + 1);
    // modUp Qj to T
    context.modUpQjtoT_23(modUp_QjtoT_temp, axax_mul, l, 1);
    // a0a1 on T to ntt
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);

    context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rlk_23->cipher_device, l);
    context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num, 2);

    context.modUpTtoPQl_23(modUp_TtoQj_buffer, exProduct_T_temp, l, 2);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l, 2);

    context.ToNTTInplace(cipher.cipher_device, 0, K, 2, l + 1, L + 1);

    // NTL::ZZ scaled_real = to_ZZ(round(cipher.scale * cnst));
    for(int i = 0; i < cipher.l+1; i++)
    {
        add_const_copy_vec[i] = cnst % context.qVec[i];
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);

    poly_add_axbx_double_add_cnst_batch_device(cipher.cipher_device, axbx1_mul, bxbx_mul, add_const_buffer, N, K, l+1, L+1);

    // context.FromNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    // dim3 resc_dim(N / rescale_block, cipher.l, 2);
	// rescaleAndEqual_kernel <<< resc_dim, rescale_block >>> (cipher.cipher_device, N, K, L+1, cipher.l, context.qiInvVecModql_device + cipher.l*(cipher.l-1)/2, context.qiInvVecModql_shoup_device + cipher.l*(cipher.l-1)/2);
    // cipher.scale = cipher.scale / context.qVec[cipher.l];
    // cipher.l -= 1;
    // context.ToNTTInplace(cipher.cipher_device, 0, K, 2, cipher.l+1, L+1);
    rescaleAndEqual(cipher);
}

//compute [t/q * (c0,c1,c2)]
void BFVScheme::compute_t_QInv_c0c1c2(uint64_tt *axbx1_mul, uint64_tt *axax_mul, uint64_tt *bxbx_mul, Ciphertext& cipher1, Ciphertext& cipher2)
{
    size_t N = static_cast<size_t>( context.N );
    int l = min(cipher1.l, cipher2.l);
    cipher1.l = l;
    int L = context.L;
    int K = context.K;
    int r_num = context.r_num;
    int gamma = context.gamma;
    int slots = cipher1.slots;

    size_t size_Q = static_cast<size_t>(L + 1);
    size_t size_R = static_cast<size_t>(r_num);
    size_t size_QR = size_Q + size_R;

    // print_device_array(cipher1.bx_device, N, "ct1.bx");
    // print_device_array(cipher2.bx_device, N, "ct2.bx");

    // context.FromNTTInplace(cipher1.ax_device, 0, K, 1, l+1, L+1);
    // context.FromNTTInplace(cipher1.bx_device, 0, K, 1, l+1, L+1);
    // context.FromNTTInplace(cipher2.ax_device, 0, K, 1, l+1, L+1);
    // context.FromNTTInplace(cipher2.bx_device, 0, K, 1, l+1, L+1);

    //=========================================ct1 BConv=========================================
    uint64_tt *ct1;
    cudaMalloc(&ct1, 3 * sizeof(uint64_tt) * N * (L + 1 + r_num));
    //ct1 ax
    uint64_tt *ct1_ptr = ct1;
    uint64_tt *ct1_Q_ptr = ct1_ptr;
    uint64_tt *ct1_R_ptr = ct1_Q_ptr + N * (L + 1);

    // print_device_array(cipher1.ax_device , N , "cipher1.ax_device");

    cudaMemcpyAsync(ct1_Q_ptr, cipher1.ax_device, N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    context.FromNTTInplace(ct1_Q_ptr, 0, K, 1, l+1, L+1);
    context.QtoR.bConv_HPS(ct1_R_ptr, ct1_Q_ptr, N, 0);

    // print_device_array(ct1_Q_ptr , N , "ct1_ax");
    // print_device_array(ct1_R_ptr , N , "ct1_R_ptr");

    // context.RtoQ.bConv_HPS(ct1_Q_ptr, ct1_R_ptr, N, 0);

    // print_device_array(ct1_Q_ptr , N , "ct1_ax");

    //ct1 bx
    ct1_ptr = ct1 + N * (L + 1 + r_num);
    ct1_Q_ptr = ct1_ptr;
    ct1_R_ptr = ct1_Q_ptr + N * (L + 1);
    cudaMemcpyAsync(ct1_Q_ptr, cipher1.bx_device, N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    context.FromNTTInplace(ct1_Q_ptr, 0, K, 1, l+1, L+1);
    context.QtoR.bConv_HPS(ct1_R_ptr, ct1_Q_ptr, N, 0);

    // print_device_array(ct1_Q_ptr , N , "ct1_bx");
    //=======================================================
    //check bconv
    // print_device_array(ct1_Q_ptr , N , size_Q , "ct1_bx_Q");
    // print_device_array(ct1_R_ptr , N , size_R , "ct1_bx_R");

    // context.RtoQ.bConv_HPS(ct1_Q_ptr, ct1_R_ptr, N, 0);
    // print_device_array(ct1_Q_ptr , N , "ct1_bx_Q");

    // context.QtoR.bConv_HPS(ct1_R_ptr, ct1_Q_ptr, N, 0);
    // print_device_array(ct1_R_ptr , N , "ct1_bx_R");

    // uint64_tt *ct1_bx_R_original;
    // cudaMalloc(&ct1_bx_R_original, N * r_num * sizeof(uint64_tt));
    // cudaMemcpyAsync(ct1_bx_R_original, ct1_R_ptr, N * r_num * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);

    //=========================================ct2 BConv=========================================
    uint64_tt *ct2;
    cudaMalloc(&ct2, 3 * sizeof(uint64_tt) * N * (L + 1 + r_num));
    //ct2 ax
    uint64_tt *ct2_ptr = ct2;
    uint64_tt *ct2_Q_ptr = ct2_ptr;
    uint64_tt *ct2_R_ptr = ct2_Q_ptr + N * (L + 1);
    cudaMemcpyAsync(ct2_Q_ptr, cipher2.ax_device, N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    context.FromNTTInplace(ct2_Q_ptr, 0, K, 1, l+1, L+1);
    context.QtoR.bConv_HPS(ct2_R_ptr, ct2_Q_ptr, N, 0);

    // print_device_array(ct2_Q_ptr , N , "ct2_ax");

    //ct2 bx
    ct2_ptr = ct2 + N * (L + 1 + r_num);
    ct2_Q_ptr = ct2_ptr;
    ct2_R_ptr = ct2_Q_ptr + N * (L + 1);
    cudaMemcpyAsync(ct2_Q_ptr, cipher2.bx_device, N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    context.FromNTTInplace(ct2_Q_ptr, 0, K, 1, l+1, L+1);
    context.QtoR.bConv_HPS(ct2_R_ptr, ct2_Q_ptr, N, 0);

    // print_device_array(ct2_Q_ptr , N , "ct2_bx_Q");
    // print_device_array(ct2_R_ptr , N , "ct2_bx_R");

    // print_device_array(ct2_Q_ptr , N , "ct2_bx");

    //axax = a1a2 , bxbx = b1b2 , axbx1 = a1b2 + a2b1
    // compute_c0c1c2(axbx1_mul, axax_mul, bxbx_mul, cipher1.ax_device, cipher2.ax_device, cipher1.bx_device, cipher2.bx_device, N, 0, K, l + 1);

    //test QR NTT is ok
    // ct1_ptr = ct1;
    // print_device_array(ct1_ptr, N, "ct1_0");
    // context.ToNTTInplace_for_QR(ct1_ptr, 0, 0, 1, size_QR, size_QR);
    // print_device_array(ct1_ptr, N, "ct1_after_NTT");
    // context.FromNTTInplace_for_QR(ct1_ptr, 0, 0, 1, size_QR, size_QR);
    // print_device_array(ct1_ptr, N, "ct1_after_INTT");

    // // check qr_cons
    // // =============================
	// uint64_tt* array_PQ = new uint64_tt[size_QR];
    // cudaDeviceSynchronize();
    // cudaMemcpyFromSymbol(array_PQ, qr_cons, sizeof(uint64_tt) * size_QR, 0, cudaMemcpyDeviceToHost);
    // printf("QR_MOD = [" );
    // for(int i = 0; i < size_QR; i++)
    // {
    //     printf("%llu, ", array_PQ[i]);
    // }
    // printf("]\n");
    // free(array_PQ);
    // // =============================

    //
    cudaMemset(context.mult_buffer, 0, sizeof(uint64_tt) * N * (context.L + 1));
    barrett_batch_3param_device(context.mult_buffer, cipher2.ax_device, context.sec_buffer, N, 0, 0, K, K, l+1);
    poly_add_batch_device(context.mult_buffer, cipher2.bx_device, N, 0, 0, K, l+1);

    context.FromNTTInplace(context.mult_buffer, 0, K, 1, l+1, L+1);
    hps_decrypt_scale_and_round(context.mult_buffer, context.mult_buffer, 0);
    context.ToNTTInplace(context.mult_buffer, 0, K, 1, l+1, L+1);//NTT

    cudaMemcpyAsync(context.mult_scale_buffer, cipher1.cipher_device, 2 * N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    barrett_2batch_device(context.mult_scale_buffer, context.mult_buffer, N, 0, 0, K, cipher1.l+1, L+1);
    //

    //NTT
    for (size_t i = 0; i < 2; i++)
    {
        ct1_ptr = ct1 + i * (L + 1 + r_num) * N;
        context.ToNTTInplace_for_QR(ct1_ptr, 0, 0, 1, size_QR, size_QR);
    
        // print_device_array(ct1_ptr , N , "ct1_device");
    }
    for (size_t i = 0; i < 2; i++)
    {
        ct2_ptr = ct2 + i * (L + 1 + r_num) * N;
        context.ToNTTInplace_for_QR(ct2_ptr, 0, 0, 1, size_QR, size_QR);

        // print_device_array(ct2_ptr , N , "ct2_device");
    }

    // (c0, c1) * (c0', c1') = (c0*c0', c0*c1' + c1*c0', c1*c1')
    uint64_tt gridDimGlb = N * (L + 1 + r_num) / blockDimGlb.x;
    tensor_prod_2x2_rns_poly<<<gridDimGlb, blockDimGlb, 0, 0>>>(
        ct1, ct2, context.QR_mod , context.QR_Mu_high , context.QR_Mu_low , ct1, N, size_QR );

    // print_device_array(ct1                           , N , "d0");
    // print_device_array(ct1 + 1 * (L + 1 + r_num) * N , N , "d1");
    // print_device_array(ct1 + 2 * (L + 1 + r_num) * N , N , "d2");

    //INTT
    for (size_t i = 0; i < 3; i++)
    {
        ct1_ptr = ct1 + i * (L + 1 + r_num) * N;
        context.FromNTTInplace_for_QR(ct1_ptr, 0, 0, 1, size_QR, size_QR);
    }

    // print_device_array(ct1                   , N , "d0_after_INTT");
    // print_device_array(ct1 + 1 * size_QR * N , N , "d1_after_INTT");
    // print_device_array(ct1 + 2 * size_QR * N , N , "d2_after_INTT");

    //=========================================ct1 BConv=========================================
    //scale and round
    uint64_tt *enc_ptr;
    uint64_tt *rVec_ = context.RtoQ.get_ibase();
    uint64_tt *rMu_high = context.RtoQ.get_ibase_Mu_high();
    uint64_tt *rMu_low = context.RtoQ.get_ibase_Mu_low();

    for (size_t i = 0; i < 3; i++)
    {
        if     ( i == 0 ) enc_ptr = axax_mul;
        else if( i == 1 ) enc_ptr = axbx1_mul;
        else              enc_ptr = bxbx_mul;

        ct1_ptr = ct1 + i * size_QR * N;
        uint64_tt *temp;
        cudaMalloc(&temp, N * r_num * sizeof(uint64_tt));
        // scale and round QlRl to Rl
        scaleAndRound_HPS_QR_R(temp, ct1_ptr, rVec_, rMu_high , rMu_low , 
            context.tRSHatInvModsDivsModr_, context.tRSHatInvModsDivsFrac_, size_Q , size_R , N , 0);

        // if(i == 2)
        // {
        //     print_device_array(temp , N , size_R , "temp");
        
        //     // cudaMemcpyAsync(temp, ct1_bx_R_original, N * r_num * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
        //     // print_device_array(temp , N , "temp");
        // }

        // Rl -> Ql
        context.RtoQ.bConv_HPS(enc_ptr, temp, N, 0);
        // context.RtoQ.bConv_HPS(enc_ptr, ct1_ptr + size_Q * N, N, 0);

        // if(i == 2)
        // {
        //     // print_device_array(enc_ptr , N , "Q_base");

        //     // context.QtoR.bConv_HPS(temp, enc_ptr, N, 0);
        //     // print_device_array(temp , N , "R_base");
    
        //     // context.RtoQ.bConv_HPS(enc_ptr, temp, N, 0);
        //     // print_device_array(enc_ptr , N , "Q_base");
        // }
    }

    // print_device_array(bxbx_mul , N , "bxbx");

    //NTT
    context.ToNTTInplace(axax_mul, 0, K, 1, l+1, L+1);
    context.ToNTTInplace(axbx1_mul, 0, K, 1, l+1, L+1);
    context.ToNTTInplace(bxbx_mul, 0, K, 1, l+1, L+1);

    // context.ToNTTInplace(cipher1.ax_device, 0, K, 1, l+1, L+1);
    // context.ToNTTInplace(cipher1.bx_device, 0, K, 1, l+1, L+1);
    // context.ToNTTInplace(cipher2.ax_device, 0, K, 1, l+1, L+1);
    // context.ToNTTInplace(cipher2.bx_device, 0, K, 1, l+1, L+1);
}

//compute [t/q * (c0,c1,c2)] for square
void BFVScheme::compute_t_QInv_c0c1c2_for_square(uint64_tt *axbx1_mul, uint64_tt *axax_mul, uint64_tt *bxbx_mul, Ciphertext& cipher)
{
    size_t N = static_cast<size_t>( context.N );
    int l = cipher.l;
    cipher.l = l;
    int L = context.L;
    int K = context.K;
    int r_num = context.r_num;
    int gamma = context.gamma;
    int slots = cipher.slots;

    size_t size_Q = static_cast<size_t>(L + 1);
    size_t size_R = static_cast<size_t>(r_num);
    size_t size_QR = size_Q + size_R;

    // context.FromNTTInplace(cipher.ax_device, 0, K, 1, l+1, L+1);
    // context.FromNTTInplace(cipher.bx_device, 0, K, 1, l+1, L+1);

    //=========================================ct1 BConv=========================================
    uint64_tt *ct1;
    cudaMalloc(&ct1, 3 * sizeof(uint64_tt) * N * (L + 1 + r_num));
    //ct1 ax
    uint64_tt *ct1_ptr = ct1;
    uint64_tt *ct1_Q_ptr = ct1_ptr;
    uint64_tt *ct1_R_ptr = ct1_Q_ptr + N * (L + 1);

    cudaMemcpyAsync(ct1_Q_ptr, cipher.ax_device, N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    context.FromNTTInplace(ct1_Q_ptr, 0, K, 1, l+1, L+1);
    context.QtoR.bConv_HPS(ct1_R_ptr, ct1_Q_ptr, N, 0);

    //ct1 bx
    ct1_ptr = ct1 + N * (L + 1 + r_num);
    ct1_Q_ptr = ct1_ptr;
    ct1_R_ptr = ct1_Q_ptr + N * (L + 1);
    cudaMemcpyAsync(ct1_Q_ptr, cipher.bx_device, N * (L + 1) * sizeof(uint64_tt), cudaMemcpyDeviceToDevice);
    context.FromNTTInplace(ct1_Q_ptr, 0, K, 1, l+1, L+1);
    context.QtoR.bConv_HPS(ct1_R_ptr, ct1_Q_ptr, N, 0);

    //NTT
    for (size_t i = 0; i < 2; i++)
    {
        ct1_ptr = ct1 + i * (L + 1 + r_num) * N;
        context.ToNTTInplace_for_QR(ct1_ptr, 0, 0, 1, size_QR, size_QR);
    
        // print_device_array(ct1_ptr , N , "ct1_device");
    }

    // (c0, c1) * (c0', c1') = (c0*c0', c0*c1' + c1*c0', c1*c1')
    uint64_tt gridDimGlb = N * (L + 1 + r_num) / blockDimGlb.x;
    tensor_prod_2x2_rns_poly_for_square<<<gridDimGlb, blockDimGlb, 0, 0>>>(
        ct1, context.QR_mod , context.QR_Mu_high , context.QR_Mu_low , ct1, N, size_QR );

    //INTT
    for (size_t i = 0; i < 3; i++)
    {
        ct1_ptr = ct1 + i * (L + 1 + r_num) * N;
        context.FromNTTInplace_for_QR(ct1_ptr, 0, 0, 1, size_QR, size_QR);
    }

    //=========================================ct1 BConv=========================================
    //scale and round
    uint64_tt *enc_ptr;
    uint64_tt *rVec_ = context.RtoQ.get_ibase();
    uint64_tt *rMu_high = context.RtoQ.get_ibase_Mu_high();
    uint64_tt *rMu_low = context.RtoQ.get_ibase_Mu_low();

    for (size_t i = 0; i < 3; i++)
    {
        if     ( i == 0 ) enc_ptr = axax_mul;
        else if( i == 1 ) enc_ptr = axbx1_mul;
        else              enc_ptr = bxbx_mul;

        ct1_ptr = ct1 + i * size_QR * N;
        uint64_tt *temp;
        cudaMalloc(&temp, N * r_num * sizeof(uint64_tt));
        // scale and round QlRl to Rl
        scaleAndRound_HPS_QR_R(temp, ct1_ptr, rVec_, rMu_high , rMu_low , 
            context.tRSHatInvModsDivsModr_, context.tRSHatInvModsDivsFrac_, size_Q , size_R , N , 0);
        // Rl -> Ql
        context.RtoQ.bConv_HPS(enc_ptr, temp, N, 0);
    }

    //NTT
    context.ToNTTInplace(axax_mul, 0, K, 1, l+1, L+1);
    context.ToNTTInplace(axbx1_mul, 0, K, 1, l+1, L+1);
    context.ToNTTInplace(bxbx_mul, 0, K, 1, l+1, L+1);

    // context.ToNTTInplace(cipher.ax_device, 0, K, 1, l+1, L+1);
    // context.ToNTTInplace(cipher.bx_device, 0, K, 1, l+1, L+1);
}