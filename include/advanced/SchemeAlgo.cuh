#pragma once

#include "SchemeAlgo.h"

SchemeAlgo::SchemeAlgo(Context_23& context, Scheme_23& scheme, SecretKey& secretkey):context(context), scheme(scheme), secretkey(secretkey)
{
    N = context.N;
	slots = context.slots;
	logN = context.logN;
	logslots = context.logslots;
    q_num = context.q_num;
    p_num = context.p_num;
    maxLevel = context.L;
    precision = context.precision;

}

void SchemeAlgo::malloc_bsgs_buffer(int bs, int gs, int sine_degree)
{
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int Qj_blockNum = context.Qj_blockNum;

    cudaMallocManaged(&bs_cipher_PQl, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2 * bs);
    // cudaMemset(bs_cipher_PQl, 0, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2 * bs);

    cudaMallocManaged(&gs_cipher_PQl, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2 );
    // cudaMemset(gs_cipher_PQl, 0, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);

    cudaMallocManaged(&gs_cipher_PQl_T, sizeof(uint64_tt) * N * t_num * Ri_blockNum* 2);
    cudaMallocManaged(&acc_cipher_PQl, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);
    cudaMallocManaged(&modDown_cipher_Ql, sizeof(uint64_tt) * N * q_num * 2);

    cudaMallocManaged(&IP_input_temp, sizeof(uint64_tt) * N * t_num * Qj_blockNum * (sine_degree/4));
    cudaMallocManaged(&IP_output_temp, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2 * (sine_degree/4));
    cudaMallocManaged(&axbx1_mul, sizeof(uint64_tt) * N * q_num * (sine_degree/4));
    cudaMallocManaged(&bxbx_mul, sizeof(uint64_tt) * N * q_num * (sine_degree/4));

    cudaMallocManaged(&chebyshev_tree_cipher_pool, sizeof(uint64_tt) * N * q_num * 2 * sine_degree);
    for(int i = 0; i < sine_degree; i++)
    {
        Ciphertext* this_cipher = new Ciphertext();

        this_cipher->N = N;
        this_cipher->L = maxLevel;
        this_cipher->l = maxLevel-4;
        this_cipher->slots = slots;
        this_cipher->scale = NTL::RR(precision);

        this_cipher->cipher_device = chebyshev_tree_cipher_pool + i * (N * q_num * 2);
        this_cipher->ax_device = this_cipher->cipher_device;
        this_cipher->bx_device = this_cipher->ax_device + (N * q_num);

        chebyshev_tree_pool.push_back(this_cipher);
    }

    add_const_copy_vec = vector<uint64_tt>(q_num * 4 * sine_degree);
    cudaMalloc(&add_const_buffer, sizeof(uint64_tt) * q_num * 4 * sine_degree);

    // for(int i = 0; i < sine_degree; i++)
    // {
    //     chebyshev_tree_pool.push_back(new Ciphertext(N, maxLevel, maxLevel-4, slots, NTL::RR(precision)));
    // }

    //cout<<"log2(sine_degree)"<<log2(sine_degree)<<endl;
    for(int i = 0; i < int(log2(sine_degree)); i++)
    {
        eval_sine_poly_pool.push_back(new Ciphertext(N, maxLevel, maxLevel-4, slots, NTL::RR(precision)));
    }
    for(int i = 0; i < sine_degree; i++)
    {
        chebyshev_poly_coeff_tree_pool.push_back(new Chebyshev_Polynomial());
    }


    plain_buffer = new Plaintext(N, maxLevel, maxLevel, slots, NTL::RR(precision));
    cudaMalloc(&complex_vals, sizeof(cuDoubleComplex) * slots);
}

void SchemeAlgo::evalLinearTransformAndEqual(Ciphertext &cipher, MatrixDiag* matrixDiag)
{
   int bs = matrixDiag->bs;
    int gs = matrixDiag->gs;
    int diag_num = matrixDiag->diag_num;
    int diag_gap = matrixDiag->diag_gap;
    int N = context.N;
    int l = cipher.l;
    int L = context.L;
    int K = context.K;
    int t_num = context.t_num;
    int Ri_blockNum = context.Ri_blockNum;
    int Qj_blockNum = context.Qj_blockNum;
    int cipher_blockNum = ceil(double(l + 1) / K);
    int swk_blockNum = ceil(double(K + l + 1) / context.gamma);

    uint64_tt* modUp_QjtoT_temp = scheme.modUp_QjtoT_temp;
    uint64_tt* modUp_RitoT_temp = scheme.modUp_RitoT_temp;
    uint64_tt* exProduct_T_temp = scheme.exProduct_T_temp;
    uint64_tt* modUp_TtoQj_buffer = scheme.modUp_TtoQj_buffer;
    //(a,b) decompose a
    context.modUpQjtoT_23(modUp_QjtoT_temp, cipher.ax_device, l); // modUp Qj to T
    context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);
    //(P*a,P*b)
    cipher_mul_P_special(bs_cipher_PQl, cipher.ax_device, N, p_num, 0, K, K, l + 1, q_num, t_num * Ri_blockNum);
    for(int i = 1; i < bs; i++)
    {
        Key_decomp* rotKey_23 = scheme.rotKey_vec_23[i * diag_gap]; // input the idx
        context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rotKey_23->cipher_device, l);
        context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num);
        
        context.modUpRitoT_23(modUp_RitoT_temp, bs_cipher_PQl, l);
        poly_add_batch_device_many_poly(exProduct_T_temp, modUp_RitoT_temp, N , (Ri_blockNum * t_num), 0, K+L+1, t_num, Ri_blockNum);
        sk_and_poly_LeftRot_many_poly_T(bs_cipher_PQl + i * N * (t_num * Ri_blockNum) * 2, exProduct_T_temp, context.rotGroups_device, N, p_num, q_num, i*diag_gap, 0, 0, t_num, Ri_blockNum);
    }
    
    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // float MulPtMat=0;


    context.ToNTTInplace(bs_cipher_PQl, 0, K+L+1, 2 * bs * Ri_blockNum, t_num, t_num);
    for(int j = 0; j < gs; j++)
    {
        // (u0, u1) <- (0, 0)
        cudaMemset(acc_cipher_PQl, 0, sizeof(uint64_tt) * N * t_num * Ri_blockNum * 2);//(u_{0},u_{1})


            // cudaEventRecord(start);
        for(int i = 0; i < bs; i++)
        {
            int diag = j * bs + i;     
            if(diag < diag_num)
            {
                poly_mul_add_3param_batch_device(acc_cipher_PQl, bs_cipher_PQl + i * N * (t_num * Ri_blockNum) * 2, 
                matrixDiag -> diag_inv_vec_encodePQl_device + diag * N * t_num, N, 0, 0, 0, K+L+1, t_num, t_num * Ri_blockNum, 
                Ri_blockNum);
            }
        }
            // cudaEventRecord(end);
            // cudaEventSynchronize(end);
            // cudaEventElapsedTime(&MulPtMat, start, end);
            // cout<<"MulPtMat: "<<MulPtMat<<endl;

        context.FromNTTInplace(acc_cipher_PQl, 0, K+L+1, 2 * Ri_blockNum, t_num, t_num);
        
        int rot_num = bs*j*diag_gap;
        if(rot_num != 0)
        {
            Key_decomp* rotKey_23 = scheme.rotKey_vec_23[rot_num]; // input the idx
            
            context.modDownPQltoQl_23(modDown_cipher_Ql, acc_cipher_PQl, l);//u0
            // d = [u1]_Ql -> T
            context.modUpQjtoT_23(modUp_QjtoT_temp, modDown_cipher_Ql, l);
            context.ToNTTInplace(modUp_QjtoT_temp, 0, K + L + 1, cipher_blockNum, t_num, t_num);
            context.external_product_T(exProduct_T_temp, modUp_QjtoT_temp, rotKey_23->cipher_device, l);
            context.FromNTTInplace_for_externalProduct(exProduct_T_temp, 0, K + L + 1, swk_blockNum, t_num, t_num, Ri_blockNum*t_num);
            context.modUpRitoT_23(modUp_RitoT_temp, acc_cipher_PQl + N * t_num * Ri_blockNum,l);
            poly_add_batch_device_many_poly(exProduct_T_temp, modUp_RitoT_temp, N , (Ri_blockNum * t_num), 0, K+L+1, t_num, Ri_blockNum);
            sk_and_poly_LeftRot_many_poly_T(gs_cipher_PQl_T, exProduct_T_temp, context.rotGroups_device, N, p_num, q_num, rot_num, 0, 0, t_num, Ri_blockNum);
            poly_add_batch_device_many_poly(gs_cipher_PQl, gs_cipher_PQl_T, N, 0, 0, K+L+1, t_num, 2 * Ri_blockNum);
        }
        else{
            poly_add_batch_device_many_poly(gs_cipher_PQl, acc_cipher_PQl, N , 0, 0, K+L+1, t_num, 2 * Ri_blockNum);
        }
    }
    context.modUpTtoPQl_23(modUp_TtoQj_buffer, gs_cipher_PQl, l);
    context.modDownPQltoQl_23(cipher.cipher_device, modUp_TtoQj_buffer, l);//(a,b) 
    cipher.scale *= context.precision;
    scheme.rescaleAndEqual_noNTT(cipher);  
}

void splitCoeffsPolyVector(int split, vector<Chebyshev_Polynomial*> chebyshev_poly_coeff_tree_pool, int tree_idx)
{
    Chebyshev_Polynomial *coeffs = chebyshev_poly_coeff_tree_pool[tree_idx];
    Chebyshev_Polynomial *coeffsq = chebyshev_poly_coeff_tree_pool[tree_idx*2];
    Chebyshev_Polynomial *coeffsr = chebyshev_poly_coeff_tree_pool[tree_idx*2 + 1];

    coeffsr->coeffs = vector<double>(split);
    if(coeffs->maxDegree == coeffs->degree())
    {
        coeffsr->maxDegree = split - 1;
    }
    else
    {
        coeffsr->maxDegree = coeffs->maxDegree - (coeffs->degree() - split + 1);
    }

    for(int i = 0; i < split; i++) coeffsr->coeffs[i] = coeffs->coeffs[i];

    coeffsq->coeffs = vector<double>(coeffs->degree() - split + 1);
    coeffsq->maxDegree = coeffs->maxDegree;

    coeffsq->coeffs[0] = coeffs->coeffs[split];

    for(int i = split+1, j = 1; i < coeffs->degree() + 1; i++, j++)
    {
        coeffsq->coeffs[i - split] = coeffs->coeffs[i] * 2;
        coeffsr->coeffs[split - j] = coeffsr->coeffs[split - j] - coeffs->coeffs[i];
    }
}

bool isNotNegligible(double a)
{
    return abs(a) > 1e-14;
}

void SchemeAlgo::prepareChebyshevCoeffsTree(int logSplit, int logDegree, int tree_idx)
{
    Chebyshev_Polynomial* poly = chebyshev_poly_coeff_tree_pool[tree_idx];
    if(poly->degree() < (1 << logSplit))
    {
        if(logSplit > 1 && poly->maxDegree%(1<<(logSplit+1)) > (1<<(logSplit-1)))
        {
            logDegree = ceil(log2(poly->degree()));
            logSplit = logDegree >> 1;

            prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx);
            return;
        }
        return;
    }
    int nextPower = 1 << logSplit;
    for(nextPower; nextPower < (poly->degree()>>1) + 1;) 
        nextPower <<= 1;

    Chebyshev_Polynomial *coeffsq = chebyshev_poly_coeff_tree_pool[tree_idx*2];
    Chebyshev_Polynomial *coeffsr = chebyshev_poly_coeff_tree_pool[tree_idx*2 + 1];

    splitCoeffsPolyVector(nextPower, chebyshev_poly_coeff_tree_pool, tree_idx);

    prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx*2);
    prepareChebyshevCoeffsTree(logSplit, logDegree, tree_idx*2+1);
}

void SchemeAlgo::evalRecurse(NTL::RR target_scale, int logSplit, int logDegree, int tree_idx)
{
    Chebyshev_Polynomial* poly = chebyshev_poly_coeff_tree_pool[tree_idx];

    if(poly->degree() < (1 << logSplit))
    {
        if(logSplit > 1 && poly->maxDegree%(1<<(logSplit+1)) > (1<<(logSplit-1)))
        {
            logDegree = ceil(log2(poly->degree()));
            logSplit = logDegree >> 1;

            evalRecurse(target_scale, logSplit, logDegree, tree_idx);
            return;
        }
        evalFromPowerBasis(target_scale, tree_idx);
        return;
    }
    int nextPower = 1 << logSplit;
    for(nextPower; nextPower < (poly->degree()>>1) + 1;) 
        nextPower <<= 1;

    // Chebyshev_Polynomial coeffsq, coeffsr;
    Chebyshev_Polynomial *coeffsq = chebyshev_poly_coeff_tree_pool[tree_idx*2];
    Chebyshev_Polynomial *coeffsr = chebyshev_poly_coeff_tree_pool[tree_idx*2 + 1];

    // splitCoeffsPolyVector(nextPower, chebyshev_poly_coeff_tree_pool, tree_idx);

    int idx_nextPower = int(log2(nextPower));
    Ciphertext* xpow = eval_sine_poly_pool[idx_nextPower];

    int level = xpow->l - 1;

    if (poly->maxDegree >= 1<<(logDegree-1)) {
		level++;
	}

    uint64_tt current_qi = context.qVec[level];

    evalRecurse(target_scale*current_qi/xpow->scale, logSplit, logDegree, tree_idx*2);
    evalRecurse(target_scale, logSplit, logDegree, tree_idx*2+1);

    scheme.multAndEqual_23(*chebyshev_tree_pool[tree_idx*2], *xpow);
    scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx*2], target_scale);

    scheme.add(*chebyshev_tree_pool[tree_idx], *chebyshev_tree_pool[tree_idx*2], *chebyshev_tree_pool[tree_idx*2+1]);

    // printf("cipher[%d] <- cipher[%d]*xpow[%d] + cipher[%d]\n", tree_idx, tree_idx*2, idx_nextPower, tree_idx*2+1);
}

void SchemeAlgo::evalFromPowerBasis(NTL::RR target_scale, int tree_idx)
{
    Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];
    int mini = 0;
    for(int i = poly->degree(); i > 0; i--)
    {
        if(isNotNegligible(poly->coeffs[i]))
            mini = max(i, mini);
    }

    double c1 = poly->coeffs[0];

    int idx_mini = int(log2(mini));

    uint64_tt currentQi = context.qVec[eval_sine_poly_pool[idx_mini]->l];
    NTL::RR ctScale = target_scale * currentQi;

    cudaMemset(chebyshev_tree_pool[tree_idx], 0, sizeof(uint64_tt) * N * (maxLevel + 1) * 2);
    chebyshev_tree_pool[tree_idx]->l = eval_sine_poly_pool[idx_mini]->l;
    chebyshev_tree_pool[tree_idx]->scale = ctScale;

    double c2 = poly->coeffs[1];
        // cout<<"key: "<<key<<"  ";

    if(isNotNegligible(c2))
    {
        NTL::RR constScale = target_scale * currentQi / eval_sine_poly_pool[0]->scale;

        scheme.multConstAndAddCipherEqual(*chebyshev_tree_pool[tree_idx], *(eval_sine_poly_pool[0]), c2, constScale);
    }
    // cout<<endl;

    if(isNotNegligible(c1))
    {
        scheme.addConstAndEqual(*chebyshev_tree_pool[tree_idx], c1);
    }


    scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx], target_scale);

    // printf("cipher[%d] <- c2*xpow[%d] + c1\n", tree_idx, 0);

    return;
}

void SchemeAlgo::evalIteration(NTL::RR target_scale, int logDegree)
{
    int evalSineDegree = chebyshev_poly_coeff_tree_pool.size();
    uint64_tt currentQi = context.qVec[eval_sine_poly_pool[0]->l];
    NTL::RR ctScale = target_scale * currentQi;
    NTL::RR constScale = ctScale / eval_sine_poly_pool[0]->scale;

    for(int tree_idx = 31; tree_idx >= 16; tree_idx--)
    {
        Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];

        double c1 = poly->coeffs[0];
        double c2 = poly->coeffs[1];

        uint64_tt currentQi = context.qVec[eval_sine_poly_pool[0]->l];

        // cudaMemset(chebyshev_tree_pool[tree_idx], 0, sizeof(uint64_tt) * N * (maxLevel + 1) * 2);
        chebyshev_tree_pool[tree_idx]->l = eval_sine_poly_pool[0]->l;
        chebyshev_tree_pool[tree_idx]->scale = ctScale;

        scheme.multConstAndAddCipherEqual(*chebyshev_tree_pool[tree_idx], *(eval_sine_poly_pool[0]), c2, constScale);
        scheme.addConstAndEqual(*chebyshev_tree_pool[tree_idx], c1);

        scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx], target_scale);
    }

    // cout<<(evalSineDegree>>1) - 1<<" "<<1<<endl;
    for(int tree_idx = 15; tree_idx >= 1; tree_idx--)
    {
        int idx_nextPower = 4 - int(log2(tree_idx));
        Ciphertext* xpow = eval_sine_poly_pool[idx_nextPower];

        Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];

        scheme.multAndEqual_23(*chebyshev_tree_pool[tree_idx*2], *xpow);
        scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx*2], target_scale);

        scheme.add(*chebyshev_tree_pool[tree_idx], *chebyshev_tree_pool[tree_idx*2], *chebyshev_tree_pool[tree_idx*2+1]);
        // printf("cipher[%d] <- cipher[%d]*xpow[%d] + cipher[%d]\n", tree_idx, tree_idx*2, idx_nextPower, tree_idx*2+1);
    }
}

#define const_layer_block 256
__global__
__launch_bounds__(
    const_layer_block, 
    POLY_MIN_BLOCKS)
void cipher_cipher_mul_const_add_const_batch_kernel(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* add_const_real_buffer, uint32_tt n, int q_num, int idx_mod, int batch_size)
{
    register uint32_tt idx_in_pq = blockIdx.y;
    register int idx_in_poly = blockIdx.x * const_layer_block + threadIdx.x;

    register uint64_tt q = pqt_cons[idx_mod + idx_in_pq];
    register uint64_tt ra;
    register uint64_tt rb0 = device_b[(idx_in_pq + 0*q_num) * n + idx_in_poly];
    register uint64_tt rb1 = device_b[(idx_in_pq + 1*q_num) * n + idx_in_poly];
    register uint64_tt rc, rc_shoup;

#pragma unroll
    for(int i = 16; i < 32; i++)
    {
        rc = add_const_real_buffer[idx_in_pq + i*q_num*4];
        rc_shoup = add_const_real_buffer[idx_in_pq + q_num + i*q_num*4];

        ra = mulMod_shoup(rb0, rc, rc_shoup, q);
        device_a[(idx_in_pq + 0*q_num) * n + idx_in_poly + i*n*q_num*2] = ra;

        ra = mulMod_shoup(rb1, rc, rc_shoup, q) + add_const_real_buffer[idx_in_pq + q_num*2 + i*q_num*4];
        csub_q(ra, q);
        device_a[(idx_in_pq + 1*q_num) * n + idx_in_poly + i*n*q_num*2] = ra;
    }
}

void SchemeAlgo::evalIterationBatch(NTL::RR target_scale, int logDegree)
{
    int evalSineDegree = chebyshev_poly_coeff_tree_pool.size();
    uint64_tt currentQi = context.qVec[eval_sine_poly_pool[0]->l];
    NTL::RR ctScale = target_scale * currentQi;
    NTL::RR constScale = ctScale / eval_sine_poly_pool[0]->scale;

    int t_num = context.t_num;
    int Qj_blockNum = context.Qj_blockNum;
    int Ri_blockNum = context.Ri_blockNum;

    int cipher_min_level = maxLevel;

/********************************************const layer********************************************/
#pragma unroll
    for(int tree_idx = evalSineDegree>>1; tree_idx < evalSineDegree; tree_idx++)
    {
        Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];

        double c1 = poly->coeffs[0];
        double c2 = poly->coeffs[1];

        Ciphertext *cipher1 = chebyshev_tree_pool[tree_idx], *cipher2 = eval_sine_poly_pool[0];

        uint64_tt currentQi = context.qVec[cipher2->l];

        cudaMemset(chebyshev_tree_pool[tree_idx], 0, sizeof(uint64_tt) * N * (maxLevel + 1) * 2);
        cipher1->l = cipher2->l;
        cipher1->scale = ctScale;

        cipher1->l = min(cipher1->l, cipher2->l);
        cipher_min_level = cipher1->l;
        // cout<<"cipher_min_level: "<<cipher_min_level<<endl;

        NTL::ZZ scaled_c2 = to_ZZ(target_scale * c2);
        NTL::ZZ scaled_c1 = to_ZZ(cipher1->scale * c1);
        for(int i = 0; i < cipher_min_level+1; i++)
        {
            add_const_copy_vec[i + tree_idx * q_num*4] = scaled_c2 % context.qVec[i];
            add_const_copy_vec[i + tree_idx * q_num*4 + q_num] = x_Shoup(add_const_copy_vec[i + tree_idx * q_num*4], context.qVec[i]);
            add_const_copy_vec[i + tree_idx * q_num*4 + q_num*2] = scaled_c1 % context.qVec[i];
        }
    }
    cudaMemcpy(add_const_buffer, add_const_copy_vec.data(), sizeof(uint64_tt) * add_const_copy_vec.size(), cudaMemcpyHostToDevice);

    dim3 mul_dim(N / const_layer_block, cipher_min_level+1);
    cipher_cipher_mul_const_add_const_batch_kernel <<< mul_dim, const_layer_block >>> (chebyshev_tree_cipher_pool, eval_sine_poly_pool[0]->cipher_device, add_const_buffer, N, q_num, context.K, evalSineDegree>>1);

/********************************************const layer rescale********************************************/
#pragma unroll
    for(int tree_idx = evalSineDegree>>1; tree_idx < evalSineDegree; tree_idx++) scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx], target_scale);

#pragma unroll
    for(int tree_level = int(log2(evalSineDegree)-2); tree_level >= 1; tree_level--)
    {
        int idx_start = (1<<tree_level), idx_end = (1<<(tree_level+1));
        // printf("idx_start: %d   idx_end: %d\n", idx_start, idx_end);
        for(int tree_idx = idx_start; tree_idx < idx_end; tree_idx++)
        {
            Ciphertext* cipher2 = eval_sine_poly_pool[4 - tree_level];
            Ciphertext* cipher1 = chebyshev_tree_pool[tree_idx*2];
            Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[tree_idx];
            // cout<<"cipher1->l: "<<cipher1->l<<endl; 

            // scheme.multAndEqual_23(*chebyshev_tree_pool[tree_idx*2], *cipher2);
            scheme.multAndEqual_beforeIP_23(*cipher1, *cipher2,
                                            IP_input_temp  + (tree_idx-idx_start)*N*t_num*Qj_blockNum, 
                                            axbx1_mul + (tree_idx-idx_start)*N*q_num,
                                            bxbx_mul + (tree_idx-idx_start)*N*q_num);
        }
        context.external_product_T_swk_reuse(IP_output_temp, 
                                            IP_input_temp, 
                                            scheme.rlk_23->cipher_device, chebyshev_tree_pool[idx_end]->l, idx_start);

        for(int tree_idx = idx_start; tree_idx < idx_end; tree_idx++)
        {
            Ciphertext* cipher2 = eval_sine_poly_pool[4 - tree_level];
            Ciphertext* cipher1 = chebyshev_tree_pool[tree_idx*2];

            scheme.multAndEqual_afterIP_23(*cipher1, *cipher2,
                                           IP_output_temp + (tree_idx-idx_start)*N*t_num*Ri_blockNum*2,
                                            axbx1_mul + (tree_idx-idx_start)*N*q_num,
                                            bxbx_mul + (tree_idx-idx_start)*N*q_num);

            scheme.rescaleAndEqual(*chebyshev_tree_pool[tree_idx*2], target_scale);
            scheme.add(*chebyshev_tree_pool[tree_idx], *chebyshev_tree_pool[tree_idx*2], *chebyshev_tree_pool[tree_idx*2+1]);
        }
    }

    Ciphertext* cipher2 = eval_sine_poly_pool[4];
    Ciphertext* cipher1 = chebyshev_tree_pool[2];
    Chebyshev_Polynomial *poly = chebyshev_poly_coeff_tree_pool[1];

    scheme.multAndEqual_23(*chebyshev_tree_pool[2], *cipher2);

    scheme.rescaleAndEqual(*chebyshev_tree_pool[2], target_scale);
    scheme.add(*chebyshev_tree_pool[1], *chebyshev_tree_pool[2], *chebyshev_tree_pool[3]);
}

void SchemeAlgo::evalPolynomialChebyshev(Ciphertext &cipher, NTL::RR target_scale)
{
    eval_sine_poly_pool_computed = vector<bool>(eval_sine_poly_pool.size(), false);

    int degree = chebyshev_poly_coeff_tree_pool[1]->degree();
    int logDegree = ceil(log2(degree));
    int logSplit = (logDegree >> 1);

    *(eval_sine_poly_pool[0]) = cipher;
    eval_sine_poly_pool_computed[0] = true;

    for(int idx = 1; idx < logDegree; idx++)
    {
        // scheme.square(*(eval_sine_poly_pool[idx]), *(eval_sine_poly_pool[idx - 1]));
        // scheme.rescaleAndEqual(*(eval_sine_poly_pool[idx]), target_scale);
        // scheme.addAndEqual(*(eval_sine_poly_pool[idx]), *(eval_sine_poly_pool[idx]));
        // scheme.addConstAndEqual(*(eval_sine_poly_pool[idx]), -1);
        scheme.square_double_add_const_rescale(*(eval_sine_poly_pool[idx]), *(eval_sine_poly_pool[idx-1]), -1);

                // scheme.decryptMsg(*plain_buffer, secretkey, *(eval_sine_poly_pool[idx]));
                // context.decode(*plain_buffer, complex_vals);
                // cout<<"t:"<<idx <<"   cipher.l: "<<eval_sine_poly_pool[idx]->l;
                // print_device_array(complex_vals, slots, "");
                // // cout<<endl;

        eval_sine_poly_pool_computed[idx] = true;
    }

    // evalRecurse(target_scale, logSplit, logDegree, 1);
    // evalIteration(target_scale, logDegree);
    evalIterationBatch(target_scale, logDegree);
    // cout<<endl;

    cipher = *chebyshev_tree_pool[1];
}