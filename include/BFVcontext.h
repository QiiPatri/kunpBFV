#pragma once
 
#include <vector>
#include <sys/time.h>
#include <map>

using namespace std;

#include "Utils.cuh"
#include "Sampler.cuh"
#include "RNG.cuh"
#include "poly_arithmetic.cuh"
#include "Plaintext.cuh"
#include "ntt_60bit.cuh"
#include "BasisConv_QR.cuh"

class BFVContext{
    public:
    BFVContext(size_t poly_modulus_degree_);

    void preComputeIndex();

	void getPrimeBFV();
	void preComputeOnCPU();
	void copyMemoryToGPU();

	__host__ void rescaleAndEqual(uint64_tt* device_a, int l);

	__host__ void encode(uint64_tt* vals, Plaintext& msg);
	__host__ void encode_ntt(uint64_tt* vals, Plaintext& msg);
	__host__ void encode2(uint64_tt* vals, Plaintext& msg);
	__host__ void decode(Plaintext& msg, uint64_tt* vals);

	//old_ntt
	__host__ void forwardNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num);
	__host__ void inverseNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num);
	//new_ntt
	__host__ void FromNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int num);
	__host__ void FromNTTInplace_for_externalProduct(uint64_tt* device_a, int start_mod_idx, int cipher_mod_len, int poly_mod_num, int block_mod_num, int poly_mod_len, int cipher_mod_num, int batch_size);
	__host__ void ToNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num,int num);
	__host__ void ToNTTInplace_for_externalProduct(uint64_tt* device_a, int start_mod_idx, int cipher_mod_len, int poly_mod_num, int block_mod_num, int poly_mod_len, int cipher_mod_num, int batch_size);
	//bfv_encode_ntt
	__host__ void FromNTTInplace_for_BFV(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len);
	__host__ void ToNTTInplace_for_BFV(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len);

	//bfv_mult_ntt
	__host__ void FromNTTInplace_for_QR(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len);
	__host__ void ToNTTInplace_for_QR(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len);

	__host__ void divByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num);
	__host__ void mulByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num);
	__host__ void poly_add_complex_const_batch_device(uint64_tt* device_a, uint64_tt* add_const_buffer, int idx_a, int idx_mod, int mod_num);
	__host__ void poly_mul_const_batch_device(uint64_tt* device_a, uint64_tt* const_real, int idx_mod, int mod_num);
	__host__ void poly_mul_const_add_cipher_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* const_real, int idx_mod, int mod_num);

	// for ckks-23 fast-ksw decomposition
	__host__ void modUpPQtoT_23(uint64_tt* output, uint64_tt* input, int l, int batch_size);
	__host__ void modUpQjtoT_23(uint64_tt* output, uint64_tt* input, int l, int batch_size);
	// __host__ void modDownTtoRi_23(uint64_tt* output, uint64_tt* modUp_QjtoT_temp, uint64_tt* exProduct_T_temp, int l);
	__host__ void modUpTtoPQl_23(uint64_tt* modUp_TtoQj_buffer, uint64_tt* exProduct_T_temp, int l, int batch_size);
	__host__ void modDownPQltoQl_23(uint64_tt* output, uint64_tt* modUp_TtoQj_buffer, int l, int batch_size);

	// for ckks-23 fast-ksw external product
	__host__ void external_product_T(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT, int l);
	__host__ void external_product_T_swk_reuse(uint64_tt* output, uint64_tt* cipher_modUp_QjtoT, uint64_tt* swk_modUp_RitoT, int l, int batch_size);
	__host__ void mult_PlaintextT(uint64_tt* cipher_modUp_QjtoT, PlaintextT& plain_T, int l);
	__host__ void mult_PlaintextT_permuted(uint64_tt* cipher_modUp_QjtoT, PlaintextT& plain_T, int rotSlots, int l);

	void hps_decrypt_scale_and_round(uint64_tt *dst, uint64_tt *src, const cudaStream_t &stream);

	// vector<uint64_tt> get_primes_below(size_t ntt_size, uint64_tt upper_bound, size_t count);

	// Encryption parameters
	int logN; ///< Logarithm of Ring Dimension
	int logNh; ///< Logarithm of Ring Dimension - 1
	int logslots;
	int slots;
	int L; ///< Maximum Level that we want to support
	int q_num; ///< num of q (usually L + 1)
	int K; ///< The number of special modulus (usually (L + 1) / dnum)
	int dnum;
	int alpha;

	int p_num;
	int t_num;
	int r_num;
	int mod_num;
	int gamma; // gamma stands for tilde_r
	int Ri_blockNum; // blockNum = ceil((p_num + q_num) / gamma)
	int Qj_blockNum;

	long N;
	long M;
	long Nh;

	long logp; 
	long precision;

	long h;
	double sigma;

	uint64_tt plain_modulus;

	vector<uint64_tt> qVec;
	vector<uint64_tt> pVec;
	vector<uint64_tt> tVec;
	vector<uint64_tt> rVec;

	vector<uint128_tt> qMuVec; // Barrett reduction
	vector<uint128_tt> pMuVec; // Barrett reduction
	vector<uint128_tt> tMuVec;
	vector<uint128_tt> rMuVec;

	vector<uint64_tt> qPsi; // psi q
	vector<uint64_tt> pPsi; // psi p
	vector<uint64_tt> tPsi;	// psi t
	vector<uint64_tt> rPsi; // psi q
	vector<uint64_tt> qPsiInv; // inv psi q
	vector<uint64_tt> pPsiInv; // inv psi p
	vector<uint64_tt> tPsiInv; // inv psi t
	vector<uint64_tt> rPsiInv; // inv psi r

	vector<uint64_tt> pqtVec; // pqt
	vector<uint64_tt> pqt2Vec;
	vector<uint128_tt> pqtMuVec; // pqt Barrett reduction
	vector<uint64_tt> pqtMuVec_high;
	vector<uint64_tt> pqtMuVec_low;

	vector<uint64_tt> pqtPsi; // pqt Psi
	vector<uint64_tt> pqtPsiInv; // pqt Psi inv

	//BFV mult
	vector<uint64_tt> qrVec; // qr
	vector<uint64_tt> qrMuVec; // qr Barrett reduction
	vector<uint64_tt> qrMuVec_high;
	vector<uint64_tt> qrMuVec_low;
	vector<uint64_tt> qrPsi; // qr Psi
	vector<uint64_tt> qrPsiInv; // qr Psi inv

	// psi powers qr
	uint64_tt* qrPsiTable_device;
	// inv psi powers qr
	uint64_tt* qrPsiInvTable_device;

	// psi powers pq
	uint64_tt* pqtPsiTable_device; // ok
	// inv psi powers pq
	uint64_tt* pqtPsiInvTable_device;  // ok

	/*******************************************BFV-ENCODE-NTT*********************************************/
	uint64_tt plainModPsi;
	uint64_tt plainModPsiInv;


	// vector<double> eval_sine_chebyshev_coeff;

	/************************************base convert from PQl to Ql****************************************/
		// P/pk					
		// [P/p0 P/p1 ... P/pk] mod qi
		// P/pk mod qi
		// size = (L + 1) * K
		// ok
		vector<vector<uint64_tt>> pHatVecModq_23;
		uint64_tt* pHatVecModq_23_device;

		// pk/P
		// inv[p012...k/p0] inv[p012...k/p1] ... inv[p012...k/pk]
		// pk/P mod pk
		// size = K
		// ok
		vector<uint64_tt> pHatInvVecModp_23;
		vector<uint64_tt> pHatInvVecModp_23_shoup;
		uint64_tt* pHatInvVecModp_23_device;

	/************************************base convert from Ri to T******************************************/
		// r_ij/Ri mod r_ij
		// {inv[R0/r_00] inv[R0/r_01] ... inv[R0/r_{0,gamma-1}]} ... {inv[R_{blockNum-1}/r_{blockNum-1,gamma-1}]}
		// {r_00/R0 mod r_00 ... r_{gamma-1}0/R0 mod r_{gamma-1}0} ...... {r_0{blockNum-1}/R_{blockNum-1} mod r_0{blockNum-1} ... r_{blockNum-1}{gamma-1}/R0 mod r_{blockNum-1}{blockNum-1}}
		// size = gamma * blockNum
		// ok
		vector<vector<uint64_tt>> RiHatInvVecModRi_23;
		vector<vector<uint64_tt>> RiHatInvVecModRi_23_shoup;
		// gamma * blockNum
		uint64_tt* 	RiHatInvVecModRi_23_device;
		uint64_tt* 	RiHatInvVecModRi_23_shoup_device;

		// Ri/r_ij mod t_k
		// {[R0/r_00] [R0/r_01] ... [R0/r_{0,gamma-1}]} ... {[R_{blockNum-1}/r_{blockNum-1,gamma-1}]}
		// {r_00/R0 mod r_00 ... r_{gamma-1}0/R0 mod t_k} ...... {r_0{blockNum-1}/R_{blockNum-1} mod r_0{blockNum-1} ... r_{blockNum-1}{gamma-1}/R0 mod r_{blockNum-1}{blockNum-1}}
		// size = gamma * t_num * blockNum
		// ok
		vector<vector<vector<uint64_tt>>> RiHatVecModT_23;
		// gamma * t_num * blockNum
		uint64_tt* RiHatVecModT_23_device;
		// Ri mod ti
		// {R0 ... R_{blockNum-1}} mod t0 ... {R0 ... R_{blockNum-1}} mod t_{t_num-1}
		vector<vector<uint64_tt>> Rimodti;


	/************************************base convert from Qj to T******************************************/
		// q_ij/Qi mod q_ij
		// {inv[Q0/q_00] inv[Q0/q_01] ... inv[Q0/r_{0,gamma-1}]} ... {inv[Q_{blockNum-1}/q_{blockNum-1,gamma-1}]}
		// 
		// size = p_num * blockNum * (L+1)
		// 
		vector<vector<vector<uint64_tt>>> QjHatInvVecModQj_23;
		vector<vector<vector<uint64_tt>>> QjHatInvVecModQj_23_shoup;
		uint64_tt* QjHatInvVecModQj_23_device;
		uint64_tt* QjHatInvVecModQj_23_shoup_device;

		// size = p_num * blockNum
		// 
		vector<vector<vector<vector<uint64_tt>>>> QjHatVecModT_23;
		uint64_tt* QjHatVecModT_23_device;
		// Qj mod ti
		// {Q0 ... Q_{blockNum-1}} mod t0 ... {Q0 ... Q_{blockNum-1}} mod t_{t_num-1}
		vector<vector<uint64_tt>> Qjmodti;
		uint64_tt* Qjmodti_device;

	/************************************base convert from T to Ri******************************************/
		// ti / T mod ti
		// {inv[T/t0] inv[T/t1] ... inv[T/t_{t_num-1}]}
		// size = t_num
		// ok
		vector<uint64_tt> THatInvVecModti_23;
		vector<uint64_tt> THatInvVecModti_23_shoup;
		uint64_tt* THatInvVecModti_23_device;
		uint64_tt* THatInvVecModti_23_shoup_device;

		// T / ti mod Ri
		// {[T/t0] [T/t1] ... [T/t_{t_num-1}] mod r_i}
		// size = t_num * (p_num + q_num)
		//
		vector<vector<uint64_tt>> THatVecModRi_23;
		uint64_tt* THatVecModRi_23_device;
		// T mod pqi
		// {T mod pq0 ... T mod pq_{n-1}}
		vector<uint64_tt> Tmodpqi;

	/************************************************rescale************************************************/
	// qi mod qj
	// inv[q1]mod qi inv[q2]mod qi inv[q3]mod qi inv[q4]mod qi ... inv[qL]mod qi
	// ql mod qi [l(l-1)/2 + i]
	// ok
	vector<vector<uint64_tt>> qiInvVecModql;
	vector<vector<uint64_tt>> qiInvVecModql_shoup;
	uint64_tt* qiInvVecModql_device;
	uint64_tt* qiInvVecModql_shoup_device;

	/************************************************decode*************************************************/
	vector<vector<uint64_tt>> QlInvVecModqi;
	uint64_tt* QlInvVecModqi_device;
	vector<vector<uint64_tt>> QlHatVecModt0;
	uint64_tt* QlHatVecModt0_device;

	/************************************copy to constant memory********************************************/
	vector<uint64_tt> halfTmodpqti; 	// T//2 mod pqti
	vector<uint64_tt> PModq;			// P mod q
	vector<uint64_tt> PinvModq;			// P^-1 mod q
	vector<uint64_tt> PinvModq_shoup;

	long randomArray_len;
	// random array for key gen
	// only sk pk relk
	// ! not ok
    uint8_tt* randomArray_device;
	// = randomArray_device
	uint8_tt* randomArray_sk_device;
	// = randomArray_device + N 
	uint8_tt* randomArray_pk_device;
	// = randomArray_device + N + (L + 1 + K) * N * sizeof(uint64_tt) / sizeof(uint8_tt)
	uint8_tt* randomArray_e_pk_device;
	uint8_tt* randomArray_swk_device;
	uint8_tt* randomArray_e_swk_device;

	// precomputed rotation group indexes
	uint64_tt* rotGroups_device;

	// N * (L+1)
	uint64_tt* decode_buffer;
	uint64_tt* encode_buffer;
	uint64_tt* mult_buffer;
	uint64_tt* mult_scale_buffer;
	uint64_tt* sec_buffer;

	//new_ntt_param
	vector<uint64_tt> n_inv_host;
    vector<uint64_tt> n_inv_shoup_host;
	//BFV mult
	vector<uint64_tt> n_inv_host_qr;
	vector<uint64_tt> n_inv_shoup_host_qr;
	//BFV part
	uint64_tt n_inv_host_BFV;
	uint64_tt n_inv_shoup_host_BFV;

	uint64_tt* plainModPsi_device;
	uint64_tt* plainModPsiInv_device;
	uint64_tt* plainMod_shoup_device;
	uint64_tt* plainMod_shoup_inv_device;

	uint64_tt* qr_psi_table_device;
	uint64_tt* qr_psiinv_table_device;
	uint64_tt* qr_psi_shoup_table_device;
	uint64_tt* qr_psiinv_shoup_table_device;

	uint64_tt* psi_table_device;
	uint64_tt* psiinv_table_device;
	uint64_tt* n_inv_device_bfv;
	uint64_tt* n_inv_shoup_device_bfv;

	uint64_tt* n_inv_device;
	uint64_tt* n_inv_shoup_device;

	uint64_tt* n_inv_device_qr;
	uint64_tt* n_inv_shoup_device_qr;
	
	uint64_tt* psi_shoup_table_device;
	uint64_tt* psiinv_shoup_table_device;

    // BFV enc/add/sub
    uint64_tt negQl_mod_t{}; // Ql mod t
    uint64_tt negQl_mod_t_shoup{}; // Ql mod t
    uint64_tt* tInv_mod_q; // t^(-1) mod q
    uint64_tt* tInv_mod_q_shoup; // t^(-1) mod q	

	vector<uint64_tt> tInv_mod_q_;

	// BFV dec
	size_t qMSB;
	size_t sizeQMSB;
	size_t tMSB;
	uint64_tt* t_QHatInv_mod_q_div_q_mod_t;
	uint64_tt* t_QHatInv_mod_q_div_q_mod_t_shoup;
	double* t_QHatInv_mod_q_div_q_frac;
	uint64_tt* t_QHatInv_mod_q_B_div_q_mod_t;
	uint64_tt* t_QHatInv_mod_q_B_div_q_mod_t_shoup;
	double* t_QHatInv_mod_q_B_div_q_frac;

	// BFV mult
	uint64_tt* t_QHatInv_mod_q_div_q_mod_q;
	uint64_tt* t_QHatInv_mod_q_div_q_mod_q_shoup;

	std::vector<uint64_tt> mod_;
	// product of all small modulus in this base, stored in 1d vector
	std::vector<uint64_tt> prod_mod_;
	// product of all small modulus's hat in this base, stored in 2d vector
	std::vector<uint64_tt> prod_hat_;
	// vector of qiHat mod qi
	std::vector<uint64_tt> hat_mod_;
	std::vector<uint64_tt> hat_mod_shoup_;
	// vector of qiHatInv mod qi
	std::vector<uint64_tt> hatInv_mod_;
	std::vector<uint64_tt> hatInv_mod_shoup_;
	// vector of 1.0 / qi
	std::vector<double> inv_;

	std::vector<uint64_tt> mod_r;
	std::vector<uint64_tt> prod_mod_r;
	std::vector<uint64_tt> prod_hat_r;
	std::vector<uint64_tt> hat_mod_r;
	std::vector<uint64_tt> hat_mod_shoup_r;
	std::vector<uint64_tt> hatInv_mod_r;
	std::vector<uint64_tt> hatInv_mod_shoup_r;
	std::vector<double> inv_r;

	// std::vector<uint64_tt> prod_mod_qr;
	std::vector<uint64_tt> prod_hat_qr;
	std::vector<uint64_tt> hat_mod_qr;
	std::vector<uint64_tt> hatInv_mod_qr;

	uint64_tt *QR_mod;
	uint64_tt *QR_Mu_high;
	uint64_tt *QR_Mu_low; 

	//QR to R scale & round
	uint64_tt *tRSHatInvModsDivsModr_;
	uint64_tt *tRSHatInvModsDivsModr_shoup_;
	double *tRSHatInvModsDivsFrac_;

	//BaseConverter
	BaseConverter QtoR;
	BaseConverter RtoQ;

	uint64_tt* batch_encode_index_device;

	//for check correctness of decode
	uint64_tt* encode_buffer_device;
	uint64_tt* decode_buffer_device;
	size_t buffer_idx = 0;
	size_t first_error_idx = 0;
	size_t error_idx_num = 0;
};