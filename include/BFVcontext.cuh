#pragma once

#include "BFVcontext.h"
#include "Encoder.cuh"
#include "BasisConv.cuh"
#include "ExternalProduct.cuh"
#include "uintarith_bfv.h"
#include "Params.cuh"
#include <assert.h>

using namespace std;

BFVContext::BFVContext(size_t poly_modulus_degree_)
{
	// qVec = {
	// 	0x7FFE28001, 0x7FFF18001, 0x7FFF80001,	0x7FFFB0001
	// };
	// pVec = { // 36 x 6
	// 	0xFFFFCA8001, 0xFFFFE80001
	// };
	// tVec = {
	// 	0xffffffffffc0001, 0xfffffffff840001,
	// 	0xfffffffff6a0001, 0xfffffffff5a0001
	// };

	// gamma = 2;

	N = poly_modulus_degree_;
	logN = log2(N);
	logslots = logN;
	M = N << 1;
	Nh = N >> 1;
	slots = N;
	long h = 64;
	double sigma = 3.2;

	getPrimeBFV();

	q_num = qVec.size();
	p_num = pVec.size();
	t_num = tVec.size();
	mod_num = p_num + q_num + t_num;
	L = q_num - 1;
	K = pVec.size();
	dnum = q_num / K;
	alpha = K;

	uint64_tt q_min = 0x7FFFFFFFFFFFFFFF;
	for(int i = 0; i < qVec.size(); ++i)
	{
		if(qVec[i] < q_min)
			q_min = qVec[i];
	}
	// printf( "%llu\n" , q_min );

	// ----------------------------------R_vec for BFV mult----------------------------------
	r_num = q_num + 1;
	rVec = get_primes_below(static_cast<size_t>(N), q_min, static_cast<size_t>(r_num));

	// printf("values_Rl = [ ");
    // for( size_t idx = 0 ; idx < rVec.size() ; ++idx )
    // {
    //     printf( "%llu " , rVec[idx] );
    // }    printf(" ]\n");
	//---------------------------------------------------------------------------------------

	Ri_blockNum = ceil(double(p_num + q_num) / gamma);
	Qj_blockNum = ceil(double(q_num) / p_num);

	assert(Ri_blockNum <= max_Riblock_num);
	assert(Qj_blockNum <= max_Qjblock_num);
	// cout<<"Ri_blockNum: "<<Ri_blockNum<<"  Qj_blockNum: "<<Qj_blockNum<<endl;
	assert(t_num <= max_tnum);
	// cout<<"t_num: "<<t_num<<endl;

	randomArray_len = 0;

	preComputeOnCPU();
	printf("preComputeOnCPU OK\n");
	copyMemoryToGPU();
	printf("copyMemoryToGPU OK\n");
	preComputeIndex();

	// printf("logN: %d Pnum: %d Qnum: %d Tnum: %d gamma: %d\n", logN, p_num, q_num, t_num, gamma);
    // printf("dnum: %d Ri_blockNum: %d, Qj_blockNum: %d\n", dnum, Ri_blockNum, Qj_blockNum);

	// check qr_cons
    // =============================
	uint64_tt size_QR = qrVec.size();
	uint64_tt* array_PQ = new uint64_tt[size_QR];
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(array_PQ, qr_cons, sizeof(uint64_tt) * size_QR, 0, cudaMemcpyDeviceToHost);
    // printf("QR_MOD = [" );
    // for(int i = 0; i < size_QR; i++)
    // {
    //     printf("%llu, ", array_PQ[i]);
    // }
    // printf("]\n");
    free(array_PQ);
    // =============================
}
void BFVContext::preComputeIndex()
{
	vector<uint64_tt> temp;
    int logn = logslots;
    // Copy from the matrix to the value vectors
    size_t row_size = slots >> 1;
    size_t m = slots << 1;
    uint64_tt gen = 5;
    uint64_tt pos = 1;
    temp.resize(slots);
	
    for (size_t i = 0; i < row_size; i++) {
        // Position in normal bit order
        uint64_tt index1 = (pos - 1) >> 1;
        uint64_tt index2 = (m - pos - 1) >> 1;

        // Set the bit-reversed locations
        temp[i] = (uint64_tt) (reverse_bits(index1, logn));
        temp[row_size | i] = static_cast<size_t>(reverse_bits(index2, logn));

        // Next primitive root
        pos *= gen;
        pos &= (m - 1);
    }
	cudaMalloc(&batch_encode_index_device, sizeof(uint64_tt) * slots);
    cudaMemcpy(batch_encode_index_device, temp.data(), sizeof(uint64_tt) * slots, cudaMemcpyHostToDevice);
	// print_device_array(batch_encode_index_device, slots, 1, "batch_encode_index_device");
}
void BFVContext::preComputeOnCPU()
{
	/****************************************[p0,p1,...,p{k-1},q0,q1,...,qL]************************************************/
	NTL::ZZ mulP(1);
	NTL::ZZ mulPQ_gamma(1);
	NTL::ZZ mulQ_alpha(1);
	NTL::ZZ mulT(1);
	{
		uint64_tt root = findMthRootOfUnity(M, plain_modulus);
		plainModPsi = root;
		plainModPsiInv = modinv128(root, plain_modulus);
	}
	for(int i = 0; i < K; i++)
	{
		pqtVec.push_back(pVec[i]);
		pqt2Vec.push_back(pVec[i]*2);
		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= pVec[i];
		pMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pqtMuVec_high.push_back(pMuVec[i].high);
		pqtMuVec_low.push_back(pMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, pVec[i]);
		pPsi.push_back(root);
		pqtPsi.push_back(root);

		mulP *= pVec[i];
	}
	for(int i = 0; i <= L; i++)
	{
		pqtVec.push_back(qVec[i]);
		pqt2Vec.push_back(qVec[i]*2);
		qrVec.push_back(qVec[i]);

		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= qVec[i];
		qMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pqtMuVec_high.push_back(qMuVec[i].high);
		qrMuVec_high.push_back(qMuVec[i].high);
		pqtMuVec_low.push_back(qMuVec[i].low);
		qrMuVec_low.push_back(qMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, qVec[i]);
		qPsi.push_back(root);
		pqtPsi.push_back(root);
		qrPsi.push_back(root);

		if(i < p_num)
			mulQ_alpha *= qVec[i];
	}
	NTL::ZZ halfT(1);
	for(int i = 0; i < t_num; i++)
	{
		pqtVec.push_back(tVec[i]);
		pqt2Vec.push_back(tVec[i]*2);
		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= tVec[i];
		tMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		pqtMuVec_high.push_back(tMuVec[i].high);
		pqtMuVec_low.push_back(tMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, tVec[i]);
		tPsi.push_back(root);
		pqtPsi.push_back(root);

		halfT *= tVec[i];
		mulT *= tVec[i];
	}
	for(int i = 0; i < r_num; i++)
	{
		qrVec.push_back(rVec[i]);

		NTL::ZZ mu(1);
		mu <<= 128;
		mu /= rVec[i];
		rMuVec.push_back({to_ulong(mu>>64), to_ulong(mu - ((mu>>64)<<64))});
		qrMuVec_high.push_back(rMuVec[i].high);
		qrMuVec_low.push_back(rMuVec[i].low);
		uint64_tt root = findMthRootOfUnity(M, rVec[i]);
		rPsi.push_back(root);
		qrPsi.push_back(root);
	}

	// printf("values_QR = [ ");
    // for( size_t idx = 0 ; idx < qrVec.size() ; ++idx )
    // {
    //     printf( "%llu " , qrVec[idx] );
    // }    printf(" ]\n");

	// for(int i = 0; i < gamma; i++) mulPQ_gamma *= pqtVec[i];

	// cout<<"mulT : "<<mulT<<endl;
	// cout<<"mulP * mulPQ_gamma * N * dnum : "<<mulP * mulPQ_gamma * N * dnum<<endl;
	assert(mulT > mulQ_alpha * mulPQ_gamma * N * dnum);
	assert(mulP > mulQ_alpha);
	// cout<<"T / (Q[:alpha] * PQ[:gamma] * N * d): "<<mulT / (mulQ_alpha*mulPQ_gamma*dnum*N)<<endl;
	// cout<<"mulP / mulQ_alpha: "<<mulP / mulQ_alpha<<endl;

	halfT /= 2;
	for(int i = 0; i < p_num + q_num + t_num; i++)
	{
		halfTmodpqti.push_back(halfT % pqtVec[i]);
	}

	/*****************************************************pq_psi_related*****************************************************/
	for (int i = 0; i < K; i++)
		pqtPsiInv.push_back(modinv128(pPsi[i], pVec[i])); // pPsiInv

    for (int i = 0; i <= L; i++)
		pqtPsiInv.push_back(modinv128(qPsi[i], qVec[i])); // qPsiInv

    for (int i = 0; i < t_num; i++)
		pqtPsiInv.push_back(modinv128(tPsi[i], tVec[i])); // tPsiInv

	/*****************************************************qr_psi_related*****************************************************/
    for (int i = 0; i <= L; i++)
		qrPsiInv.push_back(modinv128(qPsi[i], qVec[i])); // qPsiInv

    for (int i = 0; i < r_num; i++)
		qrPsiInv.push_back(modinv128(rPsi[i], rVec[i])); // tPsiInv

	/*****************************************************100x_ntt*****************************************************/
	// bfv-part
	n_inv_host_BFV = modinv128(N, plain_modulus);
	n_inv_shoup_host_BFV = x_Shoup(n_inv_host_BFV, plain_modulus);

	for (int i = 0; i < K; i++)
		n_inv_host.push_back(modinv128(N, pVec[i])); // pPsiInv

    for (int i = 0; i <= L; i++)
		n_inv_host.push_back(modinv128(N, qVec[i])); // qPsiInv

    for (int i = 0; i < t_num; i++)
		n_inv_host.push_back(modinv128(N, tVec[i])); // tPsiInv
	//=========================================================
	for (int i = 0; i <= L; i++)
		n_inv_host_qr.push_back(modinv128(N, qVec[i])); // qPsiInv
	
	for (int i = 0; i < r_num; i++)
		n_inv_host_qr.push_back(modinv128(N, rVec[i])); // rPsiInv
	//=========================================================
	for (int i = 0; i < K; i++)
		n_inv_shoup_host.push_back(x_Shoup(n_inv_host[i], pVec[i])); // pPsiInv

    for (int i = 0; i <= L; i++)
		n_inv_shoup_host.push_back(x_Shoup(n_inv_host[i+K], qVec[i])); // qPsiInv

    for (int i = 0; i < t_num; i++)
		n_inv_shoup_host.push_back(x_Shoup(n_inv_host[i+K+L+1], tVec[i])); // tPsiInv
	//=========================================================
	for (int i = 0; i <= L; i++)
		n_inv_shoup_host_qr.push_back(x_Shoup(n_inv_host_qr[i], qVec[i])); // qPsiInv
	for (int i = 0; i < r_num; i++)
		n_inv_shoup_host_qr.push_back(x_Shoup(n_inv_host_qr[i+L+1], rVec[i])); // rPsiInv
	/******************************************base convert from P x Ql to Ql************************************************/

	for(int l = 0; l <= L; l++)
	{
		pHatVecModq_23.push_back({});
		for(int i = 0; i < K; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii < K; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, pVec[ii], qVec[l]);
			}
			pHatVecModq_23[l].push_back(temp);
		}
	}

	for(int i = 0; i < K; i++)
	{
		uint64_tt temp = 1;
		for(int ii = 0; ii < K; ii++)
		{
			if(ii == i) continue;
			temp = mulMod128(temp, pVec[ii], pVec[i]);
		}
		temp = modinv128(temp, pVec[i]);
		pHatInvVecModp_23.push_back(temp);
		pHatInvVecModp_23_shoup.push_back(x_Shoup(temp, pVec[i]));
	}

	/************************************base convert from Ri to T******************************************/
	for(int i = 0; i < Ri_blockNum; i++)
	{
		RiHatInvVecModRi_23.push_back({});
		RiHatInvVecModRi_23_shoup.push_back({});
		for(int j = 0; j < gamma && i*gamma + j < p_num + q_num; j++)
		{
			uint64_tt temp = 1;
			for(int jj = 0; jj < gamma && i*gamma + jj < p_num + q_num; jj++)
			{
				if(jj == j) continue;
				temp = mulMod128(temp, pqtVec[i*gamma + jj], pqtVec[i*gamma + j]);
			}
			temp = modinv128(temp, pqtVec[i*gamma + j]);
			RiHatInvVecModRi_23[i].push_back(temp);
			RiHatInvVecModRi_23_shoup[i].push_back(x_Shoup(temp, pqtVec[i*gamma + j]));
		}
	}

	for(int i = 0; i < Ri_blockNum; i++)
	{
		RiHatVecModT_23.push_back({});
		for(int k = 0; k < t_num; k++)
		{
			uint64_tt mod = tVec[k];
			RiHatVecModT_23[i].push_back({});
			for(int j = 0; j < gamma && i*gamma + j < p_num + q_num; j++)
			{
				uint64_tt temp = 1;
				for(int jj = 0; jj < gamma && i*gamma + jj < p_num + q_num; jj++)
				{
					if(jj == j) continue;
					temp = mulMod128(temp, pqtVec[i*gamma + jj], mod);
				}
				RiHatVecModT_23[i][k].push_back(temp);
			}
		}
	}

	for(int k = 0; k < t_num; k++)
	{
		Rimodti.push_back({});
		uint64_tt mod = tVec[k];
		for(int i = 0; i < Ri_blockNum; i++)
		{
			uint64_tt temp = 1;
			for(int j = 0; j < gamma && i*gamma+j < p_num + q_num; j++)
			{
				temp = mulMod128(temp, pqtVec[i*gamma + j], mod);
			}
			Rimodti[k].push_back(temp);
		}
	}

	/************************************base convert from Qj to T******************************************/
	for(int l = 0; l <= L; l++)
	{
		QjHatInvVecModQj_23.push_back({});
		QjHatInvVecModQj_23_shoup.push_back({});
		int block_num = ceil(double(l+1) / p_num);
		for(int i = 0; i < block_num; i++)
		{
			QjHatInvVecModQj_23[l].push_back({});
			QjHatInvVecModQj_23_shoup[l].push_back({});
			for(int j = 0; j < p_num && i*p_num + j <= l; j++)
			{
				uint64_tt temp = 1;
				for(int jj = 0; jj < p_num && i*p_num + jj <= l; jj++)
				{
					if(jj == j) continue;
					temp = mulMod128(temp, qVec[i*p_num + jj], qVec[i*p_num + j]);
				}
				temp = modinv128(temp, qVec[i*p_num + j]);
				QjHatInvVecModQj_23[l][i].push_back(temp);
				QjHatInvVecModQj_23_shoup[l][i].push_back(x_Shoup(temp, qVec[i*p_num + j]));
			}
		}
	}

	for(int l = 0; l <= L; l++)
	{
		QjHatVecModT_23.push_back({});
		int block_num = ceil(double(l+1) / p_num);
		for(int i = 0; i < block_num; i++)
		{
			QjHatVecModT_23[l].push_back({});
			for(int k = 0; k < t_num; k++)
			{
				uint64_tt mod = tVec[k];
				QjHatVecModT_23[l][i].push_back({});
				for(int j = 0; j < p_num && i*p_num + j <= l; j++)
				{
					uint64_tt temp = 1;
					for(int jj = 0; jj < p_num && i*p_num + jj <= l; jj++)
					{
						if(jj == j) continue;
						temp = mulMod128(temp, qVec[i*p_num + jj], mod);
					}
					QjHatVecModT_23[l][i][k].push_back(temp);
				}
			}
		}
	}

	for(int l = 0; l <= L; l++)
	{
		Qjmodti.push_back(vector<uint64_tt>(Qj_blockNum*t_num, 0));
		for(int k = 0; k < t_num; k++)
		{
			int block_num = ceil(double(l+1) / p_num);
			for(int i = 0; i < block_num; i++)
			{
				uint64_tt temp = 1;
				for(int j = 0; j < p_num && i*p_num + j <= l; j++)
				{
					temp = mulMod128(temp, pqtVec[p_num + i*p_num + j], tVec[k]);
				}
				Qjmodti[l][i*t_num + k] = temp;
			}
		}
	}

	/************************************base convert from T to Ri******************************************/
	{
		for(int i = 0; i < t_num; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii < t_num; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, tVec[ii], tVec[i]);
			}
			temp = modinv128(temp, tVec[i]);
			THatInvVecModti_23.push_back(temp);
			THatInvVecModti_23_shoup.push_back(x_Shoup(temp, tVec[i]));
		}
	}

	for(int i = 0; i < p_num + q_num; i++)
	{
		THatVecModRi_23.push_back({});
		for(int j = 0; j < t_num; j++)
		{
			uint64_tt temp = 1;
			for(int jj = 0; jj < t_num; jj++)
			{
				if(jj == j) continue;
				temp = mulMod128(temp, tVec[jj], pqtVec[i]);
			}
			THatVecModRi_23[i].push_back(temp);
		}
	}

	for(int i = 0; i < p_num + q_num; i++)
	{
		uint64_tt temp = 1;
		for(int k = 0; k < t_num; k++)
		{
			temp = mulMod128(temp, tVec[k], pqtVec[i]);
		}
		Tmodpqi.push_back(temp);
	}

	/******************************************Fast_conv_related***************************************************/
	/*********************************************P Inv mod qi*****************************************************/
	for(int i = 0; i <= L; i++)
	{
		uint64_tt temp = 1;
		for(int j = 0; j < K; j++)
		{
			temp = mulMod128(temp, pVec[j], qVec[i]);
		}
		PModq.push_back(temp);
		temp = modinv128(temp, qVec[i]);
		PinvModq.push_back(temp);
		PinvModq_shoup.push_back(x_Shoup(temp, qVec[i]));
	}

	/*********************************************Rescale_related***************************************************/
	/**********************************************ql Inv mod qi****************************************************/
	for(int l = 1; l <= L; l++)
	{
		qiInvVecModql.push_back({});
		qiInvVecModql_shoup.push_back({});
		for(int j = 0; j < l; j++)
		{
			uint64_tt qj_inv_mod_qi = modinv128(qVec[l], qVec[j]);
			qiInvVecModql.back().push_back(qj_inv_mod_qi);
			qiInvVecModql_shoup.back().push_back(x_Shoup(qj_inv_mod_qi, qVec[j]));
		}
	}

	/************************************************decode*********************************************************/
	/**********************************************ql Inv mod qi****************************************************/
	for(int l = 0; l <= L; l++)
	{
		QlInvVecModqi.push_back({});
		for(int i = 0; i <= l; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii <= l; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, qVec[ii], qVec[i]);
			}
			temp = modinv128(temp, qVec[i]);
			QlInvVecModqi[l].push_back(temp);
		}
	}

	for(int l = 0; l <= L; l++)
	{
		QlHatVecModt0.push_back({});
		for(int i = 0; i <= l; i++)
		{
			uint64_tt temp = 1;
			for(int ii = 0; ii <= l; ii++)
			{
				if(ii == i) continue;
				temp = mulMod128(temp, qVec[ii], plain_modulus);
			}
			QlHatVecModt0[l].push_back(temp);
		}
	}

	{
		
	}
}

/**************************************memory malloc & copy on GPU**********************************************/
void BFVContext::copyMemoryToGPU()
{
    //pqPsiTable and pqPsiInvTable
    uint64_tt** pqtPsiTable = new uint64_tt*[(K+L+1+t_num)];
	uint64_tt** pqtPsiInvTable = new uint64_tt*[(K+L+1+t_num)];
	//BFV_part
	uint64_tt* plainModPsiTable = new uint64_tt[N];
	uint64_tt* plainModPsiInvTable = new uint64_tt[N];
	uint64_tt* plainMod_shoup_table = new uint64_tt[N];
	uint64_tt* plainMod_shoup_inv_table = new uint64_tt[N];
	//BFV_mult
	uint64_tt** qrPsiTable = new uint64_tt*[L+1+r_num];
	uint64_tt** qrPsiInvTable = new uint64_tt*[L+1+r_num];

	{
		fillTablePsi128_special(plainModPsi, plain_modulus, plainModPsiInv, plainModPsiTable, plainModPsiInvTable, N, n_inv_host_BFV);
		fillTablePsi_shoup128(plainModPsiTable, plain_modulus, plainModPsiInvTable, plainMod_shoup_table, plainMod_shoup_inv_table, N);
	}
	/*******************************************100x_NTT******************************************************/
	for (int i = 0; i < (K+L+1+t_num); i++)
	{
		pqtPsiTable[i] = new uint64_tt[N];
		pqtPsiInvTable[i] = new uint64_tt[N];
        fillTablePsi128_special(pqtPsi[i], pqtVec[i], pqtPsiInv[i], pqtPsiTable[i], pqtPsiInvTable[i], N, n_inv_host[i]);
    }

	uint64_tt** psi_shoup_table = new uint64_tt*[(K+L+1+t_num)];
    uint64_tt** psiinv_shoup_table = new uint64_tt*[(K+L+1+t_num)];
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		psi_shoup_table[i] = new uint64_tt[N];
		psiinv_shoup_table[i] = new uint64_tt[N];
        fillTablePsi_shoup128(pqtPsiTable[i], pqtVec[i], pqtPsiInvTable[i], psi_shoup_table[i], psiinv_shoup_table[i], N);
    }

	//qr for BFV mult
	for (int i = 0 ; i < (L+1+r_num); i++)
	{
		qrPsiTable[i] = new uint64_tt[N];
		qrPsiInvTable[i] = new uint64_tt[N];
		fillTablePsi128_special(qrPsi[i], qrVec[i], qrPsiInv[i], qrPsiTable[i], qrPsiInvTable[i], N, n_inv_host_qr[i]);
	}

	uint64_tt** qr_psi_shoup_table = new uint64_tt*[L+1+r_num];
	uint64_tt** qr_psiinv_shoup_table = new uint64_tt*[L+1+r_num];
	for (int i = 0; i < (L+1+r_num); i++)
	{
		qr_psi_shoup_table[i] = new uint64_tt[N];
		qr_psiinv_shoup_table[i] = new uint64_tt[N];
		fillTablePsi_shoup128(qrPsiTable[i], qrVec[i], qrPsiInvTable[i], qr_psi_shoup_table[i], qr_psiinv_shoup_table[i], N);
	}

	//BFV_part_malloc
	cudaMalloc(&plainModPsi_device, sizeof(uint64_tt) * N);
	cudaMalloc(&plainModPsiInv_device, sizeof(uint64_tt) * N);
	cudaMalloc(&plainMod_shoup_device, sizeof(uint64_tt) * N);
	cudaMalloc(&plainMod_shoup_inv_device, sizeof(uint64_tt) * N);
	cudaMemcpy(plainModPsi_device, plainModPsiTable, sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(plainModPsiInv_device, plainModPsiInvTable, sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(plainMod_shoup_device, plainMod_shoup_table, sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(plainMod_shoup_inv_device, plainMod_shoup_inv_table, sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
	delete plainModPsiTable;
	delete plainModPsiInvTable;
	delete plainMod_shoup_table;
	delete plainMod_shoup_inv_table;

	cudaMalloc(&psi_table_device, sizeof(uint64_tt) * N * (K+L+1+t_num));
	cudaMalloc(&psiinv_table_device, sizeof(uint64_tt) * N * (K+L+1+t_num));
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		cudaMemcpy(psi_table_device + i * N, pqtPsiTable[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(psiinv_table_device + i * N, pqtPsiInvTable[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);

		delete pqtPsiTable[i];
		delete pqtPsiInvTable[i];
	}
	delete pqtPsiTable;
	delete pqtPsiInvTable;

	cudaMalloc(&psi_shoup_table_device, sizeof(uint64_tt) *  N * (K+L+1+t_num));
	cudaMalloc(&psiinv_shoup_table_device, sizeof(uint64_tt) * N * (K+L+1+t_num));
    for (int i = 0; i < (K+L+1+t_num); i++)
	{
		cudaMemcpy(psi_shoup_table_device + i * N, psi_shoup_table[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(psiinv_shoup_table_device + i * N, psiinv_shoup_table[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		delete psi_shoup_table[i];
		delete psiinv_shoup_table[i];
	}
	delete psi_shoup_table;
	delete psiinv_shoup_table;

	//=========================QR malloc==========================
	cudaMalloc(&qr_psi_table_device, sizeof(uint64_tt) * N * (L+1+r_num));
	cudaMalloc(&qr_psiinv_table_device, sizeof(uint64_tt) * N * (L+1+r_num));
    for (int i = 0; i < (L+1+r_num); i++)
	{
		cudaMemcpy(qr_psi_table_device + i * N, qrPsiTable[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(qr_psiinv_table_device + i * N, qrPsiInvTable[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);

		delete qrPsiTable[i];
		delete qrPsiInvTable[i];
	}
	delete qrPsiTable;
	delete qrPsiInvTable;

	cudaMalloc(&qr_psi_shoup_table_device, sizeof(uint64_tt) *  N * (L+1+r_num));
	cudaMalloc(&qr_psiinv_shoup_table_device, sizeof(uint64_tt) * N * (L+1+r_num));
    for (int i = 0; i < (L+1+r_num); i++)
	{
		cudaMemcpy(qr_psi_shoup_table_device + i * N, qr_psi_shoup_table[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(qr_psiinv_shoup_table_device + i * N, qr_psiinv_shoup_table[i], sizeof(uint64_tt) * N, cudaMemcpyHostToDevice);
		delete qr_psi_shoup_table[i];
		delete qr_psiinv_shoup_table[i];
	}
	delete qr_psi_shoup_table;
	delete qr_psiinv_shoup_table;
	//============================================================

	cudaMalloc(&n_inv_device, sizeof(uint64_tt)  * (K+L+1+t_num));
	cudaMalloc(&n_inv_shoup_device, sizeof(uint64_tt)  * (K+L+1+t_num));

	cudaMemcpy(n_inv_device, n_inv_host.data(), sizeof(uint64_tt) * (K+L+1+t_num), cudaMemcpyHostToDevice);
	cudaMemcpy(n_inv_shoup_device, n_inv_shoup_host.data(), sizeof(uint64_tt) * (K+L+1+t_num), cudaMemcpyHostToDevice);
	
	cudaMalloc(&n_inv_device_bfv, sizeof(uint64_tt));
	cudaMalloc(&n_inv_shoup_device_bfv, sizeof(uint64_tt));
	cudaMemcpy(n_inv_device_bfv, &n_inv_host_BFV, sizeof(uint64_tt), cudaMemcpyHostToDevice);
	cudaMemcpy(n_inv_shoup_device_bfv, &n_inv_shoup_host_BFV, sizeof(uint64_tt), cudaMemcpyHostToDevice);	

	cudaMalloc(&n_inv_device_qr, sizeof(uint64_tt)  * (L+1+r_num));
	cudaMalloc(&n_inv_shoup_device_qr, sizeof(uint64_tt)  * (L+1+r_num));
	cudaMemcpy(n_inv_device_qr, n_inv_host_qr.data(), sizeof(uint64_tt) * (L+1+r_num), cudaMemcpyHostToDevice);
	cudaMemcpy(n_inv_shoup_device_qr, n_inv_shoup_host_qr.data(), sizeof(uint64_tt) * (L+1+r_num), cudaMemcpyHostToDevice);	
	
	/************************************base convert from PQl to Ql****************************************/
	// P/pk					
	// [P/p0 P/p1 ... P/pk] mod qi
	// P/pk mod qi
	// size = (L + 1) * K
	// ok
	cudaMalloc(&pHatVecModq_23_device, sizeof(uint64_tt) * K*(L+1));
	for(int l = 0; l <= L; l++)
	{
		cudaMemcpy(pHatVecModq_23_device + l*K, pHatVecModq_23[l].data(), sizeof(uint64_tt) * pHatVecModq_23[l].size(), cudaMemcpyHostToDevice);		
	}

	// // pk/P
	// // inv[p012...k/p0] inv[p012...k/p1] ... inv[p012...k/pk]
	// // pk/P mod pk
	// // size = K
	// // ok
	// cudaMalloc(&pHatInvVecModp_23_device, sizeof(uint64_tt) * K);
	// cudaMemcpy(pHatInvVecModp_23_device, pHatInvVecModp_23.data(), sizeof(uint64_tt) * pHatInvVecModp_23.size(), cudaMemcpyHostToDevice);
	

	// qi mod qj
	// inv[q1]mod qi inv[q2]mod qi inv[q3]mod qi inv[q4]mod qi ... inv[qL]mod qi
	// ql mod qi [l(l-1)/2 + i]
	// size = L*(L+1)/2
	cudaMalloc(&qiInvVecModql_device, sizeof(uint64_tt) * L * (L + 1) / 2);
	cudaMalloc(&qiInvVecModql_shoup_device, sizeof(uint64_tt) * L * (L + 1) / 2);
	for(int l = 0; l < L; l++)
	{
		cudaMemcpy(qiInvVecModql_device + (l+1)*l/2, qiInvVecModql[l].data(), sizeof(uint64_tt) * qiInvVecModql[l].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(qiInvVecModql_shoup_device + (l+1)*l/2, qiInvVecModql_shoup[l].data(), sizeof(uint64_tt) * qiInvVecModql_shoup[l].size(), cudaMemcpyHostToDevice);
	}

	/************************************base convert from Ri to T******************************************/
	cudaMalloc(&RiHatInvVecModRi_23_device, sizeof(uint64_tt) * gamma * Ri_blockNum);
	cudaMalloc(&RiHatInvVecModRi_23_shoup_device, sizeof(uint64_tt) * gamma * Ri_blockNum);
	for(int i = 0; i < Ri_blockNum; i++)
	{
		cudaMemcpy(RiHatInvVecModRi_23_device + i * gamma, RiHatInvVecModRi_23[i].data(), sizeof(uint64_tt) * RiHatInvVecModRi_23[i].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(RiHatInvVecModRi_23_shoup_device + i * gamma, RiHatInvVecModRi_23_shoup[i].data(), sizeof(uint64_tt) * RiHatInvVecModRi_23_shoup[i].size(), cudaMemcpyHostToDevice);
	}

	cudaMalloc(&RiHatVecModT_23_device, sizeof(uint64_tt) * gamma * t_num * Ri_blockNum);
	for(int i = 0; i < Ri_blockNum; i++)
	{
		for(int j = 0; j < t_num; j++)
		{
			cudaMemcpy(RiHatVecModT_23_device + i * gamma * t_num + j * gamma,
			RiHatVecModT_23[i][j].data(), sizeof(uint64_tt) * RiHatVecModT_23[i][j].size(), cudaMemcpyHostToDevice);
		}
	}

	/************************************base convert from Qj to T******************************************/
	cudaMalloc(&QjHatInvVecModQj_23_device, sizeof(uint64_tt) * q_num * p_num * Qj_blockNum);
	cudaMalloc(&QjHatInvVecModQj_23_shoup_device, sizeof(uint64_tt) * q_num * p_num * Qj_blockNum);
	for(int l = 0; l <= L; l++)
	{
		for(int i = 0; i < QjHatInvVecModQj_23[l].size(); i++)
		{
			cudaMemcpy(QjHatInvVecModQj_23_device + l*p_num*Qj_blockNum + i*p_num,
			QjHatInvVecModQj_23[l][i].data(), sizeof(uint64_tt) * QjHatInvVecModQj_23[l][i].size(), cudaMemcpyHostToDevice);
			cudaMemcpy(QjHatInvVecModQj_23_shoup_device + l*p_num*Qj_blockNum + i*p_num,
			QjHatInvVecModQj_23_shoup[l][i].data(), sizeof(uint64_tt) * QjHatInvVecModQj_23_shoup[l][i].size(), cudaMemcpyHostToDevice);
		}
	}

	cudaMalloc(&QjHatVecModT_23_device, sizeof(uint64_tt) * q_num * p_num * t_num * Qj_blockNum);
	for(int l = 0; l <= L; l++)
	{
		for(int i = 0; i < QjHatVecModT_23[l].size(); i++)
		{
			for(int j = 0; j < QjHatVecModT_23[l][i].size(); j++)
			{
				cudaMemcpy(QjHatVecModT_23_device + l*p_num*t_num*Qj_blockNum + i*p_num*t_num + j*p_num,
				QjHatVecModT_23[l][i][j].data(), sizeof(uint64_tt) * QjHatVecModT_23[l][i][j].size(), cudaMemcpyHostToDevice);
			}
		}
	}

	cudaMalloc(&Qjmodti_device, sizeof(uint64_tt) * q_num * t_num * Qj_blockNum);
	for(int l = 0; l <= L; l++)
	{
		cudaMemcpy(Qjmodti_device + l*t_num*Qj_blockNum, Qjmodti[l].data(), sizeof(uint64_tt) * Qjmodti[l].size(), cudaMemcpyHostToDevice);
	}

	/************************************base convert from T to Ri******************************************/
	cudaMalloc(&THatInvVecModti_23_device, sizeof(uint64_tt) * t_num);
	cudaMemcpy(THatInvVecModti_23_device, THatInvVecModti_23.data(), sizeof(uint64_tt) * THatInvVecModti_23.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&THatInvVecModti_23_shoup_device, sizeof(uint64_tt) * t_num);
	cudaMemcpy(THatInvVecModti_23_shoup_device, THatInvVecModti_23_shoup.data(), sizeof(uint64_tt) * THatInvVecModti_23_shoup.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&THatVecModRi_23_device, sizeof(uint64_tt) * t_num * (K+L+1));
	for(int i = 0; i < K+L+1; i++)
	{
		cudaMemcpy(THatVecModRi_23_device + i * t_num, THatVecModRi_23[i].data(), sizeof(uint64_tt) * t_num, cudaMemcpyHostToDevice);
	}

	/**********************************************BaseConv decode Ql to T0**************************************************/
	cudaMalloc(&QlInvVecModqi_device, sizeof(uint64_tt) * (L+1)*(L+1));
	for(int i = 0; i <= L; i++)
	{
		cudaMemcpy(QlInvVecModqi_device + i * L, QlInvVecModqi[i].data(), sizeof(uint64_tt) * QlInvVecModqi[i].size(), cudaMemcpyHostToDevice);
	}
	cudaMalloc(&QlHatVecModt0_device, sizeof(uint64_tt) * (L+1)*(L+1));
	for(int i = 0; i <= L; i++)
	{
		cudaMemcpy(QlHatVecModt0_device + i * L, QlHatVecModt0[i].data(), sizeof(uint64_tt) * QlHatVecModt0[i].size(), cudaMemcpyHostToDevice);
	}

	/*************************rotGroups***********************/
	//rotGroups
	uint64_tt* rotGroups = new uint64_tt[Nh];
	long fivePows = 1;
	for (long i = 0; i < Nh; ++i) {
		rotGroups[i]=fivePows;
		fivePows *= 5;
		fivePows %= M;
	}
	
	cudaMalloc(&rotGroups_device, sizeof(uint64_tt) * Nh);
	cudaMemcpy(rotGroups_device, rotGroups, sizeof(uint64_tt) * Nh, cudaMemcpyHostToDevice);


	// pq
	cudaMemcpyToSymbol(pqt_cons, pqtVec.data(), sizeof(uint64_tt) * pqtVec.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(pqt2_cons, pqt2Vec.data(), sizeof(uint64_tt) * pqt2Vec.size(), 0, cudaMemcpyHostToDevice);
	// pqt_mu_high
	cudaMemcpyToSymbol(pqt_mu_cons_high, pqtMuVec_high.data(), sizeof(uint64_tt) * pqtMuVec_high.size(), 0, cudaMemcpyHostToDevice);
	// pqt_mu_low
	cudaMemcpyToSymbol(pqt_mu_cons_low, pqtMuVec_low.data(), sizeof(uint64_tt) * pqtMuVec_low.size(), 0, cudaMemcpyHostToDevice);
	// T//2 mod pqti
	cudaMemcpyToSymbol(halfTmodpqti_cons, halfTmodpqti.data(), sizeof(uint64_tt) * halfTmodpqti.size(), 0, cudaMemcpyHostToDevice);
	// P mod qi
	cudaMemcpyToSymbol(Pmodqi_cons, PModq.data(), sizeof(uint64_tt) * PModq.size(), 0, cudaMemcpyHostToDevice);
	// P^-1 mod qi
	cudaMemcpyToSymbol(Pinvmodqi_cons, PinvModq.data(), sizeof(uint64_tt) * PinvModq.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Pinvmodqi_shoup_cons, PinvModq_shoup.data(), sizeof(uint64_tt) * PinvModq_shoup.size(), 0, cudaMemcpyHostToDevice);
	// pk/P mod pk
	cudaMemcpyToSymbol(pHatInvVecModp_cons, pHatInvVecModp_23.data(), sizeof(uint64_tt) * pHatInvVecModp_23.size(), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(pHatInvVecModp_shoup_cons, pHatInvVecModp_23_shoup.data(), sizeof(uint64_tt) * pHatInvVecModp_23_shoup.size(), 0, cudaMemcpyHostToDevice);
	// Ri mod ti
	uint64_tt* temp_mem_device;
	cudaMalloc(&temp_mem_device, sizeof(uint64_tt) * t_num * Ri_blockNum);
	// qr
	cudaMemcpyToSymbol(qr_cons, qrVec.data(), sizeof(uint64_tt) * qrVec.size(), 0, cudaMemcpyHostToDevice);

	cudaMalloc(&QR_mod, sizeof(uint64_tt) * qrVec.size());
	cudaMemcpy(QR_mod, qrVec.data(), sizeof(uint64_tt) * qrVec.size(), cudaMemcpyHostToDevice);

	// qr_mu_high
	cudaMemcpyToSymbol(qr_mu_cons_high, qrMuVec_high.data(), sizeof(uint64_tt) * qrMuVec_high.size(), 0, cudaMemcpyHostToDevice);
	// qr_mu_low
	cudaMemcpyToSymbol(qr_mu_cons_low, qrMuVec_low.data(), sizeof(uint64_tt) * qrMuVec_low.size(), 0, cudaMemcpyHostToDevice);

	cudaMalloc(&QR_Mu_high, sizeof(uint64_tt) * qrMuVec_high.size());
	cudaMemcpy(QR_Mu_high, qrMuVec_high.data(), sizeof(uint64_tt) * qrMuVec_high.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&QR_Mu_low, sizeof(uint64_tt) * qrMuVec_low.size());
	cudaMemcpy(QR_Mu_low, qrMuVec_low.data(), sizeof(uint64_tt) * qrMuVec_low.size(), cudaMemcpyHostToDevice);

	for(int i = 0; i < t_num; i++)
	{
		cudaMemcpy(temp_mem_device + i*Ri_blockNum, Rimodti[i].data(), sizeof(uint64_tt) * Rimodti[i].size(), cudaMemcpyHostToDevice);
	}
	cudaMemcpyToSymbol(Rimodti_cons, temp_mem_device, sizeof(uint64_tt) * t_num * Ri_blockNum, 0, cudaMemcpyDeviceToDevice);
	cudaFree(temp_mem_device);
	// T mod pqi
	cudaMemcpyToSymbol(Tmodpqi_cons, Tmodpqi.data(), sizeof(uint64_tt) * Tmodpqi.size(), 0, cudaMemcpyHostToDevice);
	
	h = 64;
	// for sk <- HWT(h)
	// cout<<"randomArray_len: "<<randomArray_len<<endl;
	// cout<<"h: "<<h<<endl;
	randomArray_len += sizeof(uint32_tt) * h + sizeof(uint8_tt) * h;
	// for pk.a <- R_{QL}^2
	randomArray_len += sizeof(uint64_tt) * N * (L+1);
	// for pk.e <- X_{QL}
	randomArray_len += sizeof(uint32_tt) * N;
	// for swk.a <- R_{PQL}^2
	randomArray_len += sizeof(uint64_tt) * dnum * N * (L+1+K);
	// for swk.e <- X_{PQL}
	randomArray_len += sizeof(uint32_tt) * dnum * N;
	randomArray_len += sizeof(uint32_tt) * dnum * N;

	// cout<<"randomArray_len: "<<randomArray_len<<endl;

	cudaMalloc(&randomArray_device, randomArray_len);
	RNG::generateRandom_device(randomArray_device, randomArray_len);
	randomArray_sk_device = randomArray_device;

	// print_device_array(randomArray_sk_device, N, "inti ra_device");

	randomArray_pk_device = randomArray_sk_device + h * sizeof(uint32_tt) / sizeof(uint8_tt) + h * sizeof(uint8_tt) / sizeof(uint8_tt);
	randomArray_e_pk_device = randomArray_pk_device + N * (L+1) * sizeof(uint64_tt) / sizeof(uint8_tt);

	randomArray_swk_device = randomArray_e_pk_device + dnum * N * sizeof(uint32_tt) / sizeof(uint8_tt);
	randomArray_e_swk_device = randomArray_swk_device + dnum * N * (L+1+K) * sizeof(uint64_tt) / sizeof(uint8_tt);

/******************************************for encode & decode********************************************/
	cudaMalloc(&decode_buffer, sizeof(uint64_tt) * N * (L+1));
	cudaMalloc(&encode_buffer, sizeof(uint64_tt) * N * (L+1));

/******************************************for BFV enc/add/sub********************************************/
	vector<uint64_tt> values_Ql(q_num);
	for (size_t i = 0; i < q_num; i++)
		values_Ql[i] = qVec[i];

	// Compute big Ql
	vector<uint64_tt> base_Ql(qVec.begin() , qVec.end());

	vector<uint64_tt> bigQl(q_num, 0);
	multiply_many_uint64(values_Ql.data(), q_num, bigQl.data());

	// printf("bigQl = [ ");
    // for( size_t idx = 0 ; idx < q_num ; ++idx )
    // {
    //     printf( "%llu " , bigQl[idx] );
    // }    printf(" ]\n");

	uint64_tt t = plain_modulus;
 	uint64_tt bigQl_mod_t_value = modulo_uint(bigQl.data(), q_num, t);
	// printf("bigqlmod : %llu\n" , bigQl_mod_t_value);
	uint64_tt negQl_mod_t_ = t - bigQl_mod_t_value;
	// printf("negQL_mod_t : %llu\n" , negQl_mod_t_);
	uint64_tt negQl_mod_tshoup_ = compute_shoup(negQl_mod_t_, t);

	negQl_mod_t = negQl_mod_t_;
	negQl_mod_t_shoup = negQl_mod_tshoup_;

	tInv_mod_q_.resize(q_num);
	vector<uint64_tt> tInv_mod_q_shoup_(q_num);
	for (size_t i = 0; i < q_num; i++) {
		uint64_tt tInv_mod_q_i_value;
		// auto &qi = base_Ql.base()[i];
		auto &qi = base_Ql[i];
		if (!try_invert_uint_mod(t, qi, tInv_mod_q_i_value))
			throw std::logic_error("invalid rns bases when computing tInv_mod_q_i");
		tInv_mod_q_[i] = tInv_mod_q_i_value;
		tInv_mod_q_shoup_[i] = compute_shoup(tInv_mod_q_i_value, qi);
	}

	cudaMalloc(&tInv_mod_q, sizeof(uint64_tt) * q_num );
	cudaMalloc(&tInv_mod_q_shoup, sizeof(uint64_tt) * q_num );
	
	cudaMemcpyAsync(tInv_mod_q, tInv_mod_q_.data(), q_num * sizeof(uint64_tt), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(tInv_mod_q_shoup, tInv_mod_q_shoup_.data(), q_num * sizeof(uint64_tt),
					cudaMemcpyHostToDevice);

	// print_device_array(tInv_mod_q, q_num, "tInv_mod_q");

/******************************************for BFV dec********************************************/
	//qHat init-------------------------------------------------------
	mod_.resize(q_num);
	prod_mod_.resize(q_num);
	prod_hat_.resize(q_num * q_num);
	hat_mod_.resize(q_num);
	hat_mod_shoup_.resize(q_num);
	hatInv_mod_.resize(q_num);
	hatInv_mod_shoup_.resize(q_num);
	inv_.resize(q_num);
	for (size_t i = 0; i < q_num; i++)
	{
		mod_[i] = qVec[i];
	}
	if (q_num > 1)
	{
		std::vector<uint64_tt> rnsbase_values(q_num);
		for (size_t i = 0; i < q_num; i++)
			rnsbase_values[i] = mod_[i];

		// Create punctured products
		for (size_t i = 0; i < q_num; i++) {
			multiply_many_uint64_except(rnsbase_values.data(), q_num, i, prod_hat_.data() + i * q_num);
		}

		// printf("prod_hat = [ ");
		// for( size_t idx = 0 ; idx < q_num ; ++idx )
		// {
		// 	printf( "%llu " , prod_hat_[idx] );
		// }    printf(" ]\n");

		// Compute the full product
		multiply_many_uint64(rnsbase_values.data(), q_num, prod_mod_.data());

		// printf("prod_mod = [ ");
		// for( size_t idx = 0 ; idx < q_num ; ++idx )
		// {
		// 	printf( "%llu " , prod_mod_[idx] );
		// }    printf(" ]\n");

		// Compute inverses of punctured products mod primes
		for (size_t i = 0; i < q_num; i++)
		{
			// punctured_prod[i] % qi
			uint64_tt qiHat_mod_qi = modulo_uint(prod_hat_.data() + i * q_num, q_num, mod_[i]);
			// qiHat_mod_qi = qiHat_mod_qi^{-1} % qi
			uint64_tt qiHatInv_mod_qi;

			if (!try_invert_uint_mod(qiHat_mod_qi, mod_[i], qiHatInv_mod_qi))
				throw invalid_argument("invalid modulus");

			hat_mod_[i] = qiHat_mod_qi;
			hat_mod_shoup_[i] = compute_shoup(qiHat_mod_qi, mod_[i]);
			hatInv_mod_[i] = qiHatInv_mod_qi;
			hatInv_mod_shoup_[i] = compute_shoup(qiHatInv_mod_qi, mod_[i]);

			// printf( "hat_mod : %llu\n" , hat_mod_[i] );
			// printf( "hatInv_mod : %llu\n" , hatInv_mod_[i] );
		}

		// printf("qiHat_mod_qi = [ ");
		// for( size_t idx = 0 ; idx < q_num ; ++idx )
		// {
		// 	printf( "%llu " , hat_mod_[idx] );
		// }printf(" ]\n");
		
		// printf("qiHat_Inv_mod_qi = [ ");
		// for( size_t idx = 0 ; idx < q_num ; ++idx )
		// {
		// 	printf( "%llu " , hatInv_mod_[idx] );
		// }printf(" ]\n");

		// puts("-------------------------------------------------");

		// compute 1.0 / qi
		for (size_t i = 0; i < q_num; i++)
		{
			uint64_tt qi = mod_[i];
			double inv = 1.0 / static_cast<double>(qi);
			inv_[i] = inv;
		}
	}else
	{
		// Only one single modulus
		prod_mod_[0] = mod_[0];
		prod_hat_[0] = 1;
		hat_mod_[0] = 1;
		hat_mod_shoup_[0] = compute_shoup(1, mod_[0]);
		hatInv_mod_[0] = 1;
		hatInv_mod_shoup_[0] = compute_shoup(1, mod_[0]);
		inv_[0] = 1.0 / static_cast<double>(mod_[0]);
	}
	//-----------------------------------------------------------

	vector<uint64_tt> v_qi(q_num);
	for (size_t i = 0; i < q_num; i++)
	{
		v_qi[i] = qVec[i];
	}
	size_t max_q_idx = max_element(v_qi.begin(), v_qi.end()) - v_qi.begin();
	size_t min_q_idx = min_element(v_qi.begin(), v_qi.end()) - v_qi.begin();
			
	// HPS Decrypt Scale&Round
	size_t qMSB_ = get_significant_bit_count(v_qi[max_q_idx]);

	// printf( "%llu %lu\n" , v_qi[max_q_idx] , qMSB_ );

	uint64_tt q_num_ = static_cast<uint64_tt>(q_num);
	size_t sizeQMSB_ = get_significant_bit_count_uint(&q_num_, 1);

	// printf( "%llu %lu\n" , q_num_ , sizeQMSB_ );

	size_t tMSB_ = get_significant_bit_count_uint(&plain_modulus, 1);
	qMSB = qMSB_;
	sizeQMSB = sizeQMSB_;
	tMSB = tMSB_;

	// printf( "\nqMSB : %llu\ntMSB : %llu\nsizeQMSB : %llu\n\n" , qMSB , tMSB , sizeQMSB );

	vector<uint64_tt> t_QHatInv_mod_q_div_q_mod_t_(q_num);
	vector<uint64_tt> t_QHatInv_mod_q_div_q_mod_t_shoup_(q_num);
	vector<uint64_tt> t_QHatInv_mod_q_div_q_mod_q_(q_num);
	vector<uint64_tt> t_QHatInv_mod_q_div_q_mod_q_shoup_(q_num);
	vector<double> t_QHatInv_mod_q_div_q_frac_(q_num);
	vector<uint64_tt> t_QHatInv_mod_q_B_div_q_mod_t_(q_num);
	vector<uint64_tt> t_QHatInv_mod_q_B_div_q_mod_t_shoup_(q_num);
	vector<double> t_QHatInv_mod_q_B_div_q_frac_(q_num);

	for (size_t i = 0; i < q_num; ++i)
	{
		auto qi = base_Ql[i];

		std::vector<uint64_tt> big_t_QHatInv_mod_qi(2, 0);

		auto qiHatInv_mod_qi = hatInv_mod_[i];
		// auto qiHatInv_mod_qi = QlInvVecModqi[L][i];

		multiply_uint(&t, 1, qiHatInv_mod_qi, 2, big_t_QHatInv_mod_qi.data());

		// printf( "big_t_QHatInv_mod_qi : %llu %llu\n" , big_t_QHatInv_mod_qi[0] , big_t_QHatInv_mod_qi[1] );

		std::vector<uint64_tt> padding_zero_qi(2, 0);
		padding_zero_qi[0] = qi;

		std::vector<uint64_tt> big_t_QHatInv_mod_q_div_qi(2, 0);

		divide_uint_inplace(big_t_QHatInv_mod_qi.data(), padding_zero_qi.data(), 2,
							big_t_QHatInv_mod_q_div_qi.data());

		// printf( "big_t_QHatInv_mod_q_div_q : %llu %llu\n" , big_t_QHatInv_mod_q_div_qi[0] , big_t_QHatInv_mod_q_div_qi[1] );

		t_QHatInv_mod_q_div_q_mod_q_[i] = modulo_uint(big_t_QHatInv_mod_q_div_qi.data(), 2, qi);
		// printf( "t_QHatInv_mod_q_div_q_mod_q : %llu\n" , t_QHatInv_mod_q_div_q_mod_q_[i] );
		t_QHatInv_mod_q_div_q_mod_q_shoup_[i] = compute_shoup(t_QHatInv_mod_q_div_q_mod_q_[i], qi);
		cudaMalloc(&t_QHatInv_mod_q_div_q_mod_q, t_QHatInv_mod_q_div_q_mod_q_.size() * sizeof(uint64_tt) );
		cudaMalloc(&t_QHatInv_mod_q_div_q_mod_q_shoup, t_QHatInv_mod_q_div_q_mod_q_shoup_.size() * sizeof(uint64_tt) );
		cudaMemcpyAsync(t_QHatInv_mod_q_div_q_mod_q, t_QHatInv_mod_q_div_q_mod_q_.data(),
						t_QHatInv_mod_q_div_q_mod_q_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(t_QHatInv_mod_q_div_q_mod_q_shoup, t_QHatInv_mod_q_div_q_mod_q_shoup_.data(),
						t_QHatInv_mod_q_div_q_mod_q_shoup_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

		uint64_tt value_t_QHatInv_mod_q_div_q_mod_t = modulo_uint(big_t_QHatInv_mod_q_div_qi.data(), 2, t);

		t_QHatInv_mod_q_div_q_mod_t_[i] = value_t_QHatInv_mod_q_div_q_mod_t;
		// printf( "t_QHatInv_mod_q_div_q_mod_t : %llu\n" , t_QHatInv_mod_q_div_q_mod_t_[i] );

		t_QHatInv_mod_q_div_q_mod_t_shoup_[i] = compute_shoup(value_t_QHatInv_mod_q_div_q_mod_t, t);
		cudaMalloc(&t_QHatInv_mod_q_div_q_mod_t, t_QHatInv_mod_q_div_q_mod_t_.size() * sizeof(uint64_tt) );
		cudaMalloc(&t_QHatInv_mod_q_div_q_mod_t_shoup, t_QHatInv_mod_q_div_q_mod_t_shoup_.size() * sizeof(uint64_tt) );
		cudaMemcpyAsync(t_QHatInv_mod_q_div_q_mod_t, t_QHatInv_mod_q_div_q_mod_t_.data(),
						t_QHatInv_mod_q_div_q_mod_t_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(t_QHatInv_mod_q_div_q_mod_t_shoup, t_QHatInv_mod_q_div_q_mod_t_shoup_.data(),
						t_QHatInv_mod_q_div_q_mod_t_shoup_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

		uint64_tt numerator = modulo_uint(big_t_QHatInv_mod_qi.data(), 2, qi);
		uint64_tt denominator = qi;
		t_QHatInv_mod_q_div_q_frac_[i] = static_cast<double>(numerator) / static_cast<double>(denominator);

		// printf( "big_t_QHatInv_mod_qi : %llu %llu\n" , big_t_QHatInv_mod_qi[0] , big_t_QHatInv_mod_qi[1] );
		// printf( "qi : %llu\n" , qi );
		// printf( "t_QHatInv_mod_q_div_q_frac : %f\n" , t_QHatInv_mod_q_div_q_frac_[i] );

		cudaMalloc(&t_QHatInv_mod_q_div_q_frac, t_QHatInv_mod_q_div_q_frac_.size() * sizeof(double) );
		cudaMemcpyAsync(t_QHatInv_mod_q_div_q_frac, t_QHatInv_mod_q_div_q_frac_.data(),
						t_QHatInv_mod_q_div_q_frac_.size() * sizeof(double), cudaMemcpyHostToDevice);

		if (qMSB_ + sizeQMSB_ >= 52)
		{
			size_t qMSBHf = qMSB_ >> 1;

			std::vector<uint64_tt> QHatInv_mod_qi_B(2, 0);
			QHatInv_mod_qi_B[0] = qiHatInv_mod_qi;
			left_shift_uint128(QHatInv_mod_qi_B.data(), qMSBHf, QHatInv_mod_qi_B.data());

			// printf( "QHatInv_mod_qi_B : %llu %llu\n" , QHatInv_mod_qi_B[0] , QHatInv_mod_qi_B[1] );

			uint64_tt QHatInv_B_mod_qi = modulo_uint(QHatInv_mod_qi_B.data(), 2, qi);
			
			// printf( "QHatInv_mod_qi_B : %llu %llu\n" , QHatInv_mod_qi_B[0] , QHatInv_mod_qi_B[1] );

			std::vector<uint64_tt> t_QHatInv_B_mod_qi(2, 0);
			multiply_uint(&t, 1, QHatInv_B_mod_qi, 2, t_QHatInv_B_mod_qi.data());

			std::vector<uint64_tt> t_QHatInv_B_mod_qi_div_qi(2, 0);
			divide_uint_inplace(t_QHatInv_B_mod_qi.data(), padding_zero_qi.data(), 2,
								t_QHatInv_B_mod_qi_div_qi.data());

			uint64_tt value_t_QHatInv_mod_q_B_div_q_mod_t = modulo_uint(t_QHatInv_B_mod_qi_div_qi.data(), 2, t);

			t_QHatInv_mod_q_B_div_q_mod_t_[i] = value_t_QHatInv_mod_q_B_div_q_mod_t;
			t_QHatInv_mod_q_B_div_q_mod_t_shoup_[i] =
					compute_shoup(value_t_QHatInv_mod_q_B_div_q_mod_t, t);
			cudaMalloc(&t_QHatInv_mod_q_B_div_q_mod_t , t_QHatInv_mod_q_B_div_q_mod_t_.size() * sizeof(uint64_tt) );
			cudaMalloc(&t_QHatInv_mod_q_B_div_q_mod_t_shoup , t_QHatInv_mod_q_B_div_q_mod_t_shoup_.size() * sizeof(uint64_tt) );
			cudaMemcpyAsync(t_QHatInv_mod_q_B_div_q_mod_t, t_QHatInv_mod_q_B_div_q_mod_t_.data(),
							t_QHatInv_mod_q_B_div_q_mod_t_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);
			cudaMemcpyAsync(
					t_QHatInv_mod_q_B_div_q_mod_t_shoup, t_QHatInv_mod_q_B_div_q_mod_t_shoup_.data(),
					t_QHatInv_mod_q_B_div_q_mod_t_shoup_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

			numerator = modulo_uint(t_QHatInv_B_mod_qi.data(), 2, qi);
			t_QHatInv_mod_q_B_div_q_frac_[i] = static_cast<double>(numerator) / static_cast<double>(denominator);
			cudaMalloc(&t_QHatInv_mod_q_B_div_q_frac , t_QHatInv_mod_q_B_div_q_frac_.size() * sizeof(double) );
			cudaMemcpyAsync(t_QHatInv_mod_q_B_div_q_frac, t_QHatInv_mod_q_B_div_q_frac_.data(),
							t_QHatInv_mod_q_B_div_q_frac_.size() * sizeof(double), cudaMemcpyHostToDevice);
		}

		// puts("----------------------------------------------");
	}

/******************************************for BFV mult*******************************************/
	//rHat init-------------------------------------------------------
	mod_r.resize(r_num);
	prod_mod_r.resize(r_num);
	prod_hat_r.resize(r_num * r_num);
	hat_mod_r.resize(r_num);
	hat_mod_shoup_r.resize(r_num);
	hatInv_mod_r.resize(r_num);
	hatInv_mod_shoup_r.resize(r_num);
	inv_r.resize(r_num);

	cudaMalloc(&mult_buffer, sizeof(uint64_tt) * N * (L+1));
	cudaMalloc(&mult_scale_buffer, sizeof(uint64_tt) * 2 * N * (L+1));
	cudaMalloc(&sec_buffer, sizeof(uint64_tt) * N * (L+K+1));

	for (size_t i = 0; i < r_num; i++)
	{
		mod_r[i] = rVec[i];
	}
	if (r_num > 1)
	{
		std::vector<uint64_tt> rnsbase_values_r(r_num);
		for (size_t i = 0; i < r_num; i++)
			rnsbase_values_r[i] = mod_r[i];

		// Create punctured products
		for (size_t i = 0; i < r_num; i++) {
			multiply_many_uint64_except(rnsbase_values_r.data(), r_num, i, prod_hat_r.data() + i * r_num);
		}

		// printf("prod_hat_r = [ ");
		// for( size_t idx = 0 ; idx < r_num ; ++idx )
		// {
		// 	printf( "%llu " , prod_hat_r[idx] );
		// }    printf(" ]\n");

		// Compute the full product
		multiply_many_uint64(rnsbase_values_r.data(), r_num, prod_mod_r.data());

		// printf("prod_mod_r = [ ");
		// for( size_t idx = 0 ; idx < r_num ; ++idx )
		// {
		// 	printf( "%llu " , prod_mod_r[idx] );
		// }    printf(" ]\n");

		// Compute inverses of punctured products mod primes
		for (size_t i = 0; i < r_num; i++)
		{
			// punctured_prod[i] % ri
			uint64_tt riHat_mod_ri = modulo_uint(prod_hat_r.data() + i * r_num, r_num, mod_r[i]);
			// riHat_mod_ri = riHat_mod_ri^{-1} % ri
			uint64_tt riHatInv_mod_ri;

			if (!try_invert_uint_mod(riHat_mod_ri, mod_r[i], riHatInv_mod_ri))
				throw invalid_argument("invalid modulus");

			hat_mod_r[i] = riHat_mod_ri;
			hat_mod_shoup_r[i] = compute_shoup(riHat_mod_ri, mod_r[i]);
			hatInv_mod_r[i] = riHatInv_mod_ri;
			hatInv_mod_shoup_r[i] = compute_shoup(riHatInv_mod_ri, mod_r[i]);

			// printf( "hat_mod_r : %llu\n" , hat_mod_r[i] );
			// printf( "hatInv_mod_r : %llu\n" , hatInv_mod_r[i] );
		}

		// printf("riHat_mod_ri = [ ");
		// for( size_t idx = 0 ; idx < r_num ; ++idx )
		// {
		// 	printf( "%llu " , hat_mod_r[idx] );
		// }printf(" ]\n");
		
		// printf("riHat_Inv_mod_ri = [ ");
		// for( size_t idx = 0 ; idx < r_num ; ++idx )
		// {
		// 	printf( "%llu " , hatInv_mod_r[idx] );
		// }printf(" ]\n");

		// puts("-------------------------------------------------");

		// compute 1.0 / ri
		for (size_t i = 0; i < r_num; i++)
		{
			uint64_tt ri = mod_r[i];
			double inv = 1.0 / static_cast<double>(ri);
			inv_r[i] = inv;
		}
	}else
	{
		// Only one single modulus
		prod_mod_r[0] = mod_r[0];
		prod_hat_r[0] = 1;
		hat_mod_r[0] = 1;
		hat_mod_shoup_r[0] = compute_shoup(1, mod_r[0]);
		hatInv_mod_r[0] = 1;
		hatInv_mod_shoup_r[0] = compute_shoup(1, mod_r[0]);
		inv_r[0] = 1.0 / static_cast<double>(mod_r[0]);
	}
	//-----------------------------------------------------------

	// puts("BaseConverter from Q to R_____________________________________________________");
	QtoR.init( qVec , rVec , prod_mod_ , prod_hat_ , hatInv_mod_ , hatInv_mod_shoup_ , prod_mod_r , inv_ , 
		qrMuVec_high , qrMuVec_low , 0 , qVec.size() );
	// puts("");puts("");

	// puts("BaseConverter from R to Q_____________________________________________________");
	RtoQ.init( rVec , qVec , prod_mod_r , prod_hat_r , hatInv_mod_r , hatInv_mod_shoup_r , prod_mod_ , inv_r ,
		qrMuVec_high , qrMuVec_low , qVec.size() , 0 );
	// puts("");puts("");

	//==========================================================================
	auto bigint_R = prod_mod_r.data();
	vector<uint64_tt> tR(r_num + 1, 0);
	multiply_uint(bigint_R, r_num, plain_modulus, r_num + 1, tR.data());

	// print_host_array(bigint_R , r_num , "R");
	// print_host_array(tR.data() , r_num + 1 , "tR");
	
	// Used for t/Q scale&round in HPS method
	vector<double> tRSHatInvModsDivsFrac(L + 1);
	vector<uint64_tt> tRSHatInvModsDivsModr(r_num * ((L + 1) + 1));
	vector<uint64_tt> tRSHatInvModsDivsModr_shoup(r_num * ((L + 1) + 1));

	//compute base_QR.SHatInvMods( s is q0,q1,...qi,r0,r1,...rj )
	//=============================================================
	prod_hat_qr.resize((L + 1 + r_num) * (L + 1 + r_num));
	hat_mod_qr.resize(L + 1 + r_num);
	hatInv_mod_qr.resize(L + 1 + r_num);

	// Create punctured products
	for (size_t i = 0; i < (L + 1 + r_num); i++) {
		multiply_many_uint64_except(qrVec.data(), (L + 1 + r_num), i, prod_hat_qr.data() + i * (L + 1 + r_num));
	}

	// Compute inverses of punctured products mod primes
	for (size_t i = 0; i < (L + 1 + r_num); i++)
	{
		// punctured_prod[i] % si
		uint64_tt siHat_mod_si = modulo_uint(prod_hat_qr.data() + i * (L + 1 + r_num), L + 1 + r_num, qrVec[i]);
		// siHat_mod_si = siHat_mod_si^{-1} % si
		uint64_tt siHatInv_mod_si;

		if (!try_invert_uint_mod(siHat_mod_si, qrVec[i], siHatInv_mod_si))
			throw invalid_argument("invalid modulus");

		hat_mod_qr[i] = siHat_mod_si;
		// hat_mod_shoup_qr[i] = compute_shoup(siHat_mod_si, qrVec[i]);
		hatInv_mod_qr[i] = siHatInv_mod_si;
		// hatInv_mod_shoup_qr[i] = compute_shoup(siHatInv_mod_si, qrVec[i]);

		// printf( "hat_mod : %llu\n" , hat_mod_[i] );
		// printf( "hatInv_mod : %llu\n" , hatInv_mod_[i] );
	}
	//=============================================================

	// first compute tRSHatInvMods
	vector<vector<uint64_tt>> tRSHatInvMods((L + 1 + r_num));
	for (size_t i = 0; i < (L + 1 + r_num); i++)
	{
		// resize tRSHatInvModsi to r_num + 2 and initialize to 0
		tRSHatInvMods[i].resize(r_num + 2, 0);
		uint64_tt SHatInvModsi;
		// auto SHatInvModsi = base_QlRl.QHatInvModq()[i];
		SHatInvModsi = hatInv_mod_qr[i];
		multiply_uint(tR.data(), r_num + 1, SHatInvModsi, r_num + 2, tRSHatInvMods[i].data());
	}

	// print_host_array(tRSHatInvMods[0].data() , r_num + 2 , "tRSHatInvMods[0]");

	// compute tRSHatInvModsDivsFrac
	for (size_t i = 0; i < (L + 1); i++)
	{
		auto qi = qVec[i];
		uint64_tt tRSHatInvModsModqi = modulo_uint(tRSHatInvMods[i].data(), r_num + 2, qi);
		tRSHatInvModsDivsFrac[i] = static_cast<double>(tRSHatInvModsModqi) / static_cast<double>(qi);
	}
	cudaMalloc(&tRSHatInvModsDivsFrac_ , tRSHatInvModsDivsFrac.size() * sizeof(double));
	cudaMemcpyAsync(tRSHatInvModsDivsFrac_, tRSHatInvModsDivsFrac.data(),
					tRSHatInvModsDivsFrac.size() * sizeof(double), cudaMemcpyHostToDevice, 0);

	// printf("tRSHatInvModsDivsFrac = [ ");
	// for( size_t idx = 0 ; idx <= (L + 1) ; ++idx )
	// {
	// 	printf( "%lf " , tRSHatInvModsDivsFrac[idx] );
	// }
	// printf(" ]\n");

	// compute tRSHatInvModsDivs
	vector<vector<uint64_tt>> tRSHatInvModsDivs((L + 1 + r_num));
	for (size_t i = 0; i < (L + 1 + r_num); i++)
	{
		// resize tRSHatInvModsDivsi to r_num + 2 and initialize to 0
		tRSHatInvModsDivs[i].resize(r_num + 2, 0);
		// align si with big integer tRSHatInvMods
		auto si = qrVec[i];
		vector<uint64_tt> bigint_si(r_num + 2, 0);
		bigint_si[0] = si;
		// div si
		std::vector<uint64_tt> temp_remainder(r_num + 2, 0);
		divide_uint(tRSHatInvMods[i].data(), bigint_si.data(), r_num + 2, tRSHatInvModsDivs[i].data(),
					temp_remainder.data());
	}

	// print_host_array(tRSHatInvModsDivs[0].data(), r_num + 2 , "rRSHatInvModsDivs[0]");

	// compute tRSHatInvModsDivsModr
	for (size_t j = 0; j < r_num; j++)
	{
		auto &rj = rVec[j];
		for (size_t i = 0; i < (L + 1); i++)
		{
			// mod rj
			uint64_tt tRSHatInvModsDivqiModrj = modulo_uint(tRSHatInvModsDivs[i].data(), r_num + 2, rj);
			tRSHatInvModsDivsModr[j * ((L + 1) + 1) + i] = tRSHatInvModsDivqiModrj;
			tRSHatInvModsDivsModr_shoup[j * ((L + 1) + 1) + i] =
					compute_shoup(tRSHatInvModsDivqiModrj, rj);
		}
		// mod rj
		uint64_tt tRSHatInvModsDivrjModrj = modulo_uint(tRSHatInvModsDivs[(L + 1) + j].data(), r_num + 2, rj);
		tRSHatInvModsDivsModr[j * ((L + 1) + 1) + (L + 1)] = tRSHatInvModsDivrjModrj;
		tRSHatInvModsDivsModr_shoup[j * ((L + 1) + 1) + (L + 1)] =
				compute_shoup(tRSHatInvModsDivrjModrj, rj);
	}

	// print_host_array(tRSHatInvModsDivsModr.data(), L + 2 , "rRSHatInvModsDivsModr[0]");

	cudaMalloc(&tRSHatInvModsDivsModr_ , tRSHatInvModsDivsModr.size() * sizeof(uint64_tt));
	cudaMalloc(&tRSHatInvModsDivsModr_shoup_ , tRSHatInvModsDivsModr_shoup.size() * sizeof(uint64_tt));
	cudaMemcpyAsync(tRSHatInvModsDivsModr_, tRSHatInvModsDivsModr.data(),
					tRSHatInvModsDivsModr.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(tRSHatInvModsDivsModr_shoup_, tRSHatInvModsDivsModr_shoup.data(),
					tRSHatInvModsDivsModr_shoup.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice, 0);

	//==========================================================================

	//check correctness of decode
	cudaMalloc(&encode_buffer_device, N * (L + 1) * sizeof(uint64_tt));
	cudaMalloc(&decode_buffer_device, N * (L + 1) * sizeof(uint64_tt));
}

#define mod_switch_block 1024

__host__ void BFVContext::forwardNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num)
{
    uint32_tt num = poly_num * mod_num;
    uint64_tt* device_target = device_a + (N * idx_poly); 
    uint64_tt* psi_powers_target = pqtPsiTable_device + (N * idx_mod);
    if(N == 65536)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(16, num);
        CTBasedNTTInner_batch<1, 65536, 15> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
        CTBasedNTTInner_batch<2, 65536, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
        CTBasedNTTInner_batch<4, 65536, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
        CTBasedNTTInner_batch<8, 65536, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

        CTBasedNTTInnerSingle_batch<16, 65536, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
    }
    else if (N == 32768)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(8, num);
        CTBasedNTTInner_batch<1, 32768, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
        CTBasedNTTInner_batch<2, 32768, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
        CTBasedNTTInner_batch<4, 32768, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

        CTBasedNTTInnerSingle_batch<8, 32768, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
    }
    else if (N == 16384)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(4, num);
        CTBasedNTTInner_batch<1, 16384, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
        CTBasedNTTInner_batch<2, 16384, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

        CTBasedNTTInnerSingle_batch<4, 16384, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
    }
    else if (N == 8192)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(2, num);
        CTBasedNTTInner_batch<1, 8192, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);

        CTBasedNTTInnerSingle_batch<2, 8192, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
    }
    else if (N == 4096)
    {
        dim3 single_dim(1, num);
        CTBasedNTTInnerSingle_batch<1, 4096, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
    }
    else if (N == 2048)
    {
        dim3 single_dim(1, num);
        CTBasedNTTInnerSingle_batch<1, 2048, 10> << <single_dim, 1024, 2048 * sizeof(uint64_tt), 0 >> > (device_target, psi_powers_target, mod_num, idx_mod);
    }
}

__host__ void BFVContext::inverseNTT_batch(uint64_tt* device_a, int idx_poly, int idx_mod, uint32_tt poly_num, uint32_tt mod_num)
{
    uint32_tt num = poly_num * mod_num;
    uint64_tt* device_target = device_a + (N * idx_poly);
    uint64_tt* psiinv_powers_target = pqtPsiInvTable_device + (N * idx_mod);
    if (N == 65536)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(16, num);
        GSBasedINTTInnerSingle_batch<16, 65536, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
        GSBasedINTTInner_batch<8, 65536, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        GSBasedINTTInner_batch<4, 65536, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        GSBasedINTTInner_batch<2, 65536, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        GSBasedINTTInner_batch<1, 65536, 15> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
    }
    else if (N == 32768)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(8, num);
        GSBasedINTTInnerSingle_batch<8, 32768, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
        GSBasedINTTInner_batch<4, 32768, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        GSBasedINTTInner_batch<2, 32768, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        GSBasedINTTInner_batch<1, 32768, 14> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
    }
    else if (N == 16384)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(4, num);
        GSBasedINTTInnerSingle_batch<4, 16384, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
        GSBasedINTTInner_batch<2, 16384, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        GSBasedINTTInner_batch<1, 16384, 13> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
    }
    else if (N == 8192)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(2, num);
        GSBasedINTTInnerSingle_batch<2, 8192, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
        
        GSBasedINTTInner_batch<1, 8192, 12> << <multi_dim, 1024, 0, 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
    }
    else if (N == 4096)
    {
        dim3 multi_dim(N / 1024 / 2, num);
        dim3 single_dim(1, num);
        GSBasedINTTInnerSingle_batch<1, 4096, 11> << <single_dim, 1024, 4096 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
    }
    else if (N == 2048)
    {
        dim3 single_dim(1, num);
        GSBasedINTTInnerSingle_batch<1, 2048, 10> << <single_dim, 1024, 2048 * sizeof(uint64_tt), 0 >> > (device_target, psiinv_powers_target, mod_num, idx_mod);
    }
}



/*****************************************************new_batch_ntt***************************************************************/
__host__ void BFVContext::ToNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);//n1
    const int per_thread_ntt_size = 8;
    const int first_stage_radix_size = 256;//N1
    const int second_radix_size = N / first_stage_radix_size;
    const int pad = 4;// the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    NTT8pointPerThread_kernel1<<<gridDim, (first_stage_radix_size / 8) * pad,
                              (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                             (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, pad, poly_mod_len);
    NTT8pointPerThread_kernel2<<<gridDim, blockDim, per_block_memory>>>
                                (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, second_radix_size, poly_mod_len);
}

__host__ void BFVContext::FromNTTInplace(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);
    const int per_thread_ntt_size = 8;
    const int second_radix_size = 256; 
    const int first_stage_radix_size = N / second_radix_size;//N1
    const int pad = 4;
    int block_size2 = (first_stage_radix_size / 8) * pad;
    int grid_size2 = N * mod_num / (8 * block_size2);
    dim3 gridDim2(grid_size2, poly_num);
    dim3 blockDim2(block_size2);
    // the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    INTT8pointPerThread_kernel1<<<gridDim, blockDim, per_block_memory>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len);
    INTT8pointPerThread_kernel2<<<gridDim2, blockDim2,
                            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, n_inv_device, n_inv_shoup_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len);
}

__host__ void BFVContext::ToNTTInplace_for_externalProduct(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len, int cipher_mod_num, int batch_size)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num, batch_size);
    dim3 blockDim(block_size);//n1
    const int per_thread_ntt_size = 8;
    const int first_stage_radix_size = 256;//N1
    const int second_radix_size = N / first_stage_radix_size;
    const int pad = 4;// the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    NTT8pointPerThread_for_ext_kernel1<<<gridDim, (first_stage_radix_size / 8) * pad,
                              (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                             (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, pad, poly_mod_len, cipher_mod_num);
    NTT8pointPerThread_for_ext_kernel2<<<gridDim, blockDim, per_block_memory>>>
                                (device_a, psi_table_device, psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, second_radix_size, poly_mod_len, cipher_mod_num);
}

__host__ void BFVContext::FromNTTInplace_for_externalProduct(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len, int cipher_mod_num, int batch_size)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num, 2);
    dim3 blockDim(block_size);
    const int per_thread_ntt_size = 8;
    const int second_radix_size = 256; 
    const int first_stage_radix_size = N / second_radix_size;//N1
    const int pad = 4;
    int block_size2 = (first_stage_radix_size / 8) * pad;
    int grid_size2 = N * mod_num / (8 * block_size2);
    dim3 gridDim2(grid_size2, poly_num, batch_size);
    dim3 blockDim2(block_size2);
    // the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    INTT8pointPerThread_for_ext_kernel1<<<gridDim, blockDim, per_block_memory>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len, cipher_mod_num);
    INTT8pointPerThread_for_ext_kernel2<<<gridDim2, blockDim2,
                            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                            (device_a, psiinv_table_device, psiinv_shoup_table_device, n_inv_device, n_inv_shoup_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len, cipher_mod_num);
}

__host__ void BFVContext::divByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num)
{
    uint64_tt* psi_powers_target = psi_table_device + (N * idx_mod);
	
	dim3 divByiAndEqual_dim(N / poly_block, mod_num, 2);
	divByiAndEqual_kernel <<< divByiAndEqual_dim, poly_block >>>(device_a, N, q_num, idx_mod, psi_powers_target);
}

__host__ void BFVContext::mulByiAndEqual(uint64_tt* device_a, int idx_mod, int mod_num)
{
    uint64_tt* psi_powers_target = psi_table_device + (N * idx_mod);

	dim3 mulByiAndEqual_dim(N / poly_block, mod_num, 2);
	mulByiAndEqual_kernel <<< mulByiAndEqual_dim, poly_block >>>(device_a, N, q_num, idx_mod, psi_powers_target);
}

__host__ void BFVContext::poly_add_complex_const_batch_device(uint64_tt* device_a, uint64_tt* add_const_buffer, int idx_a, int idx_mod, int mod_num)
{
	uint64_tt* psi_powers_target = psi_table_device + (N * idx_mod);
	uint64_tt* psi_powers_shoup_target = psi_shoup_table_device + (N * idx_mod);

    dim3 add_dim(N / poly_block, mod_num);
    poly_add_complex_const_batch_device_kernel<<< add_dim, poly_block >>>(device_a, add_const_buffer, N, psi_powers_target, psi_powers_shoup_target, idx_a, L, idx_mod);
}

__host__ void BFVContext::poly_mul_const_batch_device(uint64_tt* device_a, uint64_tt* const_real, int idx_mod, int mod_num)
{
    dim3 mul_dim(N / poly_block, mod_num, 2);
    poly_mul_const_batch_device_kernel<<< mul_dim, poly_block >>>(device_a, const_real, N, q_num, idx_mod);
}

// c1 += c2 * const
__host__ void BFVContext::poly_mul_const_add_cipher_batch_device(uint64_tt* device_a, uint64_tt* device_b, uint64_tt* const_real, int idx_mod, int mod_num)
{
    dim3 mul_dim(N / poly_block, mod_num, 2);
    poly_mul_const_batch_andAdd_device_kernel<<< mul_dim, poly_block >>>(device_a, device_b, const_real, N, q_num, idx_mod);
}

//BFV_ENCODE
__host__ void BFVContext::FromNTTInplace_for_BFV(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
	mod_num = 1;
	poly_num = 1;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);
    const int per_thread_ntt_size = 8;
    const int second_radix_size = 256; 
    const int first_stage_radix_size = N / second_radix_size;//N1
    const int pad = 4;
    int block_size2 = (first_stage_radix_size / 8) * pad;
    int grid_size2 = N * mod_num / (8 * block_size2);
    dim3 gridDim2(grid_size2, poly_num);
    dim3 blockDim2(block_size2);
    // the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    INTT8pointPerThread_for_bfv_kernel1<<<gridDim, blockDim, per_block_memory>>>
                            (device_a, plainModPsiInv_device, plainMod_shoup_inv_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len, plain_modulus);	
    INTT8pointPerThread_for_bfv_kernel2<<<gridDim2, blockDim2,
                            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                            (device_a, plainModPsiInv_device, plainMod_shoup_inv_device, n_inv_device_bfv, n_inv_shoup_device_bfv, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len, plain_modulus);
}
//BFV_DECODE
__host__ void BFVContext::ToNTTInplace_for_BFV(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
	mod_num = 1;
	poly_num = 1;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);//n1
    const int per_thread_ntt_size = 8;
    const int first_stage_radix_size = 256;//N1
    const int second_radix_size = N / first_stage_radix_size;
    const int pad = 4;// the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    NTT8pointPerThread_kernel1<<<gridDim, (first_stage_radix_size / 8) * pad,
                              (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                             (device_a, plainModPsi_device, plainMod_shoup_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, pad, poly_mod_len, plain_modulus);
    NTT8pointPerThread_kernel2<<<gridDim, blockDim, per_block_memory>>>
                                (device_a, plainModPsi_device, plainMod_shoup_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, second_radix_size, poly_mod_len, plain_modulus);
}

//bfv decrypt
void BFVContext::hps_decrypt_scale_and_round(uint64_tt *dst, uint64_tt *src, const cudaStream_t &stream)
{
    uint64_tt t = plain_modulus;
    uint64_tt gridDimGlb = N / blockDimGlb.x;
    if (qMSB + sizeQMSB < 52)
    {
        if ((qMSB + tMSB + sizeQMSB) < 52)
        {
			// puts("lazy");
            hps_decrypt_scale_and_round_kernel_small_lazy<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, t_QHatInv_mod_q_div_q_mod_t, t_QHatInv_mod_q_div_q_frac, t, N, q_num);
        }else
        {
			// puts("normal");
            hps_decrypt_scale_and_round_kernel_small<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, t_QHatInv_mod_q_div_q_mod_t, t_QHatInv_mod_q_div_q_mod_t_shoup,
                    t_QHatInv_mod_q_div_q_frac, t, N, q_num);
        }
    }else
    {
        // qMSB + sizeQMSB >= 52
        size_t qMSBHf = qMSB >> 1;
        if ((qMSBHf + tMSB + sizeQMSB) < 52)
        {
            hps_decrypt_scale_and_round_kernel_large_lazy<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, t_QHatInv_mod_q_div_q_mod_t, t_QHatInv_mod_q_div_q_frac,
                    t_QHatInv_mod_q_B_div_q_mod_t, t_QHatInv_mod_q_B_div_q_frac, t, N, q_num,
                    qMSBHf);
        }else
        {
            hps_decrypt_scale_and_round_kernel_large<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, src, t_QHatInv_mod_q_div_q_mod_t, t_QHatInv_mod_q_div_q_mod_t_shoup,
                    t_QHatInv_mod_q_div_q_frac, t_QHatInv_mod_q_B_div_q_mod_t,
                    t_QHatInv_mod_q_B_div_q_mod_t_shoup, t_QHatInv_mod_q_B_div_q_frac, t, N, q_num,
                    qMSBHf);
        }
    }
}

__host__ void BFVContext::FromNTTInplace_for_QR(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);
    const int per_thread_ntt_size = 8;
    const int second_radix_size = 256; 
    const int first_stage_radix_size = N / second_radix_size;//N1
    const int pad = 4;
    int block_size2 = (first_stage_radix_size / 8) * pad;
    int grid_size2 = N * mod_num / (8 * block_size2);
    dim3 gridDim2(grid_size2, poly_num);
    dim3 blockDim2(block_size2);
    // the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    INTT8pointPerThread_kernel1_for_QR<<<gridDim, blockDim, per_block_memory>>>
                            (device_a, qr_psiinv_table_device, qr_psiinv_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, second_radix_size, poly_mod_len);
    INTT8pointPerThread_kernel2_for_QR<<<gridDim2, blockDim2,
                            (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                            (device_a, qr_psiinv_table_device, qr_psiinv_shoup_table_device, n_inv_device_qr, n_inv_shoup_device_qr, poly_num, start_poly_idx, mod_num, start_mod_idx, N, first_stage_radix_size, pad, poly_mod_len);
}
__host__ void BFVContext::ToNTTInplace_for_QR(uint64_tt* device_a, int start_poly_idx, int start_mod_idx, int poly_num, int mod_num, int poly_mod_len)
{
    int block_size = 128;
    int grid_size = N * mod_num / (8 * block_size);
    dim3 gridDim(grid_size, poly_num);
    dim3 blockDim(block_size);//n1
    const int per_thread_ntt_size = 8;
    const int first_stage_radix_size = 256;//N1
    const int second_radix_size = N / first_stage_radix_size;
    const int pad = 4;// the same thread span，the same operation is N/8(8-point)/128(block threads' num)
    const int per_block_memory = blockDim.x * per_thread_ntt_size * sizeof(uint64_tt);
    NTT8pointPerThread_kernel1_for_QR<<<gridDim, (first_stage_radix_size / 8) * pad,
                              (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_tt)>>>
                             (device_a, qr_psi_table_device, qr_psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, pad, poly_mod_len);
    NTT8pointPerThread_kernel2_for_QR<<<gridDim, blockDim, per_block_memory>>>
                                (device_a, qr_psi_table_device, qr_psi_shoup_table_device, poly_num, start_poly_idx, mod_num, start_mod_idx, N,first_stage_radix_size, second_radix_size, poly_mod_len);
}