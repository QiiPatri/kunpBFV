#pragma once

#include <vector>
#include "uint128.cuh"
#include "uintarith_bfv.h"

//convert base from Q to P
class BaseConverter
{
private:
    size_t ibase_size;
    size_t obase_size;

    uint64_tt* ibase;
    uint64_tt* obase;
    uint64_tt* ibaseMu_high;
    uint64_tt* ibaseMu_low;
    uint64_tt* obaseMu_high;
    uint64_tt* obaseMu_low; 

    uint64_tt* QHatInvModq;
    uint64_tt* QHatInvModq_shoup;
    double* qinv;
    uint64_tt* qiHat_mod_pj;
    uint64_tt* alpha_Q_mod_pj;
    uint64_tt* negPQHatInvModq;
    uint64_tt* negPQHatInvModq_shoup;
    uint64_tt* QInvModp;
    uint64_tt* PModq;
    uint64_tt* PModq_shoup;

public:
    
    BaseConverter() = default;

    explicit BaseConverter( vector<uint64_tt> &ibase_ , vector<uint64_tt> &obase_ , vector<uint64_tt> &ibase_big_modulus_ , 
            vector<uint64_tt> &big_qiHat_, vector<uint64_tt> &QHatInvModq_, vector<uint64_tt> &QHatInvModq_shoup_ , 
            vector<uint64_tt> &obase_big_modulus_ , vector<double> &inv_ , 
            vector<uint64_tt> &MuVec_high_ , vector<uint64_tt> &MuVec_low_ , uint64_tt ibase_MuVec_idx_ , uint64_tt obase_MuVec_idx_ )
    {
        init(ibase_, obase_, ibase_big_modulus_, big_qiHat_, QHatInvModq_, QHatInvModq_shoup_, obase_big_modulus_, inv_,
            MuVec_high_, MuVec_low_, ibase_MuVec_idx_ , obase_MuVec_idx_);
    }

    inline auto get_ibase() const { return ibase; } 
    inline auto get_obase() const { return obase; } 
    inline auto get_ibase_Mu_high() const { return ibaseMu_high; }
    inline auto get_ibase_Mu_low() const { return ibaseMu_low; }
    inline auto get_obase_Mu_high() const { return obaseMu_high; }
    inline auto get_obase_Mu_low() const { return obaseMu_low; }
    inline auto get_QHatInvModq() const { return QHatInvModq; }

    void init( vector<uint64_tt> &ibase_ , vector<uint64_tt> &obase_ , vector<uint64_tt> &ibase_big_modulus_ , 
            vector<uint64_tt> &big_qiHat_, vector<uint64_tt> &QHatInvModq_, vector<uint64_tt> &QHatInvModq_shoup_ , 
            vector<uint64_tt> &obase_big_modulus_ , vector<double> &inv_ ,
            vector<uint64_tt> &MuVec_high_ , vector<uint64_tt> &MuVec_low_ , uint64_tt ibase_MuVec_idx_ , uint64_tt obase_MuVec_idx_  )
    {
        vector< vector<uint64_tt> > QHatModp_;
        vector< vector<uint64_tt> > alphaQModp_;
        vector<uint64_tt> negPQHatInvModq_;
        vector<uint64_tt> negPQHatInvModq_shoup_;
        vector< vector<uint64_tt> > QInvModp_;
        vector<uint64_tt> PModq_;
        vector<uint64_tt> PModq_shoup_;

        size_t size_Q = ibase_.size();
        size_t size_P = obase_.size();

        ibase_size = size_Q;
        obase_size = size_P;

        // Create the base-change matrix rows
        QHatModp_.resize(size_P);
        for (size_t j = 0; j < size_P; ++j)
        {
            QHatModp_[j].resize(size_Q);
            auto &pj = obase_[j];
            for (size_t i = 0; i < size_Q; ++i)
            {
                // Base-change matrix contains the punctured products of ibase elements modulo the obase
                QHatModp_[j][i] = modulo_uint(big_qiHat_.data() + i * size_Q, size_Q, pj);
            }
        }

        alphaQModp_.resize(size_Q + 1);
        for (size_t j = 0; j < size_P; ++j)
        {
            auto &pj = obase_[j];
            uint64_tt big_Q_mod_pj = modulo_uint(ibase_big_modulus_.data(), size_Q, pj);
            for (size_t alpha = 0; alpha < size_Q + 1; ++alpha)
            {
                alphaQModp_[alpha].push_back(multiply_uint_mod(alpha, big_Q_mod_pj, pj));
            }
        }

        negPQHatInvModq_.resize(size_Q);
        negPQHatInvModq_shoup_.resize(size_Q);
        PModq_.resize(size_Q);
        PModq_shoup_.resize(size_Q);

        for (size_t i = 0; i < size_Q; ++i)
        {
            auto &qi = ibase_[i];
            auto QHatInvModqi = QHatInvModq_[i];
            uint64_tt PModqi = modulo_uint(obase_big_modulus_.data(), size_P, qi);
            PModq_[i] = PModqi;
            PModq_shoup_[i] = compute_shoup(PModqi, qi);
            uint64_tt PQHatInvModqi = multiply_uint_mod(PModqi, QHatInvModqi, qi);
            uint64_tt negPQHatInvModqi = qi - PQHatInvModqi;
            negPQHatInvModq_[i] = negPQHatInvModqi;
            negPQHatInvModq_shoup_[i] = compute_shoup(negPQHatInvModqi, qi);
        }

        QInvModp_.resize(size_P);
        for (size_t j = 0; j < size_P; ++j)
        {
            QInvModp_[j].resize(size_Q);
            auto &pj = obase_[j];
            for (size_t i = 0; i < size_Q; ++i)
            {
                auto &qi = ibase_[i];
                if (!try_invert_uint_mod(qi, pj, QInvModp_[j][i]))
                {
                    throw logic_error("invalid rns bases in computing QInvModp");
                }
            }
        }

        //device to host
        cudaMalloc(&ibase, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(ibase, ibase_.data(), ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&obase, obase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(obase, obase_.data(), obase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&QHatInvModq, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(QHatInvModq, QHatInvModq_.data(), ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&QHatInvModq_shoup, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(QHatInvModq_shoup, QHatInvModq_shoup_.data(), ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&qinv , ibase_.size() * sizeof(double));
        cudaMemcpyAsync(qinv, inv_.data(), ibase_.size() * sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&qiHat_mod_pj, obase_.size() * ibase_.size() * sizeof(uint64_tt));
        for (size_t idx = 0; idx < obase_.size(); idx++)
            cudaMemcpyAsync(qiHat_mod_pj + idx * ibase_.size(), QHatModp_[idx].data(),
                ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&alpha_Q_mod_pj, (ibase_.size() + 1) * obase_.size() * sizeof(uint64_tt));
        for (size_t idx = 0; idx < ibase_.size() + 1; idx++)
            cudaMemcpyAsync(alpha_Q_mod_pj + idx * obase_.size(), alphaQModp_[idx].data(),
                            obase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&negPQHatInvModq, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(negPQHatInvModq, negPQHatInvModq_.data(),
                        ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        // print_device_array(negPQHatInvModq, ibase_.size(), "negPQHatInvModq");

        cudaMalloc(&negPQHatInvModq_shoup, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(negPQHatInvModq_shoup, negPQHatInvModq_shoup_.data(),
                        ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&QInvModp, obase_.size() * ibase_.size() * sizeof(uint64_tt));
        for (size_t idx = 0; idx < obase_.size(); idx++)
            cudaMemcpyAsync(QInvModp + idx * ibase_.size(), QInvModp_[idx].data(),
                            ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        // print_device_array(QInvModp, ibase_.size(), "QInvModp");

        cudaMalloc(&PModq, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(PModq, PModq_.data(), ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        // print_device_array(PModq, ibase_.size() , "PModq");

        cudaMalloc(&PModq_shoup, ibase_.size() * sizeof(uint64_tt));
        cudaMemcpyAsync(PModq_shoup, PModq_shoup_.data(), ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice);

        cudaMalloc(&ibaseMu_high, ibase_.size() * sizeof(uint64_tt) );
        cudaMemcpyAsync(ibaseMu_high, &MuVec_high_[ibase_MuVec_idx_], ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice );
        cudaMalloc(&ibaseMu_low, ibase_.size() * sizeof(uint64_tt) );
        cudaMemcpyAsync(ibaseMu_low, &MuVec_low_[ibase_MuVec_idx_], ibase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice );
    
        cudaMalloc(&obaseMu_high, obase_.size() * sizeof(uint64_tt) );
        cudaMemcpyAsync(obaseMu_high, &MuVec_high_[obase_MuVec_idx_], obase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice );
        cudaMalloc(&obaseMu_low, obase_.size() * sizeof(uint64_tt) );
        cudaMemcpyAsync(obaseMu_low, &MuVec_low_[obase_MuVec_idx_], obase_.size() * sizeof(uint64_tt), cudaMemcpyHostToDevice );
    }

    void bConv_HPS(uint64_tt *dst, const uint64_tt *src, size_t n, const cudaStream_t &stream) const
    {
        uint64_tt *temp;
        cudaMalloc(&temp, ibase_size * n * sizeof(uint64_tt));
    
        constexpr int unroll_factor = 2;
    
        uint64_tt gridDimGlb = ibase_size * n / unroll_factor / blockDimGlb.x;
        bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(temp, src, QHatInvModq,
                                                                          QHatInvModq_shoup, ibase,
                                                                          ibase_size, n);
    
        gridDimGlb = obase_size * n / unroll_factor / blockDimGlb.x;
        base_convert_matmul_hps_unroll2_kernel<<<
        gridDimGlb, blockDimGlb, sizeof(uint64_t) * obase_size * ibase_size, stream>>>(
                dst, temp, qiHat_mod_pj, alpha_Q_mod_pj, qinv, obaseMu_high ,obaseMu_low , ibase, ibase_size, obase,
                obase_size, n);
    }
};