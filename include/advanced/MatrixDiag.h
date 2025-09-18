#pragma once

#include <vector>
#include "../uint128.cuh"

using namespace std;

class MatrixDiag
{
public:
    MatrixDiag();
    MatrixDiag(int N, int diag_num, int bsgs_ratio, int diag_gap, int gs, int bs): N(N), diag_num(diag_num), bsgs_ratio(bsgs_ratio), diag_gap(diag_gap), gs(gs), bs(bs){}
    int N;
    int diag_num;
    // n1 / n2
    int bsgs_ratio;
    // giant step
	int gs;
    // baby stepc
    int bs;
    // gap of diagonals
    int diag_gap;

    // host
    //vector<vector<cuDoubleComplex>> diag_vec;
    // diag_num * t_num * N
    // device
    uint64_tt* diag_inv_vec_encodePQl_device;
    // host
    int cipher_first_rot_idx;
};