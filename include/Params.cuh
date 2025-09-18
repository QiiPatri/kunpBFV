#pragma once

#include "BFVcontext.h"

using namespace std;

void BFVContext::getPrimeBFV()
{
	printf("logN = %d\n", logN);

	switch (logN)
	{
	// case 13:
	// 	plain_modulus = 786433;

	// 	qVec = {
	// 		0xffff00001, 0xfff700001, 0xfff100001,   0xffdf00001,
	// 	};
	// 	pVec = { // 36 x 6
	// 		0x10a19000001, 0x10a3b000001
	// 	};
	// 	tVec = {
	// 		0xffffffffffc0001, 0xfffffffff840001,
	// 		0xfffffffff6a0001, 0xfffffffff5a0001
	// 	};
	
    //     gamma = 2;
	// 	// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
	// 	break;

	case 14:
		plain_modulus = 23068673;

		qVec = {
			0x7FFE28001, 0x7FFF18001, 0x7FFF80001,	0x7FFFB0001
		};
		pVec = { // 36 x 6
			0xFFFFCA8001, 0xFFFFE80001
		};
		tVec = {
			0xffffffffffc0001, 0xfffffffff840001,
			0xfffffffff6a0001, 0xfffffffff5a0001
		};
	
        gamma = 2;
		// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
		break;

	// case 14:

	// 	plain_modulus = 23068673;

    //     qVec = {
	// 		0x4001b00001, 
	// 		0xfff9c0001, 0xfff8e0001, 0xfff840001,	0xfff700001,
	// 		0xfff640001, 
	// 		0x1000a20001, 
	// 		0x1000b40001, 0x1000f60001,	0x10011a0001,// 0x1001220001,
    //     };
	// 	pVec = { // 36 x 6
    //         0x10a19000001, 0x10a3b000001,// 0x10a83000001,
	// 	};
	// 	tVec = { // 60 x 8
	// 		0xffffffffffc0001, 0xfffffffff840001,
	// 		0xfffffffff6a0001, 0xfffffffff5a0001,
	// 		// 0xfffffffff2a0001,// 0xfffffffff240001,
	// 	};
    //     gamma = 3;
	// 	// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
	// 	break;

	// case 15:
	// 	// plain_modulus = 23068673;
	// 	plain_modulus = 786433;
	// 	// plain_modulus = 5767169;

    //     qVec = {
	// 		// 0x1004d40001,
	// 		// 0x3004d80001,
	// 		/*0x4001b00001,*/ 0xffff00001, 0xfff700001, 0xfff100001, 0xffdf00001,
	// 		// 0x1000500001, 0x1001d00001, 0x1002300001, 0x1002700001, 
	// 		// /*0xffdf00001,*/ 0xffc300001, 0xffbe00001, 0xffbb00001, 
	// 		// 0x1002c00001, 0x1003600001, 0x1003900001 //,0x1003e00001, 

	// 		// 0x1004700001, 0x1004d00001, 0x1005100001, 0x1006900001, 
	// 		// 0xffa300001, 0xff7900001, 0xff6a00001, 0xff5700001, 
	// 		// 0xff3d00001, 0xff3c00001, 0xff3a00001, 0xff3900001,
	// 		// 0x1007400001, 0x1008700001, 0x1008d00001, 0x1009300001, 
			
	// 		// 0xff3300001, 0xff2d00001, 0xff2800001, 0xff1f00001,
    //     };
	// 	pVec = { // 36 x 6
    //         0x10a19000001, 0x10a3b000001, 0x10a83000001, 0x10b0a000001
	// 		// 0x10b2e000001, 0x10bc1000001,
	// 		// 0x10a19000001, 0x10a3b000001, 0x10a83000001, 0x1000b40001,
	// 		// 0x1000f60001, 0x10011a0001, 0x1001220001
	// 	};
	// 	tVec = { // 60 x 8
	// 		0xffffffffffc0001, 0xfffffffff840001,
	// 		0xfffffffff6a0001, 0xfffffffff5a0001,
	// 		0xfffffffff2a0001, 0xfffffffff240001,
	// 		0xffffffffefe0001, 0xffffffffeca0001,
	// 		// 0x7ffffffffcc0001, 0x7ffffffffba0001,
	// 		// 0x7ffffffffb00001, 0x7ffffffff320001,
	// 		// 0x7ffffffff2c0001, 0x7ffffffff240001,
	// 		// 0x7fffffffefa0001, 0x7fffffffede0001,
			
	// 		// 0x7fffffffe900001, 0x7fffffffe3c0001,
	// 		// 0x2ffffffee0001,0x2ffffffe00001,0x2ffffffd60001,0x2ffffffa40001,
	// 		// 0x2ffffff6a0001,0x2ffffff620001,0x2fffffefc0001,0x2fffffef60001,
	// 		// 0xfffffffa0001, 0xfffffff00001, 0xffffffde0001, 0xffffff6a0001,
	// 		// 0xffffff280001, 0xffffff060001, 0xfffffed60001, 0xfffffebc0001,
	// 		// 0xfffffe8e0001,
	// 	};
    //     gamma = 4;//8;
	// 	// len(P) stands for r, len(T) stands for r', gamma stands for tilde_r
	// 	break;

	//parameters in BGV
	case 15:
		plain_modulus = 0xC0001; 
		qVec = {
			36028797001138177, 36028797003563009, 36028797003694081, 36028797005135873, 36028797005529089, 36028797005856769,
			36028797009985537, 36028797010444289, 36028797012606977, 36028797013000193, 36028797013327873, 36028797014376449,
			36028797014573057, 36028797014704129, 36028797017456641
		};
		// qVec = {
		// 	0x7FFE0AAA8
		// 	, 0x7FFECAAA9, 0x7FFF8AAAA,	0x80004AAAB
		// };
		pVec = { // 40 x 2
			72057594037338113
		};
		tVec = { 
			0xffffffffffc0001, 0xfffffffff840001,
			0xfffffffff6a0001, 0xfffffffff5a0001,
			// 0xfffffffff2a0001, 0xfffffffff240001,
			// 0xffffffffefe0001, 0xffffffffeca0001,
			// 0x7ffffffffcc0001, 0x7ffffffffba0001,
			// 0x7ffffffffb00001, 0x7ffffffff320001,
			// 0x7ffffffff2c0001, 0x7ffffffff240001,
			// 0x7fffffffefa0001, 0x7fffffffede0001,
		};
		gamma = 4;
		break;

	default:
		break;
	}
}