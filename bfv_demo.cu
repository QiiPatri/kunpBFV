#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

#include "include/Plaintext.cuh"
#include "include/Ciphertext.cuh"
#include "include/BFVScheme.cuh"
// #include "include/bootstrapping/Bootstrapper.cuh"
#include "include/BFVcontext.cuh"
#include "include/Utils.cuh"

int main(int argc, char* argv[])
{
    // size_t poly_modulus_degree = 8192;
    // size_t poly_modulus_degree = 16384;
    size_t poly_modulus_degree = 32768;

    BFVContext context(poly_modulus_degree);
    // print_parameters(context);
    printf( "poly_modulus_degree : %lu\n" , context.N );
    printf( "plain_modulus : %llu\n" , context.plain_modulus );

    int N = context.N;
    int L = context.L;//q_num - 1
    int K = context.K;
    int slots = context.slots;
    int target_level = L;

    BFVScheme scheme(context);
    cout<<"Generate Scheme OK"<<endl;

    SecretKey sk(context);
    cout<<"Generate sk OK"<<endl;

    scheme.mallocMemory();
    scheme.addEncKey(sk);
    cout<<"Generate pk OK"<<endl;

    float gen_swk = 0;
    float enc = 0, dec = 0, ecd = 0, dcd = 0;
    float hadd = 0, cadd = 0, hmult = 0, cmult = 0;
    float temp = 0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
        scheme.addMultKey_23(sk);    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&temp, start, end);
    gen_swk = temp;
        scheme.addLeftRotKey_23(sk, 1);
    cout<<"Generate rlk OK"<<endl;

    uint64_tt* mes1;
    uint64_tt* mes2;
	mes1 = new uint64_tt[slots];
    mes2 = new uint64_tt[slots];

    uint64_tt input_mod = 800;
    // uint64_tt input_mod = context.plain_modulus;

    // for(size_t idx = 0; idx < slots; ++idx)
    // {
    //     mes1[idx] = 1;
    //     mes2[idx] = 1;
    // }
    for(size_t idx = 0; idx < slots; ++idx)
    {
        // mes1[idx] = ( idx * 17 + 28 ) % context.plain_modulus;
        mes1[idx] = ( idx + 1 ) % input_mod;
        // mes2[idx] = ( idx * 39 + 41 ) % context.plain_modulus;
        mes2[idx] = ( idx + 1 ) % input_mod;
    }
    // for(size_t idx = 6; idx < slots; ++idx)
    // {
    //     mes1[idx] = 0;
    //     mes2[idx] = 0;
    // }

    // printf("mes1 = [ ");
    // for( size_t idx = 0 ; idx <= 10 ; ++idx )
    // {
    //     printf( "%lld " , mes1[idx] );
    // }
    // printf(" ]\n");
    // printf("mes2 = [ ");
    // for( size_t idx = 0 ; idx <= 10 ; ++idx )
    // {
    //     printf( "%lld " , mes2[idx] );
    // }
    // printf(" ]\n");

    print_host__array(mes1, mes2, slots);

    // const 1 for add const
	uint64_tt* const_1;
	const_1 = new uint64_tt[context.slots];

    for(size_t idx = 0; idx < context.slots; ++idx)
    {
        const_1[idx] = 1;
    }
	uint64_tt* const_1_device;
	cudaMalloc(&const_1_device, sizeof(uint64_tt) * context.slots);
	cudaMemcpy(const_1_device, const_1, sizeof(uint64_tt) * context.slots, cudaMemcpyHostToDevice);

	Plaintext plain_const_1(N, L, L, context.slots);
	Ciphertext cipher_temp(N, L, L, context.slots);
    scheme.cipher_const_1 = cipher_temp;
	// context.encode(const_1_device , plain_const_1);
	// scheme.encryptMsg(scheme.cipher_const_1 , plain_const_1);

    uint64_tt* msg1;
    uint64_tt* msg2;
    cudaMalloc(&msg1, sizeof(uint64_tt) * slots);
    cudaMalloc(&msg2, sizeof(uint64_tt) * slots);
    
    Plaintext plain_m1(N, L, target_level, slots);
    Plaintext plain_m2(N, L, target_level, slots);

    Ciphertext c1(N, L, L, slots);
    Ciphertext c2(N, L, L, slots);
    Ciphertext c3(N, L, L, slots);

    Plaintext m1_dec(N, L, L, slots);
    Plaintext m2_dec(N, L, L, slots);
    Plaintext res1_dec(N, L, L, slots);
    Plaintext res2_dec(N, L, L, slots);
    Plaintext res3_dec(N, L, L, slots);

    size_t st_check_idx = 20;
    size_t ed_idx = 100;

    for( size_t T = 0 ; T < ed_idx ; ++T)
    {
        cudaMemcpy(msg1, mes1, sizeof(uint64_tt) * slots, cudaMemcpyHostToDevice);
        cudaMemcpy(msg2, mes2, sizeof(uint64_tt) * slots, cudaMemcpyHostToDevice);
    
        uint64_tt* msg_dec;
        cudaMalloc(&msg_dec, sizeof(uint64_tt) * slots);
        cudaMemset(msg_dec, 0, sizeof(uint64_tt) * slots);

        // scheme.encryptMsg(scheme.cipher_const_1 , plain_const_1);

        //==============================ENCODE==============================
        cudaEventRecord(start);
            context.encode(msg1 , plain_m1);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);

        if(T >= st_check_idx)
        {
            ecd += temp; 
        }
            context.encode(msg2 , plain_m2);
    
        //==============================ENCRYPT=============================
        cudaEventRecord(start);
            scheme.encryptMsg(c1 , plain_m1);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);

        if(T >= st_check_idx)
        {
            enc += temp; 
        }
            scheme.encryptMsg(c2 , plain_m2);
    
        //==============================CADD=================================
        cudaEventRecord(start);
            scheme.addConstAndEqual(c1, plain_m1);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        if(T >= st_check_idx)
        {
            cadd += temp; 
        }

        if( (T + 1) == ed_idx)
        {
            scheme.decryptMsg(res1_dec , sk , c1);
            context.decode(res1_dec , msg_dec );
            print_device_array_max(msg_dec , slots , "CADD  : " , 20);
        }

        //==============================HADD=================================
        cudaEventRecord(start);
            scheme.addAndEqual(c1 , c2);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        if(T >= st_check_idx)
        {
            hadd += temp; 
        }

        if( (T + 1) == ed_idx)
        {
            scheme.decryptMsg(res1_dec , sk , c1);
            context.decode(res1_dec , msg_dec );
            print_device_array_max(msg_dec , slots , "HADD  : " , 20);
        }
    
        //==============================CMULT================================
        cudaEventRecord(start);
            scheme.multConstAndEqual(c1, plain_m1);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        if(T >= st_check_idx)
        {
            cmult += temp; 
        }

        if( (T + 1) == ed_idx)
        {
            scheme.decryptMsg(res1_dec , sk , c1);
            context.decode(res1_dec , msg_dec );
            print_device_array_max(msg_dec , slots , "CMULT : " , 20);
        }
    
        //==============================HMULT================================
        cudaEventRecord(start);
            scheme.multAndEqual_23(c1, c2);     
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        if(T >= st_check_idx)
        {
            hmult += temp; 
        }

        if( (T + 1) == ed_idx)
        {
            scheme.decryptMsg(res1_dec , sk , c1);
            context.decode(res1_dec , msg_dec );
            print_device_array_max(msg_dec , slots , "HMULT : " , 20);
        }

        //==============================DECRYPT=============================
        cudaEventRecord(start);
            scheme.decryptMsg(res1_dec , sk , c1);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        if(T >= st_check_idx)
        {
            dec += temp; 
        }
        
        //==============================DECODE==============================
        cudaEventRecord(start);
            // context.decode(plain_m1 , msg_dec);
            context.decode(res1_dec , msg_dec );
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&temp, start, end);
        if(T >= st_check_idx)
        {
            dcd += temp; 
        }


        // if( (T + 1) == ed_idx)
        // {
        //     scheme.decryptMsg(res1_dec , sk , c1);
        //     context.decode(res1_dec , msg_dec );
        //     print_device_array_max(msg_dec , slots , "message1_dec" , 20);
        // }
        
        // print_check_mul( mes1 , mes2 , msg_dec , slots , context.plain_modulus);

        scheme.decryptMsg(res2_dec , sk , c2);
        context.decode(res2_dec , msg_dec );
        // print_device_array(msg_dec, slots, "message2_dec");

        // print_check_equal( context.first_error_idx , context.error_idx_num , context.encode_buffer_device , context.decode_buffer_device , slots);
        // printf( "first_error_idx : %u\n" , context.first_error_idx );
        // printf( "error_idx_num : %u\n" , context.error_idx_num );
        // print_device_array_max(context.encode_buffer_device + context.first_error_idx, slots - 1 , "encode_buffer" , 20);
        // print_device_array_max(context.decode_buffer_device + context.first_error_idx, slots - 1 , "decode_buffer" , 20);
    
        // scheme.decryptMsg(res3_dec , sk , c3);
        // context.decode(res3_dec , msg_dec );
        // print_device_array(msg_dec, slots, "message3_dec");

        // puts("=====================================================");
    }

    float ecd_avg = (ecd*1000) / (float)(ed_idx - st_check_idx);
    float dcd_avg = (dcd*1000) / (float)(ed_idx - st_check_idx);
    float enc_avg = (enc*1000) / (float)(ed_idx - st_check_idx);
    float dec_avg = (dec*1000) / (float)(ed_idx - st_check_idx);
    float hadd_avg = (hadd*1000) / (float)(ed_idx - st_check_idx);
    float cadd_avg = (cadd*1000) / (float)(ed_idx - st_check_idx);
    float hmult_avg = (hmult*1000) / (float)(ed_idx - st_check_idx);
    float cmult_avg = (cmult*1000) / (float)(ed_idx - st_check_idx);

    printf("gen swk: %f μs\n", gen_swk);
    printf("Time: encode: %f μs decode: %f μs\n", ecd_avg, dcd_avg);
    printf("Time: enc: %f μs dec: %f μs\n", enc_avg, dec_avg);
    printf("Time: hadd: %f μs cadd: %f μs\n", hadd_avg, cadd_avg);
    printf("Time: hmult: %f μs cmult: %f μs\n", hmult_avg, cmult_avg);

}