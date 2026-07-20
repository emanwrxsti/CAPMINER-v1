#pragma once
#include <stdint.h>
#include "cuda_whirlpool_tables.cuh"

__device__ __forceinline__ unsigned long long dec64le_cuda(const unsigned char* p) {
    return ((unsigned long long)p[0])
        | ((unsigned long long)p[1] << 8)
        | ((unsigned long long)p[2] << 16)
        | ((unsigned long long)p[3] << 24)
        | ((unsigned long long)p[4] << 32)
        | ((unsigned long long)p[5] << 40)
        | ((unsigned long long)p[6] << 48)
        | ((unsigned long long)p[7] << 56);
}

__device__ __forceinline__ void enc64le_cuda(unsigned char* p, unsigned long long x) {
    p[0] = (unsigned char)(x);
    p[1] = (unsigned char)(x >> 8);
    p[2] = (unsigned char)(x >> 16);
    p[3] = (unsigned char)(x >> 24);
    p[4] = (unsigned char)(x >> 32);
    p[5] = (unsigned char)(x >> 40);
    p[6] = (unsigned char)(x >> 48);
    p[7] = (unsigned char)(x >> 56);
}

__device__ __forceinline__ unsigned int BCUDA(unsigned long long x, int n) {
    return (unsigned int)((x >> (8 * n)) & 0xffULL);
}

#define WROUND_ELT(in, i0,i1,i2,i3,i4,i5,i6,i7) \
    (plain_T0[BCUDA(in##i0,0)] ^ plain_T1[BCUDA(in##i1,1)] ^ \
     plain_T2[BCUDA(in##i2,2)] ^ plain_T3[BCUDA(in##i3,3)] ^ \
     plain_T4[BCUDA(in##i4,4)] ^ plain_T5[BCUDA(in##i5,5)] ^ \
     plain_T6[BCUDA(in##i6,6)] ^ plain_T7[BCUDA(in##i7,7)])

#define WROUND(in,out,c0,c1,c2,c3,c4,c5,c6,c7) do { \
    out##0 = WROUND_ELT(in,0,7,6,5,4,3,2,1) ^ (c0); \
    out##1 = WROUND_ELT(in,1,0,7,6,5,4,3,2) ^ (c1); \
    out##2 = WROUND_ELT(in,2,1,0,7,6,5,4,3) ^ (c2); \
    out##3 = WROUND_ELT(in,3,2,1,0,7,6,5,4) ^ (c3); \
    out##4 = WROUND_ELT(in,4,3,2,1,0,7,6,5) ^ (c4); \
    out##5 = WROUND_ELT(in,5,4,3,2,1,0,7,6) ^ (c5); \
    out##6 = WROUND_ELT(in,6,5,4,3,2,1,0,7) ^ (c6); \
    out##7 = WROUND_ELT(in,7,6,5,4,3,2,1,0) ^ (c7); \
} while(0)

#define WTRANSFER(dst, src) do { \
    dst##0=src##0; dst##1=src##1; dst##2=src##2; dst##3=src##3; \
    dst##4=src##4; dst##5=src##5; dst##6=src##6; dst##7=src##7; \
} while(0)

__device__ void whirlpool_compress_cuda(const unsigned char block[64], unsigned long long state[8]) {
    unsigned long long n0,n1,n2,n3,n4,n5,n6,n7;
    unsigned long long sn0,sn1,sn2,sn3,sn4,sn5,sn6,sn7;
    unsigned long long h0,h1,h2,h3,h4,h5,h6,h7;
    unsigned long long t0,t1,t2,t3,t4,t5,t6,t7;

    sn0 = n0 = dec64le_cuda(block + 0);
    sn1 = n1 = dec64le_cuda(block + 8);
    sn2 = n2 = dec64le_cuda(block + 16);
    sn3 = n3 = dec64le_cuda(block + 24);
    sn4 = n4 = dec64le_cuda(block + 32);
    sn5 = n5 = dec64le_cuda(block + 40);
    sn6 = n6 = dec64le_cuda(block + 48);
    sn7 = n7 = dec64le_cuda(block + 56);

    h0=state[0]; h1=state[1]; h2=state[2]; h3=state[3];
    h4=state[4]; h5=state[5]; h6=state[6]; h7=state[7];

    n0 ^= h0; n1 ^= h1; n2 ^= h2; n3 ^= h3;
    n4 ^= h4; n5 ^= h5; n6 ^= h6; n7 ^= h7;

    #pragma unroll
    for(int r=0; r<10; ++r) {
        WROUND(h,t,plain_RC[r],0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL);
        WTRANSFER(h,t);
        WROUND(n,t,h0,h1,h2,h3,h4,h5,h6,h7);
        WTRANSFER(n,t);
    }

    state[0] ^= n0 ^ sn0;
    state[1] ^= n1 ^ sn1;
    state[2] ^= n2 ^ sn2;
    state[3] ^= n3 ^ sn3;
    state[4] ^= n4 ^ sn4;
    state[5] ^= n5 ^ sn5;
    state[6] ^= n6 ^ sn6;
    state[7] ^= n7 ^ sn7;
}

// Compute the Whirlpool key schedule (round keys K_1..K_10) from the midstate,
// using the SAME round function as whirlpool_compress_cuda above -- so the keys
// are bit-identical to the per-nonce inline schedule. out_rk[r*8 + j] = K_{r+1}[j].
__device__ void whirlpool_round_keys_cuda(const unsigned long long mid[8],
                                          unsigned long long out_rk[80]) {
    unsigned long long h0=mid[0],h1=mid[1],h2=mid[2],h3=mid[3],
                       h4=mid[4],h5=mid[5],h6=mid[6],h7=mid[7];
    unsigned long long t0,t1,t2,t3,t4,t5,t6,t7;
    #pragma unroll
    for(int r=0; r<10; ++r) {
        WROUND(h,t, plain_RC[r],0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL);
        WTRANSFER(h,t);
        out_rk[r*8+0]=h0; out_rk[r*8+1]=h1; out_rk[r*8+2]=h2; out_rk[r*8+3]=h3;
        out_rk[r*8+4]=h4; out_rk[r*8+5]=h5; out_rk[r*8+6]=h6; out_rk[r*8+7]=h7;
    }
}

__device__ void cap_whirlpool80_cuda(const unsigned char header80[80], unsigned char out32[32]) {
    unsigned long long state[8] = {0,0,0,0,0,0,0,0};

    unsigned char block0[64];
    unsigned char block1[64];

    #pragma unroll
    for(int i=0;i<64;i++) block0[i] = header80[i];

    #pragma unroll
    for(int i=0;i<64;i++) block1[i] = 0;

    #pragma unroll
    for(int i=0;i<16;i++) block1[i] = header80[64+i];

    block1[16] = 0x80;

    // Whirlpool/Sphlib uses 256-bit big-endian length at end.
    // 80 bytes = 640 bits. Put low 64-bit length at final 8 bytes.
    block1[56] = 0x00;
    block1[57] = 0x00;
    block1[58] = 0x00;
    block1[59] = 0x00;
    block1[60] = 0x00;
    block1[61] = 0x00;
    block1[62] = 0x02;
    block1[63] = 0x80;

    whirlpool_compress_cuda(block0, state);
    whirlpool_compress_cuda(block1, state);

    unsigned char wh[64];
    #pragma unroll
    for(int i=0;i<8;i++) enc64le_cuda(wh + i*8, state[i]);

    #pragma unroll
    for(int i=0;i<32;i++) out32[i] = wh[i] ^ wh[i+32];
}

#undef WROUND_ELT
#undef WROUND
#undef WTRANSFER
