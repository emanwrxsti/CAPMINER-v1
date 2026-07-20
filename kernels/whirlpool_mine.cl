// OpenCL kernel skeleton named whirlpool_mine.
// Replace placeholder mix with exact CapStash Core Whirlpool PoW before real mining.
__kernel void whirlpool_mine(__global const uchar* header76,
                             uint start_nonce,
                             uint count,
                             __global uint* found,
                             __global uint* found_nonce,
                             __global uchar* found_hash) {
    uint idx = get_global_id(0);
    if(idx >= count || *found != 0) return;
    uint nonce = start_nonce + idx;
    uint s = 0xC0DEC0DEu;
    for(int i=0;i<76;i++) s = rotate(s ^ header76[i] ^ (uint)(i*0x9e3779b9u), 5u) + 0x7f4a7c15u;
    s ^= nonce;
    if(((s >> 16) & 0xffffu) == 0) {
        if(atomic_cmpxchg(found,0,1)==0) {
            *found_nonce = nonce;
            for(int i=0;i<32;i++) found_hash[i] = (uchar)((s >> ((i&3)*8)) ^ (i*37));
        }
    }
}
