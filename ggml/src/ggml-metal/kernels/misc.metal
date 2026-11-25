#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_argmax_part_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * tmp,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row  = tgpig.y;
    const int part = tgpig.x;
    const int ne00 = (int) args.ne00;
    const int npart = (int) args.npart;
    const int chunk = (ne00 + npart - 1) / npart;
    const int start = part * chunk;
    const int end   = min(start + chunk, ne00);

    device const float * x_row = (device const float *) ((device const char *) src0 + row * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = start + tpitg.x; i00 < end; i00 += ntg.x) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        device float * tmp_max = (device float *) tmp;
        device int32_t * tmp_arg = (device int32_t *) tmp + npart * (int) args.ne01;

        tmp_max[row*npart + part] = max_val;
        tmp_arg[row*npart + part] = arg_val;
    }
}

kernel void kernel_argmax_final_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * tmp,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row = tgpig.x;
    const int npart = (int) args.npart;

    device const float * tmp_max = (device const float *) tmp;
    device const int32_t * tmp_arg = (device const int32_t *) tmp + npart * (int) args.ne01;

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i = tpitg.x; i < npart; i += ntg.x) {
        if (tmp_max[row*npart + i] > lmax) {
            lmax = tmp_max[row*npart + i];
            larg = tmp_arg[row*npart + i];
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        ((device int32_t *) dst)[row] = arg_val;
    }
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

constant bool FC_pad_circular [[function_constant(FC_PAD + 0)]];

// circular means on a torus, so the coordinates wrap around
static inline int32_t wrap_around(int32_t coord, int32_t size) {
    return (coord + size) % size;
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t ne00 = args.ne00;
    const int32_t ne01 = args.ne01;
    const int32_t ne02 = args.ne02;
    const int32_t ne03 = args.ne03;

    int32_t i01 = i1 - args.lp1;
    int32_t i02 = i2 - args.lp2;
    int32_t i03 = i3 - args.lp3;

    if (FC_pad_circular) {
        i01 = wrap_around(i01, ne01);
        i02 = wrap_around(i02, ne02);
        i03 = wrap_around(i03, ne03);
    }

    device float * dst_ptr = (device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    // the row lies in the padded region, so no source row backs it
    if (i01 < 0 || i01 >= ne01 ||
        i02 < 0 || i02 >= ne02 ||
        i03 < 0 || i03 >= ne03) {
        for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
            const int32_t i0 = k0*1024 + tpitg.x + l0;
            if (i0 >= args.ne0) {
                break;
            }

            dst_ptr[i0] = 0.0f;
        }

        return;
    }

    device const char * src0_row = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        int32_t i00 = i0 - args.lp0;

        if (FC_pad_circular) {
            i00 = wrap_around(i00, ne00);
        }

        dst_ptr[i0] = i00 >= 0 && i00 < ne00 ? *((device const float *) (src0_row + i00*args.nb00)) : 0.0f;
    }
}

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    using T4 = typename value_type_4<T>::type;

    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N, typename src_t>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NW + lane])*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = val2 - val + 2*((lane & i) == 0)*val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
// those up to NT go through threadgroup memory; the rest stay in registers.
// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
// TODO: try to unroll loops
template<int N, int NT, typename src_t>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NT + tid])*scale;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_f32_t kernel_fwht<64,  float>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_f32_t kernel_fwht<128, float>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_f32_t kernel_fwht<256, float>;

template [[host_name("kernel_fwht_f16_64")]]  kernel kernel_fwht_f16_t kernel_fwht<64,  half>;
template [[host_name("kernel_fwht_f16_128")]] kernel kernel_fwht_f16_t kernel_fwht<128, half>;
template [[host_name("kernel_fwht_f16_256")]] kernel kernel_fwht_f16_t kernel_fwht<256, half>;

template [[host_name("kernel_fwht_f32_512")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_2048")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_4096")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_8192")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;

template [[host_name("kernel_fwht_f16_512")]]  kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_1024")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_2048")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_4096")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_8192")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        const float wv = *(device const float *) (weights + ih*args.nb_w0 + it*args.nb_w1);
        result = fma(xv, wv, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_post_nocomb_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float post_lane = 0.0f;
    if (tiisg < hc) {
        post_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    }

    float post_reg[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(post_lane, idst);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        const float rv = *(device const float *) (rb + idst*args.nb_r1);
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = xv*post_reg[idst] + rv;
    }
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

constant bool FC_top_k_mask_has_score  [[function_constant(FC_TOP_K_MASK + 0)]];
constant bool FC_top_k_mask_has_blocks [[function_constant(FC_TOP_K_MASK + 1)]];
constant bool FC_top_k_mask_has_extra  [[function_constant(FC_TOP_K_MASK + 2)]];
constant bool FC_top_k_mask_vec        [[function_constant(FC_TOP_K_MASK + 3)]];

// the fill runs before the scatter, both are fully parallel over the whole dst and over the picks of one row
template <typename T>
kernel void kernel_top_k_mask_fill(
        constant ggml_metal_kargs_top_k_mask & args,
        device char * dst,
        uint  tid[[thread_position_in_grid]],
        uint ntot[[threads_per_grid]]) {
    using T4 = typename value_type_4<T>::type;

    const T ninf = T(-INFINITY);

    if (FC_top_k_mask_vec) {
        device T4 * dst_4 = (device T4 *) dst;
        for (int64_t i = tid; i < args.n_elems/4; i += ntot) {
            dst_4[i] = T4(ninf);
        }
    } else {
        device T * dst_t = (device T *) dst;
        for (int64_t i = tid; i < args.n_elems; i += ntot) {
            dst_t[i] = ninf;
        }
    }
}

typedef decltype(kernel_top_k_mask_fill<half>)  kernel_top_k_mask_fill_h_t;
typedef decltype(kernel_top_k_mask_fill<float>) kernel_top_k_mask_fill_f_t;

template [[host_name("kernel_top_k_mask_fill_f16")]] kernel kernel_top_k_mask_fill_h_t kernel_top_k_mask_fill<half>;
template [[host_name("kernel_top_k_mask_fill_f32")]] kernel kernel_top_k_mask_fill_f_t kernel_top_k_mask_fill<float>;

// one threadgroup per row: copy the mask cells of the live picks and of the extra cells over the fill
template <typename T>
kernel void kernel_top_k_mask(
        constant ggml_metal_kargs_top_k_mask & args,
        device const  char * mask,
        device const  char * top_k,
        device const  char * score,
        device const  char * blocks,
        device const  char * extra,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3 ntg_max[[threads_per_threadgroup]]) {
    using T4 = typename value_type_4<T>::type;

    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    const int   tiitg = tpitg.x;
    const int     ntg = ntg_max.x;

    device const char * mask_row  = mask  + (i1 % args.m1)*args.nbm1 + (i2 % args.m2)*args.nbm2 + (i3 % args.m3)*args.nbm3;
    device       char * dst_row   = dst   + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;
    device const char * top_k_row = top_k + i1*args.nbk1 + i2*args.nbk2 + i3*args.nbk3;

    device const T * mask_row_t = (device const T *) mask_row;
    device       T * dst_row_t  = (device       T *) dst_row;

    for (int i = tiitg; i < args.n_top; i += ntg) {
        const int32_t p = *(device const int32_t *) (top_k_row + i*args.nbk0);
        if (p < 0 || p >= args.n_pick) {
            continue;
        }
        if (FC_top_k_mask_has_score && !(*(device const float *) (score + p*args.nbs0 + i1*args.nbs1 + i2*args.nbs2 + i3*args.nbs3) > -INFINITY)) {
            continue;
        }
        if (FC_top_k_mask_has_blocks) {
            FOR_UNROLL (int k = 0; k < args.n_blk; ++k) {
                const int32_t c = *(device const int32_t *) (blocks + k*args.nbb0 + p*args.nbb1);
                if (c >= 0 && c < args.n_kv) {
                    dst_row_t[c] = mask_row_t[c];
                }
            }
        } else if (p < args.n_kv) {
            dst_row_t[p] = mask_row_t[p];
        }
    }

    if (FC_top_k_mask_has_extra) {
        for (int i = tiitg; i < args.n_extra; i += ntg) {
            const int32_t c = *(device const int32_t *) (extra + i*args.nbe0 + i1*args.nbe1 + i2*args.nbe2 + i3*args.nbe3);
            if (c >= 0 && c < args.n_kv) {
                dst_row_t[c] = mask_row_t[c];
            }
        }
    }
}

typedef decltype(kernel_top_k_mask<half>)  kernel_top_k_mask_h_t;
typedef decltype(kernel_top_k_mask<float>) kernel_top_k_mask_f_t;

template [[host_name("kernel_top_k_mask_f16")]] kernel kernel_top_k_mask_h_t kernel_top_k_mask<half>;
template [[host_name("kernel_top_k_mask_f32")]] kernel kernel_top_k_mask_f_t kernel_top_k_mask<float>;

// softmax attention of the queries over the stack rows s_0..s_{k-1} and s_k = rms_norm(x), one threadgroup per token
// dst[:, iq] = sum_j w_j*s_j with w = softmax_j(scale*dot(s_j, q_iq))
// situ(g, u) = (s1*tanh(s0*g + b0) + b1)*sigmoid(g) * (s3*tanh(s2*u + b2) + b3), 4 values per thread
// each step is the same float4 math as the SCALE, UNARY and MUL kernels; OPQ keeps each step result as is
// (args.zero is 0 at run time, so the compiler cannot merge two steps), so the result is bitwise the unfused one
kernel void kernel_situ_f32(
        constant ggml_metal_kargs_situ & args,
        device const char * g,
        device const char * u,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i0 = tgpig.x*ntg.x + tpitg.x;
    if (4*i0 >= args.ne0) {
        return;
    }

    const int i1 = tgpig.y;
    const int i2 = tgpig.z;

    const float4 x = ((device const float4 *) (g + i2*args.nbg2 + i1*args.nbg1))[i0];
    const float4 y = ((device const float4 *) (u + i2*args.nbu2 + i1*args.nbu1))[i0];

    const uint4 z = uint4(args.zero);
#define OPQ(v) as_type<float4>(as_type<uint4>(v) ^ z)
    const float4 t0 = OPQ(args.s[0] * x + args.b[0]);
    const float4 t1 = OPQ(precise::tanh(t0));
    const float4 t2 = OPQ(args.s[1] * t1 + args.b[1]);
    const float4 sg = OPQ(1 / (1 + exp(-x)));
    const float4 a  = OPQ(t2 * sg);

    const float4 v0 = OPQ(args.s[2] * y + args.b[2]);
    const float4 v1 = OPQ(precise::tanh(v0));
    const float4 v2 = OPQ(args.s[3] * v1 + args.b[3]);
#undef OPQ

    ((device float4 *) (dst + i2*args.nb2 + i1*args.nb1))[i0] = a * v2;
}

kernel void kernel_depth_mix_f32(
        constant ggml_metal_kargs_depth_mix & args,
        device const char * x,
        device const char * prev,
        device const char * q,
        device       char * dst,
        threadgroup float * shmem [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    constexpr short NS = GGML_METAL_DEPTH_MIX_NS;
    constexpr short NQ = GGML_METAL_DEPTH_MIX_NQ;

    const short nsg = ntg/32;
    const int   n4  = args.n/4;
    const int   k   = args.k;
    const int   nq  = args.nq;

    device const float4 * x4 = (device const float4 *) (x + (uint64_t) tgpig*args.nbx2);
    device const char   * pt = prev + (uint64_t) tgpig*args.nbp2;
    device const char   * qt = q    + (uint64_t) tgpig*args.nbq2;

    // the threadgroup covers the row in one pass when n/4 <= 1024
    const bool on = tpitg < n4;

    float4 s[NS];
    float4 qv[NQ];
    FOR_UNROLL (short j = 0; j < NS; ++j) {
        s[j] = 0.0f;
        if (on && j < k) {
            s[j] = ((device const float4 *) (pt + j*args.nbp1))[tpitg];
        }
    }
    FOR_UNROLL (short iq = 0; iq < NQ; ++iq) {
        qv[iq] = 0.0f;
        if (on && iq < nq) {
            qv[iq] = ((device const float4 *) (qt + iq*args.nbq1))[tpitg];
        }
    }
    const float4 xv = on ? x4[tpitg] : 0.0f;

    float ss = simd_sum(dot(xv, xv));
    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ss = 0.0f;
    for (short i = 0; i < nsg; ++i) {
        ss += shmem[i];
    }
    const float4 y = xv*(1.0f/sqrt(ss/args.n + args.eps));

    // dot products, lane v of each simdgroup then holds the partial sum of pair v = j*NQ + iq
    threadgroup float * sd = shmem + nsg;
    FOR_UNROLL (short j = 0; j < NS; ++j) {
        if (j <= k) {
            FOR_UNROLL (short iq = 0; iq < NQ; ++iq) {
                if (iq < nq) {
                    const float sv = simd_sum(dot(j < k ? s[j] : y, qv[iq]));
                    if (tiisg == 0) {
                        sd[(j*NQ + iq)*nsg + sgitg] = sv;
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // lane v sums pair v over the simdgroups, then the softmax runs over the lanes of each query
    const short v  = tiisg;
    const short jv = v/NQ;
    const bool  ok = jv <= k && v%NQ < nq;

    float wv = -INFINITY;
    if (ok) {
        float sv = 0.0f;
        for (short g = 0; g < nsg; ++g) {
            sv += sd[v*nsg + g];
        }
        wv = sv*args.scale;
    }

    float m = wv;
    FOR_UNROLL (short j = 0; j < NS; ++j) {
        m = max(m, simd_shuffle(wv, (ushort) (j*NQ + v%NQ)));
    }
    wv = ok ? exp(wv - m) : 0.0f;

    float sum = 0.0f;
    FOR_UNROLL (short j = 0; j < NS; ++j) {
        sum += simd_shuffle(wv, (ushort) (j*NQ + v%NQ));
    }
    wv *= 1.0f/sum;

    // all lanes take part in the shuffles
    float w[NS*NQ];
    FOR_UNROLL (short i = 0; i < NS*NQ; ++i) {
        w[i] = simd_shuffle(wv, (ushort) i);
    }

    if (!on) {
        return;
    }

    device char * dt = dst + (uint64_t) tgpig*args.nb2;

    FOR_UNROLL (short iq = 0; iq < NQ; ++iq) {
        if (iq < nq) {
            float4 o = 0.0f;
            FOR_UNROLL (short j = 0; j < NS; ++j) {
                if (j <= k) {
                    o += w[j*NQ + iq]*(j < k ? s[j] : y);
                }
            }
            ((device float4 *) (dt + iq*args.nb1))[tpitg] = o;
        }
    }
}

// dst = a - tanh(wf*x)*a + tanh(wg*x)*b, NR rows per simdgroup and args.nt tokens per threadgroup
// each lane holds NK values of every row, so the weights are read once per threadgroup
template <typename T0, short NR>
kernel void kernel_tanh_gate(
        constant ggml_metal_kargs_tanh_gate & args,
        device const char * wf,
        device const char * wg,
        device const char * x,
        device const char * a,
        device const char * b,
        device       char * dst,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    constexpr short NK = GGML_METAL_TANH_GATE_NK;

    const int r0 = (tgpig.x*nsg + sgitg)*NR;

    float vf[NR][NK];
    float vg[NR][NK];

    for (short r = 0; r < NR; ++r) {
        const int row = r0 + r;
        device const T0 * pf = (device const T0 *) (wf + (uint64_t) min(row, args.ne01 - 1)*args.nb01);
        device const T0 * pg = (device const T0 *) (wg + (uint64_t) min(row, args.ne01 - 1)*args.nb01);
        for (short k = 0; k < NK; ++k) {
            const int i = tiisg + 32*k;
            vf[r][k] = i < args.ne00 ? float(pf[i]) : 0.0f;
            vg[r][k] = i < args.ne00 ? float(pg[i]) : 0.0f;
        }
    }

    const int t0 = tgpig.y*args.nt;
    const int t1 = min(t0 + args.nt, args.ne11);

    for (int t = t0; t < t1; ++t) {
        device const float * xt = (device const float *) (x + (uint64_t) t*args.nb11);

        float xv[NK];
        for (short k = 0; k < NK; ++k) {
            const int i = tiisg + 32*k;
            xv[k] = i < args.ne00 ? xt[i] : 0.0f;
        }

        device const float * at = (device const float *) (a   + (uint64_t) t*args.nba1);
        device const float * bt = (device const float *) (b   + (uint64_t) t*args.nbb1);
        device       float * dt = (device       float *) (dst + (uint64_t) t*args.nb1);

        for (short r = 0; r < NR; ++r) {
            float sf = 0.0f;
            float sg = 0.0f;
            for (short k = 0; k < NK; ++k) {
                sf += vf[r][k]*xv[k];
                sg += vg[r][k]*xv[k];
            }
            sf = simd_sum(sf);
            sg = simd_sum(sg);

            const int row = r0 + r;
            if (tiisg == r && row < args.ne01) {
                const float av = at[row];
                const float fa = precise::tanh(sf)*av;
                const float gb = precise::tanh(sg)*bt[row];
                dt[row] = (av - fa) + gb;
            }
        }
    }
}

// same op as kernel_tanh_gate for many tokens: one thread per row, NT tokens per threadgroup with x in threadgroup memory
// each dot product is summed in order in one thread, so no simd reduction per output
template <typename T0, short NT>
kernel void kernel_tanh_gate_batch(
        constant ggml_metal_kargs_tanh_gate & args,
        device const char * wf,
        device const char * wg,
        device const char * x,
        device const char * a,
        device const char * b,
        device       char * dst,
        threadgroup   float * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int t0 = tgpig.y*NT;
    const int nt = min((int) NT, args.ne11 - t0);

    for (int i = tpitg.x; i < NT*args.ne00; i += ntg.x) {
        const int t = i/args.ne00;
        const int k = i - t*args.ne00;
        shmem[i] = t < nt ? ((device const float *) (x + (uint64_t) (t0 + t)*args.nb11))[k] : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int row = tgpig.x*ntg.x + tpitg.x;
    if (row >= args.ne01) {
        return;
    }

    device const T0 * pf = (device const T0 *) (wf + (uint64_t) row*args.nb01);
    device const T0 * pg = (device const T0 *) (wg + (uint64_t) row*args.nb01);

    float sf[NT];
    float sg[NT];
    for (short t = 0; t < NT; ++t) {
        sf[t] = 0.0f;
        sg[t] = 0.0f;
    }

    for (int k = 0; k < args.ne00; ++k) {
        const float vf = float(pf[k]);
        const float vg = float(pg[k]);
        for (short t = 0; t < NT; ++t) {
            const float xv = shmem[t*args.ne00 + k];
            sf[t] += vf*xv;
            sg[t] += vg*xv;
        }
    }

    for (short t = 0; t < NT; ++t) {
        if (t >= nt) {
            break;
        }
        const float av = ((device const float *) (a + (uint64_t) (t0 + t)*args.nba1))[row];
        const float bv = ((device const float *) (b + (uint64_t) (t0 + t)*args.nbb1))[row];
        const float fa = precise::tanh(sf[t])*av;
        const float gb = precise::tanh(sg[t])*bv;
        ((device float *) (dst + (uint64_t) (t0 + t)*args.nb1))[row] = (av - fa) + gb;
    }
}

typedef decltype(kernel_tanh_gate_batch<float, 16>) kernel_tanh_gate_batch_t;

template [[host_name("kernel_tanh_gate_batch_f32")]]  kernel kernel_tanh_gate_batch_t kernel_tanh_gate_batch<float, 16>;
template [[host_name("kernel_tanh_gate_batch_f16")]]  kernel kernel_tanh_gate_batch_t kernel_tanh_gate_batch<half, 16>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_tanh_gate_batch_bf16")]] kernel kernel_tanh_gate_batch_t kernel_tanh_gate_batch<bfloat, 16>;
#endif

typedef decltype(kernel_tanh_gate<float, 4>) kernel_tanh_gate_t;

template [[host_name("kernel_tanh_gate_f32")]]  kernel kernel_tanh_gate_t kernel_tanh_gate<float, 4>;
template [[host_name("kernel_tanh_gate_f16")]]  kernel kernel_tanh_gate_t kernel_tanh_gate<half, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_tanh_gate_bf16")]] kernel kernel_tanh_gate_t kernel_tanh_gate<bfloat, 4>;
#endif

// dst[kv, ti, h, s] = half((table[rel[kv, t], h, t] - sinks[h, t]) + mask[kv, ti, 0, s]) with t = s*n_tps + ti
// one thread per (kv, t), the heads are a loop so rel and mask are read once
kernel void kernel_rel_bias_mask(
        constant ggml_metal_kargs_rel_bias_mask & args,
        device const char * table,
        device const char * sinks,
        device const char * rel,
        device const char * mask,
        device       char * dst,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int kv = tgpig.x*ntg.x + tpitg.x;
    if (kv >= args.n_kv) {
        return;
    }

    const int t  = tgpig.y;
    const int s  = t / args.n_tps;
    const int ti = t - s*args.n_tps;

    const int32_t r = ((device const int32_t *) (rel + (uint64_t) t*args.nbr1))[kv];
    const float   m = (float) ((device const half *) (mask + (uint64_t) ti*args.nbm1 + (uint64_t) s*args.nbm3))[kv];

    device const char * tt = table + (uint64_t) t*args.nbt2 + (uint64_t) r*sizeof(float);
    device const char * st = sinks + (uint64_t) t*args.nbs2;
    device       char * dt = dst   + (uint64_t) ti*args.nb1 + (uint64_t) s*args.nb3 + (uint64_t) kv*sizeof(half);

    for (int h = 0; h < args.n_head; ++h) {
        const float v = *((device const float *) (tt + (uint64_t) h*args.nbt1)) - *((device const float *) (st + (uint64_t) h*args.nbs1));
        *((device half *) (dt + (uint64_t) h*args.nb2)) = (half) (v + m);
    }
}

// dst[c, tt] = scale*(res[c, tt] + sum_k x[t + k, c, s]*w[k, c]) + bias with tt = s*n_t + t, one thread per (c, tt)
// dst = a*sigmoid(g) with the float4 math of the SIGMOID and MUL kernels; OPQ keeps the sigmoid result as is (see kernel_situ_f32)
kernel void kernel_sigmoid_mul_f32(
        constant ggml_metal_kargs_sigmoid_mul & args,
        constant uint32_t & zero,
        device const char * a,
        device const char * g,
        device       char * dst,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int i0 = tgpig.x*ntg.x + tpitg.x;
    if (4*i0 >= args.ne0) {
        return;
    }

    const int i1 = tgpig.y;

    const float4 x = ((device const float4 *) (g + (uint64_t) i1*args.nbg1))[i0];
    const float4 v = ((device const float4 *) (a + (uint64_t) i1*args.nba1))[i0];

    const float4 sg = as_type<float4>(as_type<uint4>(1 / (1 + exp(-x))) ^ uint4(zero));

    ((device float4 *) (dst + (uint64_t) i1*args.nb1))[i0] = v*sg;
}

// dst[i, e, t] = res[i, e, t]*s[ids[e, t]]: the same product as the MUL of the gathered expert scales
kernel void kernel_expert_scale(
        constant ggml_metal_kargs_expert_scale & args,
        device const char * res,
        device const char * s,
        device const char * ids,
        device       char * dst,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int i = tgpig.x*ntg.x + tpitg.x;
    if (i >= args.ne0) {
        return;
    }

    const int e = tgpig.y;
    const int t = tgpig.z;

    const int32_t id = *((device const int32_t *) (ids + (uint64_t) e*args.nbi0 + (uint64_t) t*args.nbi1));
    const float   sc = *((device const float *) (s + (uint64_t) id*args.nbs1));

    const float r = ((device const float *) (res + (uint64_t) e*args.nbr1 + (uint64_t) t*args.nbr2))[i];
    ((device float *) (dst + (uint64_t) e*args.nb1 + (uint64_t) t*args.nb2))[i] = r*sc;
}

// kernel_ssm_conv_res over [past state | new inputs] without the concat, and the last n_win tokens store the state after them
// the sum is the same loop as kernel_ssm_conv_res, so the result matches the unfused nodes
// has_res2: the ADD of a second residual after the scale (plain taps only)
// with input-dependent taps (n_dyn > 0) the conv is the float4 sum of kernel_ssm_conv_f32_f32_4, then each lag adds x*tap,
// then the residual and the scale, each step rounded as the unfused nodes round it
kernel void kernel_conv_store(
        constant ggml_metal_kargs_conv_store & args,
        device const char * past,
        device const char * x,
        device const char * w,
        device const char * res,
        device const char * idx,
        device       char * state,
        device       char * dst,
        device const char * dyn,
        device const char * res2,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int c = tgpig.x*ntg.x + tpitg.x;
    if (c >= args.n_ch) {
        return;
    }

    const int tt = tgpig.y;
    const int s  = tt / args.n_t;
    const int t  = tt - s*args.n_t;

    float xs[GGML_METAL_CONV_STORE_NC_MAX];
    for (int k = 0; k < args.nc; ++k) {
        const int col = t + k;
        xs[k] = col < args.nc - 1
            ? *((device const float *) (past + (uint64_t) col*args.nbp0 + (uint64_t) c*args.nbp1 + (uint64_t) s*args.nbp2))
            : *((device const float *) (x + (uint64_t) (col - args.nc + 1)*args.nbx0 + (uint64_t) c*args.nbx1 + (uint64_t) s*args.nbx2));
    }

    device const float * ws = (device const float *) (w + (uint64_t) c*args.nbw1);

    const float r = ((device const float *) (res + (uint64_t) tt*args.nbr1))[c];

    if (args.n_dyn > 0) {
#define OPQ(v) as_type<float>(as_type<uint>(v) ^ args.zero)
        float sum = 0.0f;
        for (int k = 0; k < args.nc/4; ++k) {
            sum += dot(float4(xs[4*k + 0], xs[4*k + 1], xs[4*k + 2], xs[4*k + 3]), ((device const float4 *) ws)[k]);
        }
        sum = OPQ(sum);

        device const float * tap = (device const float *) (dyn + (uint64_t) (c/args.n_ch_h)*args.nbd1 + (uint64_t) tt*args.nbd2);
        for (int lag = 0; lag < args.n_dyn; ++lag) {
            sum = OPQ(sum + OPQ(xs[args.nc - 1 - lag]*tap[lag]));
        }

        ((device float *) (dst + (uint64_t) tt*args.nb1))[c] = args.scale*OPQ(r + sum) + args.bias;
#undef OPQ
    } else {
        float sum = 0.0f;
        for (int k = 0; k < args.nc; ++k) {
            sum += xs[k]*ws[k];
        }

        float out = args.scale*(r + sum) + args.bias;
        if (args.has_res2) {
            // the second residual is added to the scale result as the unfused SCALE rounds it
            out = as_type<float>(as_type<uint>(out) ^ args.zero) + ((device const float *) (res2 + (uint64_t) tt*args.nbq1))[c];
        }

        ((device float *) (dst + (uint64_t) tt*args.nb1))[c] = out;
    }

    const int win = t - (args.n_t - args.n_win);
    if (win >= 0) {
        const int slot = args.has_idx ? ((device const int32_t *) idx)[win] : 0;
        device float * out = (device float *) (state + (uint64_t) slot*args.nbs1 + (uint64_t) c*args.nbs2 + (uint64_t) s*args.nbs3);
        for (int j = 0; j < args.nc - 1; ++j) {
            out[j] = xs[j + 1];
        }
    }
}

kernel void kernel_ssm_conv_res(
        constant ggml_metal_kargs_ssm_conv_res & args,
        device const char * x,
        device const char * w,
        device const char * res,
        device       char * dst,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int c = tgpig.x*ntg.x + tpitg.x;
    if (c >= args.n_ch) {
        return;
    }

    const int tt = tgpig.y;
    const int s  = tt / args.n_t;
    const int t  = tt - s*args.n_t;

    device const float * xs = (device const float *) (x + (uint64_t) c*args.nbx1 + (uint64_t) s*args.nbx2) + t;
    device const float * ws = (device const float *) (w + (uint64_t) c*args.nbw1);

    float sum = 0.0f;
    for (int k = 0; k < args.nc; ++k) {
        sum += xs[k]*ws[k];
    }

    const float r = ((device const float *) (res + (uint64_t) tt*args.nbr1))[c];
    ((device float *) (dst + (uint64_t) tt*args.nb1))[c] = args.scale*(r + sum) + args.bias;
}
