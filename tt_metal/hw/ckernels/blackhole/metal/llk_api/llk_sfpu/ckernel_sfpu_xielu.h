// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _sfpu_neg_exp_f32_(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;

    // exp(x) is below the smallest normal fp32 value for x <= -88.0f.
    // SFPU flushes subnormals to zero, so avoid range reduction entirely in
    // that region. This also keeps the float-to-int rounding helper and the
    // polynomial residual in their valid ranges for arbitrarily negative x.
    v_if(val <= -88.0f) { return result; }
    v_endif;

    sfpi::vFloat z = val * sfpi::vConstFloatPrgm0;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    constexpr float LN2_HI = -0.6931152343750000f;
    constexpr float LN2_LO = -3.19461832987e-05f;
    sfpi::vFloat r_hi = k * LN2_HI + val;
    sfpi::vFloat r = k * LN2_LO + r_hi;

    sfpi::vFloat p = PolynomialEvaluator::eval(
        r,
        1.0f,
        1.0f,
        0.5f,
        1.0f / 6.0f,
        1.0f / 24.0f,
        1.0f / 120.0f,
        1.0f / 720.0f,
        1.0f / 5040.0f);

    sfpi::vInt p_exp = sfpi::exexp(p, sfpi::ExponentMode::Biased);
    sfpi::vInt new_exp = p_exp + k_int;
    result = sfpi::setexp(p, new_exp);
    return result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline void _xielu_mad_(sfpi::vFloat mul_a, sfpi::vFloat mul_b, sfpi::vFloat addend) {
    sfpi::vFloat result = mul_a * mul_b + addend;
    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_xielu(const uint32_t param0, const uint32_t param1) {
    sfpi::vFloat alpha_p = Converter::as_float(param0);
    sfpi::vFloat alpha_n = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat beta_mul_x = 0.5f * x;
        v_if(x > 0.0f) {
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_p * x, x, beta_mul_x);
        }
        v_elseif(x >= sfpi::vConstFloatPrgm1) {
            sfpi::vFloat exp_term = sfpi::vConstFloatPrgm2 - x;
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
        }
        v_elseif(x > -0.5f) {
            sfpi::vFloat exp_term = x * x *
                                    PolynomialEvaluator::eval(
                                        x,
                                        0.500000059604644775390625f,
                                        0.16666667163372039794921875f,
                                        4.16650883853435516357421875e-2f,
                                        8.333188481628894805908203125e-3f,
                                        1.400390756316483020782470703125e-3f,
                                        1.99588379473425447940826416015625e-4f);
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
        }
        v_else {
            sfpi::vFloat exp_term = _sfpu_neg_exp_f32_(x) - 1.0f - x;
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void xielu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;
    sfpi::vConstFloatPrgm1 = -1e-6f;
    sfpi::vConstFloatPrgm2 = -0.0000009999995427f;
}

}  // namespace ckernel::sfpu
