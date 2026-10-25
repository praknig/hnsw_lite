/**
 * @file half.h
 * @brief Conversions between float32 and the two 16-bit formats: IEEE 754 binary16
 * (float16) and bfloat16. Used by the half-precision codecs (codec.h).
 *
 * float16: 5 exponent bits, 10 mantissa bits. Finite values up to 65504, about 3.3
 * significant digits, subnormals down to 2^-24.
 * bfloat16: the top 16 bits of a float32 (8 exponent bits, 7 mantissa bits): float32's
 * range with about 2.4 significant digits.
 *
 * Both conversions from float32 round to the nearest value, ties to even, as the
 * hardware instructions do. A finite input never becomes infinite: a value beyond the
 * format's largest saturates to it (callers refuse such values first, see
 * HalfCodec::check; saturation is the defence behind that check).
 */
#pragma once

#include <bit>
#include <cstdint>

namespace vecdb {

/// float32 to float16 bits (round to nearest even; beyond 65504 saturates to it).
inline std::uint16_t float_to_f16(float f) noexcept {
    std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = x & 0x80000000u;
    x ^= sign;
    std::uint16_t out;
    if (x >= 0x7F800000u) {  // infinity or NaN (callers never pass them: kept as such)
        out = x > 0x7F800000u ? 0x7E00 : 0x7C00;
    } else if (x >= 0x477FF000u) {  // 65520 and above would round to infinity: saturate
        out = 0x7BFF;
    } else if (x < 0x38800000u) {  // below 2^-14: a subnormal float16, or zero
        // Adding 0.5 aligns the value so that the float addition itself rounds to the
        // nearest subnormal, ties to even (the classic "magic number" conversion).
        const float magic = std::bit_cast<float>(0x3F000000u);  // 0.5
        const float aligned = std::bit_cast<float>(x) + magic;
        out = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(aligned) - 0x3F000000u);
    } else {  // a normal float16: rebias the exponent, round the 13 dropped bits to even
        const std::uint32_t odd = (x >> 13) & 1u;
        x += 0xC8000FFFu + odd;  // (15 - 127) << 23, plus the rounding bias
        out = static_cast<std::uint16_t>(x >> 13);
        if (out >= 0x7C00) out = 0x7BFF;  // rounding carried into infinity: saturate
    }
    return static_cast<std::uint16_t>(out | (sign >> 16));
}

/// float16 bits to float32 (exact: every float16 value is a float32 value).
inline float f16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;  // zero
        } else {  // subnormal: shift until the implicit bit appears
            std::uint32_t e = 113;  // 127 - 15 + 1
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((mantissa & 0x3FFu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);  // infinity or NaN
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

/// float32 to bfloat16 bits (round to nearest even; a finite value that would round
/// to infinity saturates to the largest bfloat16 instead).
inline std::uint16_t float_to_bf16(float f) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    if ((x & 0x7F800000u) == 0x7F800000u) return static_cast<std::uint16_t>((x >> 16) | ((x & 0xFFFFu) ? 0x40u : 0u));  // inf, NaN
    const std::uint32_t rounded = x + 0x7FFFu + ((x >> 16) & 1u);
    std::uint16_t out = static_cast<std::uint16_t>(rounded >> 16);
    if ((out & 0x7F80u) == 0x7F80u) out = static_cast<std::uint16_t>((out & 0x8000u) | 0x7F7Fu);  // saturate
    return out;
}

/// bfloat16 bits to float32 (exact).
inline float bf16_to_float(std::uint16_t h) noexcept { return std::bit_cast<float>(std::uint32_t(h) << 16); }

/// The largest finite float16 value.
inline constexpr float kF16Max = 65504.0f;

}  // namespace vecdb