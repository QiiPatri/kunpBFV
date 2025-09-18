#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda/std/type_traits>

constexpr int MOD_BIT_COUNT_MAX = 61;
constexpr int MOD_BIT_COUNT_MIN = 2;

// MUST dividible by poly_degree
constexpr dim3 blockDimGlb(128);

template<typename T, typename...>
struct IsUInt64
        : std::conditional<
                std::is_integral<T>::value && std::is_unsigned<T>::value && (sizeof(T) == sizeof(std::uint64_t)),
                std::true_type, std::false_type>::type {
};

template<typename T, typename... Rest>
constexpr bool is_uint64_v = IsUInt64<T, Rest...>::value;

template<typename T>
[[nodiscard]] inline T sub_safe(T in1, T in2)
{
    if (in1 < 0 && (in2 > (std::numeric_limits<T>::max)() + in1))
    {
        throw std::logic_error("signed underflow");
    }else if (in1 > 0 && (in2 < (std::numeric_limits<T>::min)() + in1))
    {
        throw std::logic_error("signed overflow");
    }
    return static_cast<T>(in1 - in2);
}
template<typename T>
[[nodiscard]] inline constexpr T mul_safe(T in1, T in2)
{
    if constexpr (cuda::std::is_unsigned<T>::value)
    {
        if (in1 && (in2 > (std::numeric_limits<T>::max)() / in1))
        {
            throw std::logic_error("unsigned overflow");
        }
    }else
    {
        // Positive inputs
        if ((in1 > 0) && (in2 > 0) && (in2 > (std::numeric_limits<T>::max)() / in1))
        {
            throw std::logic_error("signed overflow");
        }
        // Negative inputs
        else if ((in1 < 0) && (in2 < 0) && ((-in2) > (std::numeric_limits<T>::max)() / (-in1)))
        {
            throw std::logic_error("signed overflow");
        }
        // Negative in1; positive in2
        else if ((in1 < 0) && (in2 > 0) && (in2 > (std::numeric_limits<T>::max)() / (-in1)))
        {
            puts("negative in1");
            throw std::logic_error("signed underflow");
        }
        // Positive in1; negative in2
        else if ((in1 > 0) && (in2 < 0) && (in2 < (std::numeric_limits<T>::min)() / in1))
        {
            puts("negative in2");
            throw std::logic_error("signed underflow");
        }
    }
    return static_cast<T>(in1 * in2);
}

inline void get_msb_index_generic(unsigned long *result, uint64_tt value)
{
    static const unsigned long deBruijnTable64[64] =
    {
        63, 0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54,
        33, 42, 3, 61, 51, 37, 40, 49, 18, 28, 20, 55, 30,
        34, 11, 43, 14, 22, 4, 62, 57, 46, 52, 38, 26, 32,
        41, 50, 36, 17, 19, 29, 10, 13, 21, 56, 45, 25, 31,
        35, 16, 9, 12, 44, 24, 15, 8, 23, 7, 6, 5
    };

    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;

    *result = deBruijnTable64[((value - (value >> 1)) * uint64_tt(0x07EDD5E59A4E28C2)) >> 58];
}
//最低位为第1位，如34359410689对应35
[[nodiscard]] inline int get_significant_bit_count(uint64_tt value)
{
    if (value == 0)
    {
        return 0;
    }

    unsigned long result = 0;
    get_msb_index_generic(&result, value);
    return static_cast<int>(result + 1);
}

constexpr int bytes_per_uint64 = sizeof(uint64_tt);

constexpr int bits_per_nibble = 4;

constexpr int bits_per_byte = 8;

constexpr int bits_per_uint64 = bytes_per_uint64 * bits_per_byte;

constexpr int nibbles_per_byte = 2;

constexpr int nibbles_per_uint64 = bytes_per_uint64 * nibbles_per_byte;

[[nodiscard]] inline int get_significant_bit_count_uint(const uint64_tt *value, std::size_t uint64_count)
{
    value += uint64_count - 1;
    for (; *value == 0 && uint64_count > 1; uint64_count--)
    {
        value--;
    }

    return static_cast<int>(uint64_count - 1) * bits_per_uint64 + get_significant_bit_count(*value);
}