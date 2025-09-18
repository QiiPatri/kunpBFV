#pragma once

#include <functional>
#include <type_traits>
#include "common.h"
#include "uint128.cuh"

template<typename T>
[[nodiscard]] inline constexpr T add_safe(T in1, T in2)
{
    if (in1 > 0 && (in2 > (std::numeric_limits<T>::max)() - in1))
    {
        throw std::logic_error("signed overflow");
    } else if (in1 < 0 && (in2 < (std::numeric_limits<T>::min)() - in1))
    {
        throw std::logic_error("signed underflow");
    }
    return static_cast<T>(in1 + in2);
}
template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
[[nodiscard]] inline T divide_round_up(T value, T divisor)
{
    return (add_safe(value, divisor - 1)) / divisor;
}
//Return the sum of two operands (with carry) | The sum is stored in result | Return value is the carry
[[nodiscard]] inline unsigned char add_uint64(uint64_tt operand1, uint64_tt operand2, unsigned char carry, uint64_tt *result)
{
    operand1 += operand2;
    *result = operand1 + carry;
    return (operand1 < operand2) || (~operand1 < carry);
}
[[nodiscard]] inline unsigned char add_uint64(uint64_tt operand1, uint64_tt operand2, uint64_tt *result)
{
    *result = operand1 + operand2;
    return static_cast<unsigned char>(*result < operand1);
}
inline void set_zero_uint64(std::size_t uint64_count, uint64_tt *result)
{
    std::fill_n(result, uint64_count, uint64_tt(0));
}

inline void set_uint(uint64_tt value, std::size_t uint64_count, uint64_tt *result)
{
    *result++ = value;
    for (; --uint64_count; result++)
    {
        *result = 0;
    }
}
inline void set_uint(const uint64_tt *value, std::size_t uint64_count, uint64_tt *result)
{
    if ((value == result) || !uint64_count)
    {
        return;
    }
    std::copy_n(value, uint64_count, result);
}

//fill the result with the value
inline void set_uint64(uint64_tt value, std::size_t uint64_count, uint64_tt *result)
{
    *result++ = value;
    for( ; --uint64_count ; ++result)
    {
        *result = 0;
    }
}
inline void set_uint64(const uint64_tt *value, std::size_t uint64_count, uint64_tt *result)
{
    if( (value == result) || !uint64_count )
    {
        return;
    }
    std::copy_n(value, uint64_count, result);
}

//------------------------------------------------------------------------------------------
inline unsigned char add_uint(
        const uint64_tt *operand1, std::size_t operand1_uint64_count, const uint64_tt *operand2,
        std::size_t operand2_uint64_count, unsigned char carry, std::size_t result_uint64_count, uint64_tt *result)
{
    for (std::size_t i = 0; i < result_uint64_count; i++)
    {
        uint64_tt temp_result;
        carry = add_uint64(
                (i < operand1_uint64_count) ? *operand1++ : 0, (i < operand2_uint64_count) ? *operand2++ : 0, carry,
                &temp_result);
        *result++ = temp_result;
    }
    return carry;
}

inline unsigned char add_uint(const uint64_tt *operand1, const uint64_tt *operand2, std::size_t uint64_count, uint64_tt *result)
{
    // Unroll first iteration of loop. We assume uint64_count > 0.
    unsigned char carry = add_uint64(*operand1++, *operand2++, result++);

    // Do the rest
    for (; --uint64_count; operand1++, operand2++, result++)
    {
        uint64_tt temp_result;
        carry = add_uint64(*operand1, *operand2, carry, &temp_result);
        *result = temp_result;
    }
    return carry;
}

inline unsigned char add_uint(const uint64_tt *operand1, std::size_t uint64_count, uint64_tt operand2, uint64_tt *result)
{
    // Unroll first iteration of loop. We assume uint64_count > 0.
    unsigned char carry = add_uint64(*operand1++, operand2, result++);

    // Do the rest
    for (; --uint64_count; operand1++, result++)
    {
        uint64_tt temp_result;
        carry = add_uint64(*operand1, uint64_tt(0), carry, &temp_result);
        *result = temp_result;
    }
    return carry;
}

template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>
[[nodiscard]] inline unsigned char sub_uint64_generic(T operand1, S operand2, unsigned char borrow, uint64_tt *result)
{
    auto diff = operand1 - operand2;
    *result = diff - (borrow != 0);
    return (diff > operand1) || (diff < borrow);
}

template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>
[[nodiscard]] inline unsigned char sub_uint64(T operand1, S operand2, unsigned char borrow, uint64_tt *result)
{
    return sub_uint64_generic(operand1, operand2, borrow, result);
}

template<typename T, typename S, typename R, typename = std::enable_if_t<is_uint64_v < T, S, R>>>
[[nodiscard]] inline unsigned char sub_uint64(T operand1, S operand2, R *result)
{
    *result = operand1 - operand2;
    return static_cast<unsigned char>(operand2 > operand1);
}

inline unsigned char sub_uint(
        const uint64_tt *operand1, std::size_t operand1_uint64_count, const uint64_tt *operand2,
        std::size_t operand2_uint64_count, unsigned char borrow, std::size_t result_uint64_count, uint64_tt *result)
{
    for (std::size_t i = 0; i < result_uint64_count; i++, operand1++, operand2++, result++)
    {
        uint64_tt temp_result;
        borrow = sub_uint64(
                (i < operand1_uint64_count) ? *operand1 : 0, (i < operand2_uint64_count) ? *operand2 : 0, borrow,
                &temp_result);
        *result = temp_result;
    }
    return borrow;
}

inline unsigned char sub_uint(
        const uint64_tt *operand1, const uint64_tt *operand2, std::size_t uint64_count, uint64_tt *result)
    {
    // Unroll first iteration of loop. We assume uint64_count > 0.
    unsigned char borrow = sub_uint64(*operand1++, *operand2++, result++);

    // Do the rest
    for (; --uint64_count; operand1++, operand2++, result++)
    {
        uint64_tt temp_result;
        borrow = sub_uint64(*operand1, *operand2, borrow, &temp_result);
        *result = temp_result;
    }
    return borrow;
}

inline unsigned char sub_uint(const uint64_tt *operand1, std::size_t uint64_count, uint64_tt operand2, uint64_tt *result)
{
    // Unroll first iteration of loop. We assume uint64_count > 0.
    unsigned char borrow = sub_uint64(*operand1++, operand2, result++);

    // Do the rest
    for (; --uint64_count; operand1++, operand2++, result++)
    {
        uint64_tt temp_result;
        borrow = sub_uint64(*operand1, uint64_tt(0), borrow, &temp_result);
        *result = temp_result;
    }
    return borrow;
}

// Return the first (right first) non-zero uint64 value.
[[nodiscard]] inline std::size_t get_significant_uint64_count_uint(const uint64_tt *value, std::size_t uint64_count)
{
    value += uint64_count - 1;
    for(; uint64_count && !*value; uint64_count--)
    {
        --value;
    }
    return uint64_count;
}
inline void multiply_uint64(uint64_tt operand1, uint64_tt operand2, uint64_tt *result128)
{
    auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
    auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
    operand1 >>= 32;
    operand2 >>= 32;

    auto middle1 = operand1 * operand2_coeff_right;
    uint64_tt middle;
    auto left = operand1 * operand2 + (static_cast<uint64_tt>(add_uint64(middle1, operand2 * operand1_coeff_right, &middle)) << 32);
    auto right = operand1_coeff_right * operand2_coeff_right;
    auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

    result128[1] = static_cast<uint64_tt>(left + (middle >> 32) + (temp_sum >> 32));
    result128[0] = static_cast<uint64_tt>((temp_sum << 32) | (right & 0x00000000FFFFFFFF));
}
//Return the product of operand1 (multiple uint64_tt) with operand2 (one uint64_tt)
void multiply_uint64(const uint64_tt *operand1, size_t operand1_uint64_count, const uint64_tt operand2, size_t result_uint64_count, uint64_tt *result)
{
    if (!operand1_uint64_count || !operand2)
    {
        // If either operand is 0, then result is 0.
        set_zero_uint64(result_uint64_count, result);
        return;
    }
    if (result_uint64_count == 1)
    {
        *result = *operand1 * operand2;
        return;
    }

    // Clear out result.
    set_zero_uint64(result_uint64_count, result);

    // Multiply operand1 and operand2.
    uint64_tt carry = 0;
    size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
    for (size_t operand1_index = 0; operand1_index < operand1_index_max; ++operand1_index)
    {
        uint64_tt temp_result[2];
        multiply_uint64(*operand1++, operand2, temp_result);
        uint64_tt temp;
        carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, &temp);
        *result++ = temp;
    }

    // Write carry if there is room in result
    if (operand1_index_max < result_uint64_count)
    {
        *result = carry;
    }
}
void multiply_uint64(const uint64_tt *operand1, size_t operand1_uint64_count, const uint64_tt *operand2, size_t operand2_uint64_count, size_t result_uint64_count, uint64_tt *result)
{
    if (!operand1_uint64_count || !operand2_uint64_count)
    {
        // If either operand is 0, then result is 0.
        set_zero_uint64(result_uint64_count, result);
        return;
    }
    if (result_uint64_count == 1)
    {
        *result = *operand1 * *operand2;
        return;
    }

    // obtain first non-zero uint64.
    operand1_uint64_count = get_significant_uint64_count_uint(operand1, operand1_uint64_count);
    operand2_uint64_count = get_significant_uint64_count_uint(operand2, operand2_uint64_count);

    if (operand1_uint64_count == 1)
    {
        multiply_uint64(operand2, operand2_uint64_count, *operand1, result_uint64_count, result);
        return;
    }
    if (operand2_uint64_count == 1)
    {
        multiply_uint64(operand1, operand1_uint64_count, *operand2, result_uint64_count, result);
        return;
    }

    // Clear out result.
    set_zero_uint64(result_uint64_count, result);

    // Multiply operand1 and operand2.
    size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
    for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++)
    {
        const uint64_tt *inner_operand2 = operand2;
        uint64_tt *inner_result = result++;
        uint64_tt carry = 0;
        size_t operand2_index = 0;
        size_t operand2_index_max = min(operand2_uint64_count, result_uint64_count - operand1_index);
        for (; operand2_index < operand2_index_max; ++operand2_index)
        {
            // Perform 64-bit multiplication of operand1 and operand2
            uint64_tt temp_result[2];
            multiply_uint64(*operand1, *inner_operand2++, temp_result);
            carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, temp_result);
            uint64_tt temp;
            carry += add_uint64(*inner_result, temp_result[0], 0, &temp);
            *inner_result++ = temp;
        }

        // Write carry if there is room in result
        if (operand1_index + operand2_index_max < result_uint64_count)
        {
            *inner_result = carry;
        }

        ++operand1;
    }
}

void multiply_uint(
        const uint64_tt *operand1, size_t operand1_uint64_count, uint64_tt operand2, size_t result_uint64_count,
        uint64_tt *result)
{
    if (!operand1_uint64_count || !operand2)
    {
        // If either operand is 0, then result is 0.
        std::fill_n(result, result_uint64_count, uint64_tt(0));
        return;
    }
    if (result_uint64_count == 1)
    {
        *result = *operand1 * operand2;
        return;
    }

    // Clear out result.
    std::fill_n(result, result_uint64_count, uint64_tt(0));

    // Multiply operand1 and operand2.
    uint64_tt carry = 0;
    size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
    for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++)
    {
        uint64_tt temp_result[2];
        multiply_uint64(*operand1++, operand2, temp_result);
        uint64_tt temp;
        carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, &temp);
        *result++ = temp;
    }

    // Write carry if there is room in result
    if (operand1_index_max < result_uint64_count)
    {
        *result = carry;
    }
}

inline void multiply_many_uint64(uint64_tt *operands, std::size_t count, uint64_tt *result)
{
    if (!count)
        return;
    // Set result to operands[0]
    set_uint64(operands[0], count, result);
    // Compute product
    std::vector<uint64_tt> temp_mpi(count);
    for (std::size_t i = 1; i < count; i++) {
        multiply_uint64(result, i, operands[i], i + 1, temp_mpi.data());
        set_uint64(temp_mpi.data(), i + 1, result);
    }
}

template<typename T, typename S, typename = std::enable_if_t<is_uint64_v < T, S>>>
inline void multiply_uint64_hw64_generic(T operand1, S operand2, uint64_tt *hw64) {
    auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
    auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
    operand1 >>= 32;
    operand2 >>= 32;

    auto middle1 = operand1 * operand2_coeff_right;
    uint64_tt middle;
    auto left = operand1 * operand2 +
                (static_cast<T>(add_uint64(middle1, operand2 * operand1_coeff_right, &middle)) << 32);
    auto right = operand1_coeff_right * operand2_coeff_right;
    auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

    *hw64 = static_cast<uint64_tt>(left + (middle >> 32) + (temp_sum >> 32));
}

template<typename T, typename = std::enable_if_t<is_uint64_v < T>>>
inline void multiply_many_uint64_except(T *operands, std::size_t count, std::size_t except, T *result)
{
    // An empty product; return 1
    if (count == 1 && except == 0) 
    {
        set_uint(1, count, result);
        return;
    }

    // Set result to operands[0] unless except is 0
    set_uint(except == 0 ? uint64_tt(1) : static_cast<uint64_tt>(operands[0]), count, result);

    // Compute punctured product
    std::vector<uint64_tt> temp_mpi(count);
    for (std::size_t i = 1; i < count; i++)
    {
        if (i != except)
        {
            multiply_uint(result, i, operands[i], i + 1, temp_mpi.data());
            set_uint(temp_mpi.data(), i + 1, result);
        }
    }
}

inline void left_shift_uint192(const uint64_tt *operand, int shift_amount, uint64_tt *result)
{
    const auto bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);

    const auto shift_amount_sz = static_cast<std::size_t>(shift_amount);

    if (shift_amount_sz & (bits_per_uint64_sz << 1)) {
        result[2] = operand[0];
        result[1] = 0;
        result[0] = 0;
    } else if (shift_amount_sz & bits_per_uint64_sz) {
        result[2] = operand[1];
        result[1] = operand[0];
        result[0] = 0;
    } else {
        result[2] = operand[2];
        result[1] = operand[1];
        result[0] = operand[0];
    }

    // How many bits to shift in addition to word shift
    std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

    if (bit_shift_amount) {
        std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

        // Warning: if bit_shift_amount == 0 this is incorrect
        result[2] = (result[2] << bit_shift_amount) | (result[1] >> neg_bit_shift_amount);
        result[1] = (result[1] << bit_shift_amount) | (result[0] >> neg_bit_shift_amount);
        result[0] = result[0] << bit_shift_amount;
    }
}

inline void right_shift_uint192(const uint64_tt *operand, int shift_amount, uint64_tt *result)
{
    const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);

    const std::size_t shift_amount_sz = static_cast<std::size_t>(shift_amount);

    if (shift_amount_sz & (bits_per_uint64_sz << 1)) {
        result[0] = operand[2];
        result[1] = 0;
        result[2] = 0;
    } else if (shift_amount_sz & bits_per_uint64_sz) {
        result[0] = operand[1];
        result[1] = operand[2];
        result[2] = 0;
    } else {
        result[2] = operand[2];
        result[1] = operand[1];
        result[0] = operand[0];
    }

    // How many bits to shift in addition to word shift
    std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

    if (bit_shift_amount) {
        std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

        // Warning: if bit_shift_amount == 0 this is incorrect
        result[0] = (result[0] >> bit_shift_amount) | (result[1] << neg_bit_shift_amount);
        result[1] = (result[1] >> bit_shift_amount) | (result[2] << neg_bit_shift_amount);
        result[2] = result[2] >> bit_shift_amount;
    }
}

void divide_uint192_inplace(uint64_tt *numerator, uint64_tt denominator, uint64_tt *quotient)
{
    // We expect 192-bit input
    size_t uint64_count = 3;

    // Clear quotient. Set it to zero.
    quotient[0] = 0;
    quotient[1] = 0;
    quotient[2] = 0;

    // Determine significant bits in numerator and denominator.
    int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
    int denominator_bits = get_significant_bit_count(denominator);

    // If numerator has fewer bits than denominator, then done.
    if (numerator_bits < denominator_bits) {
        return;
    }

    // Only perform computation up to last non-zero uint64s.
    uint64_count =  divide_round_up(numerator_bits, bits_per_uint64);

    // Handle fast case.
    if (uint64_count == 1) {
        *quotient = *numerator / denominator;
        *numerator -= *quotient * denominator;
        return;
    }

    // Create temporary space to store mutable copy of denominator.
    vector<uint64_tt> shifted_denominator(uint64_count, 0);
    shifted_denominator[0] = denominator;

    // Create temporary space to store difference calculation.
    vector<uint64_tt> difference(uint64_count);

    // Shift denominator to bring MSB in alignment with MSB of numerator.
    int denominator_shift = numerator_bits - denominator_bits;

    left_shift_uint192(shifted_denominator.data(), denominator_shift, shifted_denominator.data());
    denominator_bits += denominator_shift;

    // Perform bit-wise division algorithm.
    int remaining_shifts = denominator_shift;
    while (numerator_bits == denominator_bits) {
        // NOTE: MSBs of numerator and denominator are aligned.

        // Even though MSB of numerator and denominator are aligned,
        // still possible numerator < shifted_denominator.
        if (sub_uint(numerator, shifted_denominator.data(), uint64_count, difference.data())) {
            // numerator < shifted_denominator and MSBs are aligned,
            // so current quotient bit is zero and next one is definitely one.
            if (remaining_shifts == 0) {
                // No shifts remain and numerator < denominator so done.
                break;
            }

            // Effectively shift numerator left by 1 by instead adding
            // numerator to difference (to prevent overflow in numerator).
            add_uint(difference.data(), numerator, uint64_count, difference.data());

            // Adjust quotient and remaining shifts as a result of shifting numerator.
            left_shift_uint192(quotient, 1, quotient);
            remaining_shifts--;
        }
        // Difference is the new numerator with denominator subtracted.

        // Update quotient to reflect subtraction.
        quotient[0] |= 1;

        // Determine amount to shift numerator to bring MSB in alignment with denominator.
        numerator_bits = get_significant_bit_count_uint(difference.data(), uint64_count);
        int numerator_shift = denominator_bits - numerator_bits;
        if (numerator_shift > remaining_shifts) {
            // Clip the maximum shift to determine only the integer
            // (as opposed to fractional) bits.
            numerator_shift = remaining_shifts;
        }

        // Shift and update numerator.
        if (numerator_bits > 0) {
            left_shift_uint192(difference.data(), numerator_shift, numerator);
            numerator_bits += numerator_shift;
        } else {
            // Difference is zero so no need to shift, just set to zero.
            std::fill_n(numerator, uint64_count, uint64_tt(0));
        }

        // Adjust quotient and remaining shifts as a result of shifting numerator.
        left_shift_uint192(quotient, numerator_shift, quotient);
        remaining_shifts -= numerator_shift;
    }

    // Correct numerator (which is also the remainder) for shifting of
    // denominator, unless it is just zero.
    if (numerator_bits > 0) {
        right_shift_uint192(numerator, denominator_shift, numerator);
    }
}

template<typename T, typename = std::enable_if_t<is_uint64_v<T>>>
[[nodiscard]] inline uint64_tt barrett_reduce_64(T input, const uint64_tt &modulus)
{
    // Reduces input using base 2^64 Barrett reduction
    // floor(2^64 / mod) == floor( floor(2^128 / mod) )
    uint64_tt tmp[2];
    const uint64_tt const_ratio[3]{0, 0, 0};
    multiply_uint64_hw64_generic(input, const_ratio[1], tmp + 1);

    // Barrett subtraction
    tmp[0] = input - tmp[1] * modulus;

    // One more subtraction is enough
    return tmp[0] >= modulus ? tmp[0] - modulus : tmp[0];
}

inline void get_const_ratio(uint64_tt *const_ratio_, uint64_tt value )
{
    if (value == 0)
    {
        return ;
    }else if((value >> MOD_BIT_COUNT_MAX != 0) || (value == 1))
    {
        throw invalid_argument("value can be at most 61-bit and cannot be 1");
    }else
    {
        // Compute Barrett ratios for 64-bit words (barrett_reduce_128)
        uint64_tt numerator[3]{0, 0, 1};
        uint64_tt quotient[3]{0, 0, 0};

        // quotient = numerator（1<<128）/ value_,
        // numerator = numerator - quotient * value
        divide_uint192_inplace(numerator, value, quotient);

        const_ratio_[0] = quotient[0];
        const_ratio_[1] = quotient[1];

        // We store also the remainder
        const_ratio_[2] = numerator[0];
    }
}

template<typename T, typename = std::enable_if_t<is_uint64_v<T>>>
[[nodiscard]] inline uint64_tt barrett_reduce_128(const T *input, const uint64_tt &modulus)
{
    // Reduces input using base 2^64 Barrett reduction
    // input allocation size must be 128 bits
    // uint64_tt tmp1, tmp2[2], tmp3, carry;
    uint64_tt tmp1, tmp3, carry;
    std::vector<uint64_tt> tmp2(2);
    uint64_tt const_ratio[3] = {0, 0, 0};
    get_const_ratio(const_ratio , modulus);

    // Multiply input and const_ratio
    // Round 1
    multiply_uint64_hw64_generic(input[0], const_ratio[0], &carry);

    multiply_uint64(input[0], const_ratio[1], tmp2.data());

    tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

    // Round 2
    multiply_uint64(input[1], const_ratio[0], tmp2.data());

    carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

    // This is all we care about
    tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

    // Barrett subtraction
    tmp3 = input[0] - tmp1 * modulus;

    // One more subtraction is enough
    return tmp3 >= modulus ? tmp3 - modulus : tmp3;
}

//Returns value mod modulus | Correctness follows the condition of barrett_reduce_128.
[[nodiscard]] inline uint64_tt modulo_uint(const uint64_tt *value, const size_t value_uint64_count, const uint64_tt &modulus)
{
    if (value_uint64_count == 1)
    {
        // If value < modulus no operation is needed
        if (*value < modulus)
            return *value;
        else
            return barrett_reduce_64(*value, modulus);
    }
    // Temporary space for 128-bit reductions
    uint64_tt temp[2]{0, value[value_uint64_count - 1]};
    for (size_t k = value_uint64_count - 1; k--;)
    {
        temp[0] = value[k];
        temp[1] = barrett_reduce_128(temp, modulus);
    }
    // Save the result modulo i-th prime
    return temp[1];
}
inline uint64_tt compute_shoup(const uint64_tt operand, const uint64_tt modulus)
{
    // Using __uint128_t to avoid overflow during multiplication
    __uint128_t temp = operand;
    temp <<= 64; // multiplying by 2^64
    return temp / modulus;
}
[[nodiscard]] inline auto xgcd(uint64_tt x, uint64_tt y) -> std::tuple<uint64_tt, std::int64_t, std::int64_t>
{
    /* Extended GCD:
    Returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
    The numbers x, y are such that gcd = ax + by.
    */
    std::int64_t prev_a = 1;
    std::int64_t a = 0;
    std::int64_t prev_b = 0;
    std::int64_t b = 1;
    // printf("x : %llu , y : %llu\n", x , y);
    while (y != 0)
    {
        std::int64_t q = std::int64_t(x / y);
        std::int64_t temp = std::int64_t(x % y);
        x = y;
        y = uint64_tt(temp);

        temp = a;
        a = sub_safe(prev_a, mul_safe(q, a));
        prev_a = temp;

        // puts("okkk");

        temp = b;
        b = sub_safe(prev_b, mul_safe(q, b));
        prev_b = temp;

        // puts("okkk2");
    }
    // puts("done");
    return std::make_tuple(x, prev_a, prev_b);
}
bool try_invert_uint_mod(uint64_tt value, uint64_tt modulus, uint64_tt &result)
{
    if(value == 0)
    {
        return false;
    }
    auto gcd_tuple = xgcd(value, modulus);
    if(get<0>(gcd_tuple) != 1)
    {
        return false;
    }else if (get<1>(gcd_tuple) < 0)
    {
        result = static_cast<uint64_tt>(get<1>(gcd_tuple)) + modulus;
        return true;
    }else
    {
        result = static_cast<uint64_tt>(get<1>(gcd_tuple));
        return true;
    }
}

//------------------------------------------------------------------
// shift operators

inline void left_shift_uint(const uint64_tt *operand, int shift_amount, std::size_t uint64_count, uint64_tt *result)
{
    const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
    // How many words to shift
    std::size_t uint64_shift_amount = static_cast<std::size_t>(shift_amount) / bits_per_uint64_sz;

    // Shift words
    for (std::size_t i = 0; i < uint64_count - uint64_shift_amount; i++) {
        result[uint64_count - i - 1] = operand[uint64_count - i - 1 - uint64_shift_amount];
    }
    for (std::size_t i = uint64_count - uint64_shift_amount; i < uint64_count; i++) {
        result[uint64_count - i - 1] = 0;
    }

    // How many bits to shift in addition
    std::size_t bit_shift_amount =
            static_cast<std::size_t>(shift_amount) - (uint64_shift_amount * bits_per_uint64_sz);

    if (bit_shift_amount) {
        std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

        for (std::size_t i = uint64_count - 1; i > 0; i--) {
            result[i] = (result[i] << bit_shift_amount) | (result[i - 1] >> neg_bit_shift_amount);
        }
        result[0] = result[0] << bit_shift_amount;
    }
}
inline void right_shift_uint(const uint64_tt *operand, int shift_amount, std::size_t uint64_count, uint64_tt *result)
{
    const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
    // How many words to shift
    std::size_t uint64_shift_amount = static_cast<std::size_t>(shift_amount) / bits_per_uint64_sz;

    // Shift words
    for (std::size_t i = 0; i < uint64_count - uint64_shift_amount; i++)
    {
        result[i] = operand[i + uint64_shift_amount];
    }
    for (std::size_t i = uint64_count - uint64_shift_amount; i < uint64_count; i++)
    {
        result[i] = 0;
    }

    // How many bits to shift in addition
    std::size_t bit_shift_amount = static_cast<std::size_t>(shift_amount) - (uint64_shift_amount * bits_per_uint64_sz);

    if (bit_shift_amount)
    {
        std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

        for (std::size_t i = 0; i < uint64_count - 1; i++)
        {
            result[i] = (result[i] >> bit_shift_amount) | (result[i + 1] << neg_bit_shift_amount);
        }
        result[uint64_count - 1] = result[uint64_count - 1] >> bit_shift_amount;
    }
}
inline void left_shift_uint128(const uint64_tt *operand, int shift_amount, uint64_tt *result)
{
    const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);
    const std::size_t shift_amount_sz = static_cast<std::size_t>(shift_amount);

    // Early return
    if (shift_amount_sz & bits_per_uint64_sz)
    {
        result[1] = operand[0];
        result[0] = 0;
    } else {
        result[1] = operand[1];
        result[0] = operand[0];
    }

    // How many bits to shift in addition to word shift
    std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

    // Do we have a word shift
    if (bit_shift_amount)
    {
        std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

        // Warning: if bit_shift_amount == 0 this is incorrect
        result[1] = (result[1] << bit_shift_amount) | (result[0] >> neg_bit_shift_amount);
        result[0] = result[0] << bit_shift_amount;
    }
}

inline void right_shift_uint128(const uint64_tt *operand, int shift_amount, uint64_tt *result)
{
    const std::size_t bits_per_uint64_sz = static_cast<std::size_t>(bits_per_uint64);

    const std::size_t shift_amount_sz = static_cast<std::size_t>(shift_amount);

    if (shift_amount_sz & bits_per_uint64_sz)
    {
        result[0] = operand[1];
        result[1] = 0;
    } else {
        result[1] = operand[1];
        result[0] = operand[0];
    }

    // How many bits to shift in addition to word shift
    std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

    if (bit_shift_amount)
    {
        std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

        // Warning: if bit_shift_amount == 0 this is incorrect
        result[0] = (result[0] >> bit_shift_amount) | (result[1] << neg_bit_shift_amount);
        result[1] = result[1] >> bit_shift_amount;
    }
}
//-----------------------------------------------------------------------------
void divide_uint_inplace(uint64_tt *numerator, const uint64_tt *denominator, size_t uint64_count, uint64_tt *quotient)
{
    if (!uint64_count)
    {
        return;
    }

    // Clear quotient. Set it to zero.
    std::fill_n(quotient, uint64_count, uint64_tt(0));
    // Determine significant bits in numerator and denominator.
    int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
    int denominator_bits = get_significant_bit_count_uint(denominator, uint64_count);

    // If numerator has fewer bits than denominator, then done.
    if (numerator_bits < denominator_bits)
    {
        return;
    }

    // Only perform computation up to last non-zero uint64s.
    uint64_count = divide_round_up(numerator_bits, bits_per_uint64);

    // Handle fast case.
    if (uint64_count == 1)
    {
        *quotient = *numerator / *denominator;
        *numerator -= *quotient * *denominator;
        return;
    }
    std::vector<uint64_tt> alloc_anchor(uint64_count << 1);

    // Create temporary space to store mutable copy of denominator.
    uint64_tt *shifted_denominator = alloc_anchor.data();

    // Create temporary space to store difference calculation.
    uint64_tt *difference = shifted_denominator + uint64_count;

    // Shift denominator to bring MSB in alignment with MSB of numerator.
    int denominator_shift = numerator_bits - denominator_bits;
    left_shift_uint(denominator, denominator_shift, uint64_count, shifted_denominator);
    denominator_bits += denominator_shift;

    // Perform bit-wise division algorithm.
    int remaining_shifts = denominator_shift;
    while (numerator_bits == denominator_bits)
    {
        // NOTE: MSBs of numerator and denominator are aligned.

        // Even though MSB of numerator and denominator are aligned,
        // still possible numerator < shifted_denominator.
        if (sub_uint(numerator, shifted_denominator, uint64_count, difference))
        {
            // numerator < shifted_denominator and MSBs are aligned,
            // so current quotient bit is zero and next one is definitely one.
            if (remaining_shifts == 0) {
                // No shifts remain and numerator < denominator so done.
                break;
            }

            // Effectively shift numerator left by 1 by instead adding
            // numerator to difference (to prevent overflow in numerator).
            add_uint(difference, numerator, uint64_count, difference);

            // Adjust quotient and remaining shifts as a result of
            // shifting numerator.
            left_shift_uint(quotient, 1, uint64_count, quotient);
            remaining_shifts--;
        }
        // Difference is the new numerator with denominator subtracted.

        // Update quotient to reflect subtraction.
        quotient[0] |= 1;

        // Determine amount to shift numerator to bring MSB in alignment
        // with denominator.
        numerator_bits = get_significant_bit_count_uint(difference, uint64_count);
        int numerator_shift = denominator_bits - numerator_bits;
        if (numerator_shift > remaining_shifts) {
            // Clip the maximum shift to determine only the integer
            // (as opposed to fractional) bits.
            numerator_shift = remaining_shifts;
        }

        // Shift and update numerator.
        if (numerator_bits > 0) {
            left_shift_uint(difference, numerator_shift, uint64_count, numerator);
            numerator_bits += numerator_shift;
        } else {
            // Difference is zero so no need to shift, just set to zero.
            std::fill_n(numerator, uint64_count, uint64_tt(0));
        }

        // Adjust quotient and remaining shifts as a result of shifting numerator.
        left_shift_uint(quotient, numerator_shift, uint64_count, quotient);
        remaining_shifts -= numerator_shift;
    }

    // Correct numerator (which is also the remainder) for shifting of
    // denominator, unless it is just zero.
    if (numerator_bits > 0) {
        right_shift_uint(numerator, denominator_shift, uint64_count, numerator);
    }
}
inline void divide_uint(
            const uint64_tt *numerator, const uint64_tt *denominator, size_t uint64_count,
            uint64_tt *quotient, uint64_tt *remainder)
{
        set_uint(numerator, uint64_count, remainder);
        divide_uint_inplace(remainder, denominator, uint64_count, quotient);
}

[[nodiscard]] __inline__ __device__ uint64_tt multiply_and_reduce_shoup(const uint64_tt& operand1,
                                                                        const uint64_tt& operand2,
                                                                        const uint64_tt& operand2_shoup,
                                                                        const uint64_tt& modulus)
{
    const uint64_tt hi = __umul64hi(operand1, operand2_shoup);
    uint64_tt res = operand1 * operand2 - hi * modulus;
    csub_q(res, modulus);
    return res;
}

bool is_prime(const uint64_tt num)
{
    if (num <= 1) return false;  
    if (num == 2) return true;   
    if (num % 2 == 0) return false;

    for (uint64_tt i = 3; i * i <= num; i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

vector <uint64_tt> get_primes_below(size_t ntt_size, uint64_tt upper_bound, size_t count)
{
    vector<uint64_tt> destination;
    uint64_tt factor = mul_safe(uint64_tt(2), uint64_tt(ntt_size));

    uint64_tt value = upper_bound;
    int bit_size = get_significant_bit_count(upper_bound);
    // Start with value - 2 * ntt_size
    // try {
    //     value = sub_safe(value, factor);
    // }
    // catch (const logic_error &) {
    //     throw logic_error("failed to find enough qualifying primes 1");
    // }
    value -= factor;
    // printf("value : %llu\n" , value);

    uint64_tt lower_bound = uint64_tt(0x1) << (bit_size - 1);

    while (count > 0 && value > lower_bound)
    {
        uint64_tt new_mod(value);
        if (is_prime(new_mod)) {
            destination.emplace_back(new_mod);
            count--;
        }
        value -= factor;
    }
    if (count > 0) {
        throw logic_error("failed to find enough qualifying primes 2");
    }

    return destination;
}

[[nodiscard]] inline uint64_tt multiply_uint_mod(
        uint64_tt operand1, uint64_tt operand2, const uint64_tt &modulus)
{
    uint64_tt z[2];
    multiply_uint64(operand1, operand2, z);
    return barrett_reduce_128(z, modulus);
}

__forceinline__ __device__ void ld_two_uint64(uint64_tt& x, uint64_tt& y, const uint64_tt* ptr)
{
    x = ptr[0];
    y = ptr[1];
}
__forceinline__ __device__ void st_two_uint64(uint64_tt* ptr, const uint64_tt& x, const uint64_tt& y)
{
    ptr[0] = x;
    ptr[1] = y;
}

__forceinline__ __device__ uint64_tt multiply_and_barrett_reduce_uint64(const uint64_tt& operand1,
                                                                           const uint64_tt& operand2,
                                                                           const uint64_tt& modulus,
                                                                           const uint64_tt* barrett_mu)
{
    const uint128_tt product = multiply_uint64_uint64(operand1, operand2);
    return barrett_reduce_uint128_uint64(product, modulus, barrett_mu);
}