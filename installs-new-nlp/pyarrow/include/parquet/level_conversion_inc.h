// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#pragma once

#include "parquet/level_conversion.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_writer.h"
#include "arrow/util/logging.h"
#include "arrow/util/simd.h"
#include "parquet/exception.h"
#include "parquet/level_comparison.h"

namespace parquet {
namespace internal {
#ifndef PARQUET_IMPL_NAMESPACE
#error "PARQUET_IMPL_NAMESPACE must be defined"
#endif
namespace PARQUET_IMPL_NAMESPACE {

// clang-format off
/* Python code to generate lookup table:

kLookupBits = 5
count = 0
print('constexpr int kLookupBits = {};'.format(kLookupBits))
print('constexpr uint8_t kPextTable[1 << kLookupBits][1 << kLookupBits] = {')
print(' ', end = '')
for mask in range(1 << kLookupBits):
    for data in range(1 << kLookupBits):
        bit_value = 0
        bit_len = 0
        for i in range(kLookupBits):
            if mask & (1 << i):
                bit_value |= (((data >> i) & 1) << bit_len)
                bit_len += 1
        out = '0x{:02X},'.format(bit_value)
        count += 1
        if count % (1 << kLookupBits) == 1:
            print(' {')
        if count % 8 == 1:
            print('    ', end = '')
        if count % 8 == 0:
            print(out, end = '\n')
        else:
            print(out, end = ' ')
        if count % (1 << kLookupBits) == 0:
            print('  },', end = '')
print('\n};')

*/
// clang-format on

constexpr int kLookupBits = 5;
constexpr uint8_t kPextTable[1 << kLookupBits][1 << kLookupBits] = {
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02,
        0x03, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01,
        0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02, 0x03, 0x00, 0x01, 0x00,
        0x01, 0x02, 0x03, 0x02, 0x03, 0x00, 0x01, 0x00, 0x01, 0x02, 0x03,
        0x02, 0x03, 0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02, 0x03,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x00, 0x00, 0x01,
        0x01, 0x02, 0x02, 0x03, 0x03, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02,
        0x03, 0x03, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x01, 0x02,
        0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
        0x06, 0x07, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02,
        0x03, 0x02, 0x03, 0x02, 0x03, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01,
        0x00, 0x01, 0x02, 0x03, 0x02, 0x03, 0x02, 0x03, 0x02, 0x03,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03,
        0x03, 0x02, 0x02, 0x03, 0x03, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x02, 0x02, 0x03, 0x03,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x07, 0x04, 0x05, 0x06, 0x07, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01,
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x04, 0x05, 0x06, 0x07,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02,
        0x02, 0x03, 0x03, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
        0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02, 0x03, 0x04, 0x05, 0x04,
        0x05, 0x06, 0x07, 0x06, 0x07, 0x00, 0x01, 0x00, 0x01, 0x02, 0x03,
        0x02, 0x03, 0x04, 0x05, 0x04, 0x05, 0x06, 0x07, 0x06, 0x07,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05,
        0x05, 0x06, 0x06, 0x07, 0x07, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02,
        0x03, 0x03, 0x04, 0x04, 0x05, 0x05, 0x06, 0x06, 0x07, 0x07,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
        0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
        0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02, 0x03, 0x02, 0x03,
        0x02, 0x03, 0x02, 0x03, 0x02, 0x03, 0x02, 0x03, 0x02, 0x03,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x02, 0x02,
        0x03, 0x03, 0x02, 0x02, 0x03, 0x03, 0x02, 0x02, 0x03, 0x03,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02,
        0x03, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x04, 0x05,
        0x06, 0x07, 0x04, 0x05, 0x06, 0x07, 0x04, 0x05, 0x06, 0x07,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03,
        0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02, 0x03, 0x00, 0x01, 0x00,
        0x01, 0x02, 0x03, 0x02, 0x03, 0x04, 0x05, 0x04, 0x05, 0x06, 0x07,
        0x06, 0x07, 0x04, 0x05, 0x04, 0x05, 0x06, 0x07, 0x06, 0x07,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x00, 0x00, 0x01,
        0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05, 0x05, 0x06, 0x06,
        0x07, 0x07, 0x04, 0x04, 0x05, 0x05, 0x06, 0x06, 0x07, 0x07,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x01, 0x02,
        0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
        0x0E, 0x0F, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02,
        0x03, 0x02, 0x03, 0x02, 0x03, 0x04, 0x05, 0x04, 0x05, 0x04, 0x05,
        0x04, 0x05, 0x06, 0x07, 0x06, 0x07, 0x06, 0x07, 0x06, 0x07,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03,
        0x03, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05, 0x05, 0x04, 0x04,
        0x05, 0x05, 0x06, 0x06, 0x07, 0x07, 0x06, 0x06, 0x07, 0x07,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x07, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x08, 0x09,
        0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x0C, 0x0D, 0x0E, 0x0F,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02,
        0x02, 0x03, 0x03, 0x03, 0x03, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05,
        0x05, 0x05, 0x06, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x07,
    },
    {
        0x00, 0x01, 0x00, 0x01, 0x02, 0x03, 0x02, 0x03, 0x04, 0x05, 0x04,
        0x05, 0x06, 0x07, 0x06, 0x07, 0x08, 0x09, 0x08, 0x09, 0x0A, 0x0B,
        0x0A, 0x0B, 0x0C, 0x0D, 0x0C, 0x0D, 0x0E, 0x0F, 0x0E, 0x0F,
    },
    {
        0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05,
        0x05, 0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x09, 0x09, 0x0A, 0x0A,
        0x0B, 0x0B, 0x0C, 0x0C, 0x0D, 0x0D, 0x0E, 0x0E, 0x0F, 0x0F,
    },
    {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
        0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
        0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    },
};

inline uint64_t ExtractBitsSoftware(uint64_t bitmap, uint64_t select_bitmap) {
  // A software emulation of _pext_u64

  // These checks should be inline and are likely to be common cases.
  if (select_bitmap == ~uint64_t{0}) {
    return bitmap;
  } else if (select_bitmap == 0) {
    return 0;
  }

  // Fallback to lookup table method
  uint64_t bit_value = 0;
  int bit_len = 0;
  constexpr uint8_t kLookupMask = (1U << kLookupBits) - 1;
  while (select_bitmap != 0) {
    const auto mask_len = ARROW_POPCOUNT32(select_bitmap & kLookupMask);
    const uint64_t value = kPextTable[select_bitmap & kLookupMask][bitmap & kLookupMask];
    bit_value |= (value << bit_len);
    bit_len += mask_len;
    bitmap >>= kLookupBits;
    select_bitmap >>= kLookupBits;
  }
  return bit_value;
}

#ifdef ARROW_HAVE_BMI2

// Use _pext_u64 on 64-bit builds, _pext_u32 on 32-bit builds,
#if UINTPTR_MAX == 0xFFFFFFFF

using extract_bitmap_t = uint32_t;
inline extract_bitmap_t ExtractBits(extract_bitmap_t bitmap,
                                    extract_bitmap_t select_bitmap) {
  return _pext_u32(bitmap, select_bitmap);
}

#else

using extract_bitmap_t = uint64_t;
inline extract_bitmap_t ExtractBits(extract_bitmap_t bitmap,
                                    extract_bitmap_t select_bitmap) {
  return _pext_u64(bitmap, select_bitmap);
}

#endif

#else  // !defined(ARROW_HAVE_BMI2)

// Use 64-bit pext emulation when BMI2 isn't available.
using extract_bitmap_t = uint64_t;
inline extract_bitmap_t ExtractBits(extract_bitmap_t bitmap,
                                    extract_bitmap_t select_bitmap) {
  return ExtractBitsSoftware(bitmap, select_bitmap);
}

#endif

static constexpr int64_t kExtractBitsSize = 8 * sizeof(extract_bitmap_t);

template <bool has_repeated_parent>
int64_t DefLevelsBatchToBitmap(const int16_t* def_levels, const int64_t batch_size,
                               int64_t upper_bound_remaining, LevelInfo level_info,
                               ::arrow::internal::FirstTimeBitmapWriter* writer) {
  DCHECK_LE(batch_size, kExtractBitsSize);

  // Greater than level_info.def_level - 1 implies >= the def_level
  auto defined_bitmap = static_cast<extract_bitmap_t>(
      internal::GreaterThanBitmap(def_levels, batch_size, level_info.def_level - 1));

  if (has_repeated_parent) {
    // Greater than level_info.repeated_ancestor_def_level - 1 implies >= the
    // repeated_ancestor_def_level
    auto present_bitmap = static_cast<extract_bitmap_t>(internal::GreaterThanBitmap(
        def_levels, batch_size, level_info.repeated_ancestor_def_level - 1));
    auto selected_bits = ExtractBits(defined_bitmap, present_bitmap);
    int64_t selected_count = ::arrow::bit_util::PopCount(present_bitmap);
    if (ARROW_PREDICT_FALSE(selected_count > upper_bound_remaining)) {
      throw ParquetException("Values read exceeded upper bound");
    }
    writer->AppendWord(selected_bits, selected_count);
    return ::arrow::bit_util::PopCount(selected_bits);
  } else {
    if (ARROW_PREDICT_FALSE(batch_size > upper_bound_remaining)) {
      std::stringstream ss;
      ss << "Values read exceeded upper bound";
      throw ParquetException(ss.str());
    }

    writer->AppendWord(defined_bitmap, batch_size);
    return ::arrow::bit_util::PopCount(defined_bitmap);
  }
}

template <bool has_repeated_parent>
void DefLevelsToBitmapSimd(const int16_t* def_levels, int64_t num_def_levels,
                           LevelInfo level_info, ValidityBitmapInputOutput* output) {
  ::arrow::internal::FirstTimeBitmapWriter writer(
      output->valid_bits,
      /*start_offset=*/output->valid_bits_offset,
      /*length=*/output->values_read_upper_bound);
  int64_t set_count = 0;
  output->values_read = 0;
  int64_t values_read_remaining = output->values_read_upper_bound;
  while (num_def_levels > kExtractBitsSize) {
    set_count += DefLevelsBatchToBitmap<has_repeated_parent>(
        def_levels, kExtractBitsSize, values_read_remaining, level_info, &writer);
    def_levels += kExtractBitsSize;
    num_def_levels -= kExtractBitsSize;
    values_read_remaining = output->values_read_upper_bound - writer.position();
  }
  set_count += DefLevelsBatchToBitmap<has_repeated_parent>(
      def_levels, num_def_levels, values_read_remaining, level_info, &writer);

  output->values_read = writer.position();
  output->null_count += output->values_read - set_count;
  writer.Finish();
}

}  // namespace PARQUET_IMPL_NAMESPACE
}  // namespace internal
}  // namespace parquet
