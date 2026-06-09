/**
 * @file tehm.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 * (create) 2021-10-06 (rev) 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_FRAMEWORK
#define CUSZ_FRAMEWORK

#include <type_traits>

#include "compressor.hh"
#include "cusz/type.h"
#include "hf/hf.hh"

namespace psz {

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

};  // namespace psz

namespace cusz {

template <
    typename InDtype,
    typename ErrCtrl = u1,
    typename Metadata = u4,
    bool FastLowPrecision = std::is_same<InDtype, f4>::value>
struct TEHM {
 public:
  /**
   *
   *  Compressor<T, E, (FP)>
   *             |  |   ^
   *  outlier <--+  |   +---- default fast-low-precision
   *                v
   *        Encoder<E, H>
  */

  using T = InDtype;
  using E = ErrCtrl;  // predefined for mem. overlapping
  using FP = typename FastLowPrecisionTrait<FastLowPrecision>::type;
  // using H = u4;
  // using Hfailsafe = u8;
  using M = Metadata;

  /* Lossless Codec*/
  using Codec = cusz::HuffmanCodec<E, M>;
};

using CompressorF4 = cusz::Compressor<cusz::TEHM<f4, u1, u4, true>>;
using CompressorF8 = cusz::Compressor<cusz::TEHM<f8, u1, u4, false>>;
using CompressorF8Fast = cusz::Compressor<cusz::TEHM<f8, u1, u4, true>>;
using CompressorF8U2 = cusz::Compressor<cusz::TEHM<f8, u2, u4, false>>;
using CompressorF8U4 = cusz::Compressor<cusz::TEHM<f8, u4, u4, false>>;

}  // namespace cusz

#endif
