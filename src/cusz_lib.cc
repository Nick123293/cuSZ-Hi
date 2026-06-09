/**
 * @file cusz_lib.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-01
 * (rev.1) 2023-01-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "compressor.hh"
#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "hf/hf.hh"
#include "port.hh"
#include "tehm.hh"

pszpredictor pszdefault_predictor() { return {Spline}; }
pszquantizer pszdefault_quantizer() { return {128}; }
pszhfrc pszdefault_hfcoder() { return {Sword, Coarse, 1024, 768}; }
pszframe* pszdefault_framework()
{
  return new pszframe{
      pszdefault_predictor(), pszdefault_quantizer(), pszdefault_hfcoder(),
      20};
}

namespace {

void ensure_supported_pipeline(psz_dtype dtype, psz_predtype pred_type)
{
  if (dtype == F8 and pred_type == Spline) {
    throw std::runtime_error(
        "F8 with Spline is not implemented yet; use the Lorenzo predictor "
        "for double-precision data.");
  }
}

void ensure_matching_dtype(psz_dtype requested, psz_dtype actual)
{
  if (requested != actual) {
    throw std::runtime_error(
        "Compressor dtype does not match the context/archive dtype.");
  }
}

}  // namespace

pszcompressor* psz_create(pszframe* _framework, psz_dtype _type)
{
  auto comp = new pszcompressor{.framework = _framework, .type = _type};

  if (comp->type == F4) {
    using Compressor = cusz::CompressorF4;
    comp->compressor = new Compressor();
  }
  else if (comp->type == F8) {
    using Compressor = cusz::CompressorF8;
    comp->compressor = new Compressor();
  }
  else {
    throw std::runtime_error("Type is not supported.");
  }

  return comp;
}

pszerror psz_release(pszcompressor* comp)
{
  delete comp;
  return CUSZ_SUCCESS;
}

// TODO config is redundant when it comes to CLI
pszerror psz_compress_init(
    pszcompressor* comp, pszlen const uncomp_len, pszctx* ctx)
{
  comp->ctx = ctx;
  comp->ctx->dtype = comp->type;
  pszctx_set_len(comp->ctx, uncomp_len);
  ensure_supported_pipeline(comp->type, comp->ctx->pred_type);

  // Be cautious of autotuning! The default value of pardeg is not robust.
  cusz::CompressorHelper::autotune_coarse_parhf(comp->ctx);

  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);
    cor->init(comp->ctx);
  }
  else if (comp->type == F8) {
    auto cor = (cusz::CompressorF8*)(comp->compressor);
    cor->init(comp->ctx);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror psz_compress(
    pszcompressor* comp, void* in, pszlen const uncomp_len,
    ptr_pszout compressed, size_t* comp_bytes, pszheader* header, void* record,
    void* stream)
{
  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);

    cor->compress(
        comp->ctx, (f4*)(in), compressed, comp_bytes, stream);
    cor->export_header(*header);
    cor->export_timerecord((psz::TimeRecord*)record);
  }
  else if (comp->type == F8) {
    auto cor = (cusz::CompressorF8*)(comp->compressor);

    cor->compress(
        comp->ctx, (f8*)(in), compressed, comp_bytes, stream);
    cor->export_header(*header);
    cor->export_timerecord((psz::TimeRecord*)record);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror psz_decompress_init(pszcompressor* comp, pszheader* header)
{
  comp->header = header;
  ensure_matching_dtype(comp->type, header->dtype);
  ensure_supported_pipeline(header->dtype, header->pred_type);

  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);
    cor->init(header);
  }
  else if (comp->type == F8) {
    auto cor = (cusz::CompressorF8*)(comp->compressor);
    cor->init(header);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror psz_decompress(
    pszcompressor* comp, pszout compressed, size_t const comp_len,
    void* decompressed, void* outlier_tmp, pszlen const decomp_len, void* record, void* stream)
{
  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);

    cor->decompress(
        comp->header, compressed, (f4*)(decompressed), (f4*)(outlier_tmp), (GpuStreamT)stream);
    cor->export_timerecord((psz::TimeRecord*)record);
  }
  else if (comp->type == F8) {
    auto cor = (cusz::CompressorF8*)(comp->compressor);

    cor->decompress(
        comp->header, compressed, (f8*)(decompressed), (f8*)(outlier_tmp), (GpuStreamT)stream);
    cor->export_timerecord((psz::TimeRecord*)record);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}
