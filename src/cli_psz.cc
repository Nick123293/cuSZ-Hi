#include <fstream>
#include <stdexcept>

#include "header.h"
#include "pipeline/cli.inl"
#include "port.hh"
#include "utils/query.hh"

namespace {

void infer_reconstruct_dtype_from_header(psz_context* ctx)
{
  if (not ctx->task_reconstruct) return; //Ignored unless doing decompression

  std::ifstream in(ctx->infile, std::ios::binary);
  if (not in) {
    throw std::runtime_error(
        "failed to open compressed input for dtype inference.");
  }

  psz_header header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(header))) {
    throw std::runtime_error(
        "compressed input is too small to contain a cuSZ-Hi header.");
  }

  if (header.dtype != F4 and header.dtype != F8) {
    throw std::runtime_error("compressed input has an unsupported dtype.");
  }

  ctx->dtype = header.dtype;
}

}  // namespace

int main(int argc, char** argv)
{
  auto ctx = new psz_context;
  pszctx_create_from_argv(ctx, argc, argv);
  infer_reconstruct_dtype_from_header(ctx);

  if (ctx->verbose) {
    CPU_QUERY;
    GPU_QUERY;
  }

  if (ctx->dtype == F4) {
    cusz::CLI<f4> cusz_cli;
    cusz_cli.dispatch(ctx);
  }
  else if (ctx->dtype == F8) {
    cusz::CLI<f8> cusz_cli;
    cusz_cli.dispatch(ctx);
  }
  else {
    throw std::runtime_error("Unsupported dtype.");
  }

  delete ctx;
}
