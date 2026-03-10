rm -rf build
mkdir build
cmake . -B build \
	-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
	-D PSZ_BACKEND=cuda \
	-D PSZ_BUILD_EXAMPLES=off \
	-D CMAKE_CUDA_ARCHITECTURES="89" \
	-D CMAKE_BUILD_TYPE=Release
cmake --build build -- -j
cd build
