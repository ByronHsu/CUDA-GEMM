nsys profile --force-overwrite true -o matrixMulMultiGPUTiling_onlyKernel_N1 ./matrixMulMultiGPUTiling 1 1024 1024 1024 1024
nsys profile --force-overwrite true -o matrixMulMultiGPUTiling_onlyKernel_N2 ./matrixMulMultiGPUTiling 2 1024 1024 1024 1024
nsys profile --force-overwrite true -o matrixMulMultiGPUTiling_onlyKernel_N3 ./matrixMulMultiGPUTiling 3 1024 1024 1024 1024
nsys profile --force-overwrite true -o matrixMulMultiGPUTiling_onlyKernel_N1 ./matrixMulMultiGPUTiling 4 1024 1024 1024 1024