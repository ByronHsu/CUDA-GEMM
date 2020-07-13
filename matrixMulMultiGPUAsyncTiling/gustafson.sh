#!/bin/bash

#!/bin/bash

for i in {1..8}
do
    let j=1024*i
    echo $j
    ./matrixMulMultiGPUTiling 1 $j 1024 1024 $j
    ./matrixMulMultiGPUTiling $i $j 1024 1024 $j
done

# ./matrixMulMultiGPUTiling 1 1024 1024 1024 1024

# ./matrixMulMultiGPUTiling 1 1024 1024 1024 1024

# ./matrixMulMultiGPUTiling 2 4096 4096 4096 4096
# ./matrixMulMultiGPUTiling 3 4096 4096 4096 4096
# ./matrixMulMultiGPUTiling 4 4096 4096 4096 4096
# ./matrixMulMultiGPUTiling 5 4096 4096 4096 4096
# ./matrixMulMultiGPUTiling 6 4096 4096 4096 4096
# ./matrixMulMultiGPUTiling 7 4096 4096 4096 4096
# ./matrixMulMultiGPUTiling 8 4096 4096 4096 4096