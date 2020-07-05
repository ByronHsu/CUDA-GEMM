#!/bin/bash

for i in {1..8}
do
   ./matrixMulMultiGPUTiling $i 4096 4096 4096 4096
done