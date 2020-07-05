#!/bin/bash
j=1024
for i in {1..4}
do
   ./matrixMulMultiGPUTiling $i $j $j $j $j
done

j=2048
for i in {1..4}
do
   ./matrixMulMultiGPUTiling $i $j $j $j $j
done

j=4096
for i in {1..4}
do
   ./matrixMulMultiGPUTiling $i $j $j $j $j
done