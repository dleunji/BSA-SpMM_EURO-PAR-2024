#!/bin/bash
BLOCK_SIZE=32;
ALPHA=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.56 0.5 0.55 0.6 0.64 0.7 0.75 0.8 0.85 0.9 0.95 );
N_COL=0;
ROOT_DIR="${PRJ_DIR}/data/suitesparse_mtx";
OUTPUT_FILE="${PRJ_DIR}/result/suitesparse_reorder.csv";

mkdir -p result

if [ -e ${OUTPUT_FILE} ]; then
    rm -rf ${OUTPUT_FILE}
    touch ${OUTPUT_FILE}
fi
echo "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,alpha,delta,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method,spmm" >> ${OUTPUT_FILE}

for SP in "$ROOT_DIR"/*; do
    echo $SP
    for A in ${ALPHA[@]}; do
        ./bsa_spmm_benchmark -f ${SP} -i 1 -z 1 -b ${BLOCK_SIZE} -a ${A} -c 1 -n ${N_COL} -x 10 -p 1 -m 2 -v 0 -o ${OUTPUT_FILE} -s 0
    done

done