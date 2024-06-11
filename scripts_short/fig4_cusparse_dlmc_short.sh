#!/bin/bash
BLOCK_SIZE=32;
N_COL=128;
ROOT_DIR="${PRJ_DIR}/data/dlmc";
MODELS=( "transformer" "rn50" );
SUB_DATASET=( "variational_dropout" "magnitude_pruning" "random_pruning" );
OUTPUT_FILE="${PRJ_DIR}/result/dlmc-cusparse.csv";

mkdir -p result

if [ -e ${OUTPUT_FILE} ]; then
    rm -rf ${OUTPUT_FILE}
    touch ${OUTPUT_FILE}
fi
echo "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,alpha,delta,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method,spmm" >> ${OUTPUT_FILE}
for MODEL in "${MODELS[@]}";do
    for SUB in "${SUB_DATASET[@]}"; do
        for SP in "$ROOT_DIR/$MODEL/$SUB"/*; do
            # echo $SP
            SPARSITY=$(basename $SP)
            SPARSITY=$(basename $SPARSITY)

            # 50% <= sparsity <= 90%
            SPARSITY=$(echo "$SPARSITY <= 0.9" | bc)
            # echo $SPARSITY
            if [ $SPARSITY -eq 1 ]; then
                CNT=0
                STOP=3
                for FILE in $SP/*; do
                    echo $CNT
                    if [ $CNT -eq $STOP ]; then
                        break
                    fi
                    echo $FILE
                    ./bsa_spmm_benchmark -f ${FILE} -i 0 -n ${N_COL} -z 0 -c 0 -x 2 -p 1 -m 0 -v 0 -o ${OUTPUT_FILE} -s 1
                    CNT=$(($CNT+1))
                done
            fi
        done
    done
done