#!/bin/bash
BLOCK_SIZE=32;
TAU=( 0.1 0.3 0.5 0.7 0.9 );
N_COL=128;
ROOT_DIR="${PRJ_DIR}/data/dlmc";
MODELS=( "transformer" "rn50" );
SUB_DATASET=( "variational_dropout" "magnitude_pruning" "random_pruning" );
OUTPUT_FILE="${PRJ_DIR}/result/dlmc-1-sa.csv";

mkdir -p ${PRJ_DIR}/result

if [ -e ${OUTPUT_FILE} ]; then
    rm -rf ${OUTPUT_FILE}
    touch ${OUTPUT_FILE}
fi
    echo "matrix,rows,cols,nonzeros,symmetrize,blocking_algo,tau,row_block_size,col_block_size,use_pattern,sim_use_groups,sim_measure,reorder,exp_name,b_cols,warmup,exp_repetitions,multiplication_algo,n_streams,time_to_block,time_to_merge,time_to_compare,VBR_nzcount,VBR_nzblocks_count,VBR_average_height,VBR_longest_row,avg_time_multiply," >> ${OUTPUT_FILE}

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
                for FILE in $SP/*; do
                    echo $FILE
                    for T in "${TAU[@]}"; do
                        # To use Tensor core, fixed size is required.
                        ${PRJ_DIR}/baselines/1-sa/programs/cuda/cuda_multiply -a 5 -b ${BLOCK_SIZE} -B ${BLOCK_SIZE} -f ${FILE} -F 1 -g 1 -o ${OUTPUT_FILE} -p 1 -P 1 -m 1 -M 4 -c ${N_COL} -w 0 -x 10 -r 0 -t ${T}
                    done
                done
            fi
        done
    done
done