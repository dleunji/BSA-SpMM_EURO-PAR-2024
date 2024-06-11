#!/bin/bash
N_COL=128;
ROOT_DIR="${PRJ_DIR}/data/dlmc";
MODELS=( "transformer" "rn50" );
SUB_DATASET=( "variational_dropout" "magnitude_pruning" "random_pruning" );
OUTPUT_FILE="${PRJ_DIR}/result/dlmc-sputnik.csv";


mkdir -p result

if [ -e ${OUTPUT_FILE} ]; then
    rm -rf ${OUTPUT_FILE}
    touch ${OUTPUT_FILE}
fi

echo "matrix,avg_total_time" >> ${OUTPUT_FILE}
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
                    echo $FILE
                    if [ $CNT -eq $STOP ]; then
                        break
                    fi
                    python3.9 ${PRJ_DIR}/baselines/sputnik/ncu_profile_iter.py --bm ${FILE} -k ${N_COL} -v 1 --kernel sputnik --print --prof --job spmm --precision half --iter 2 >> ${OUTPUT_FILE}
                    CNT=$(($CNT+1))
                done
            fi
        done
    done
done