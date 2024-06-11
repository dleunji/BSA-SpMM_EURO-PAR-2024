#!/bin/bash
BLOCK_SIZE=32;
DELTA=( 0.2 0.4 0.6 0.8 );
N_COL=128;
ROOT_DIR="${PRJ_DIR}/data/dlmc";
INFILE="$ROOT_DIR/2048_512_dlmc_data.txt"
MODELS=( "transformer" "rn50" );
# SUB_DATASET=( "variational_dropout" "magnitude_pruning" "random_pruning" );
OUTPUT_FILE_NON="${PRJ_DIR}/result/reordering-dlmc-non.csv";
OUTPUT_FILE_BSA="${PRJ_DIR}/result/reordering-dlmc-bsa.csv";

mkdir -p result

if [ -e ${OUTPUT_FILE_NON} ]; then
    rm -rf ${OUTPUT_FILE_NON}
    touch ${OUTPUT_FILE_NON}
fi

if [ -e ${OUTPUT_FILE_BSA} ]; then
    rm -rf ${OUTPUT_FILE_BSA}
    touch ${OUTPUT_FILE_BSA}
fi

echo "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,alpha,delta,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method,spmm" >> ${OUTPUT_FILE_BSA}
echo "matrix,avg_reordering_time,avg_csr_spmm_time,avg_bellpack_spmm_time,avg_total_time,avg_density_of_tiles,alpha,delta,num_tiles,nnz_in_bellpack,nnz_in_csr,cluster_cnt,n_cols,rows,cols,block_size,method,spmm" >> ${OUTPUT_FILE_NON}
CNT=0
STOP=120
FILE=$(cat $INFILE)
for LINE in $FILE;do
    if [ $CNT -eq $STOP ]; then
        break
    fi 
    echo $CNT
    FILE="$ROOT_DIR/$LINE.smtx"
    echo $FILE
    for D in ${DELTA[@]}; do
        # BSA reordering
        ./reordering_benchmark -f ${FILE} -i 0 -z 1 -b ${BLOCK_SIZE} -d ${D}-n ${N_COL} -x 2 -p 1 -m 2 -v 0 -o ${OUTPUT_FILE_BSA}
        # non-reordering
        ./reordering_benchmark -f ${FILE} -i 0 -z 1 -b ${BLOCK_SIZE} -d ${D}-n ${N_COL} -x 2 -p 1 -m 0 -v 0 -o ${OUTPUT_FILE_NON}
    done
    CNT=$(($CNT+1))
done