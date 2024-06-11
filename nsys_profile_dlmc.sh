# FILE="dlmc/transformer/l0_regularization/0.5/body_decoder_layer_0_encdec_attention_multihead_attention_q.smtx"
FILE="data/test.smtx"
BLOCK_SIZE=2
D=0.35
A=0.3
N_COL=4
# cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly => https://github.com/apuaaChen/vectorSparse
# nsys profile -t cuda,cusparse,cublas --gpu-metrics-device all --cudabacktrace=all
./bsa_spmm_benchmark -f ${FILE} -i 0 -z 1 -b ${BLOCK_SIZE} -d ${D} -a ${A} -c 1 -n ${N_COL} -x 1 -p 1 -m 0 -v 1 -s 0