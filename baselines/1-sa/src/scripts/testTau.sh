export MATRICES_PATH=$1
export RESULTS_PATH=$2
export PROGRAM=$3


BLOCK_SIZEs=(64 128 256 512 1024)
B_COLs=(1024 2048 4096 8192)
EXPERIMENTs=("BCSR_no_reord" "BCSR_reord" "BELLPACK_no_block" "CSR" "GEMM")
taufile=" results/collected_experiments/suitsparse_all/tau.csv"

declare -A experiments
experiments["BCSR_no_reord"]="-F 1 -a 5 -M 6"
experiments["BCSR_reord"]="-F 1 -a 2 -M 6"
experiments["BELLPACK_no_block"]="-F 1 -a 2 -M 3"
experiments["CSR"]="-M 2"
experiments["GEMM"]="-M 1"

USE_PATTERN=1
USE_GROUP=0
REORDERING=0
SIM=1 #0: hamming 1:jaccard; +2 for OPENMP versions
BASIC_ARGS="-P 1 -v 1 -r ${REORDERING} -m ${SIM} -p ${USE_PATTERN} -g ${USE_GROUP} -R 1"

function create_launch {

script_body="#!/bin/bash -l
#SBATCH --job-name="${EXP_NAME}"
#SBATCH --time=00:10:00
#SBATCH --output="${RESULTS_PATH}_scripts/outputs/${EXP_NAME}".%j.o
#SBATCH --error="${RESULTS_PATH}_scripts/errors/${EXP_NAME}".%j.e
#SBATCH --account=flavio.vella
#SBATCH --ntasks=1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module load cuda/12.1
./${PROGRAM} ${BASIC_ARGS} ${EXP_ARGS} ${ARGS}

sleep 1s
"
script_name=___tmp_script_${EXP_NAME}
script_folder=${RESULTS_PATH}_scripts

if [[ -f "${script_folder}/${script_name}" ]]; then
	echo "experiment exists already"
else
	echo "${script_body}" > ${script_folder}/${script_name}.sbatch
# 	sbatch ${script_folder}/${script_name}.sbatch
fi
}

mkdir ${RESULTS_PATH}

total=$((${#EXPERIMENTs[@]}*${#B_COLs[@]}*${#BLOCK_SIZEs[@]}))

matrices_processed=0
for fullpath in ${MATRICES_PATH}/*.*; do

	progress=0 #progress bar
	((matrices_processed++))

	MATRIX_FILE=$(basename -- "${fullpath}")
	MATRIX_NAME="${MATRIX_FILE%.*}"
	echo "============= processing matrix ${MATRIX_NAME} ($matrices_processed matrices processed)"
	MATRIX_FOLDER=${RESULTS_PATH}/${MATRIX_NAME}
	mkdir ${MATRIX_FOLDER} 2>/dev/null
	for b_cols in ${B_COLs[@]};do
		for block in ${BLOCK_SIZEs[@]}; do
			for exp in ${EXPERIMENTs[@]}; do

				#progress bar stuff
    			percent=$((progress * 100 / total))
	    		echo -ne "Experiments processed: [$percent%]\r"
				((progress++))
				#===================

				B=$block
				b=$block
				export EXP_NAME="blocking_G_${MATRIX_NAME}_b_${b}_B_${B}_a_${a}_e_${exp}"
				export OUTFILE=${MATRIX_FOLDER}/${EXP_NAME}.txt
				if [[ -f "${OUTFILE}" ]]; then
					echo "FILE ${OUTFILE} ALREADY EXISTS. SKIPPING"
				else
					if [ "${exp}" == "BCSR_reord" ]; then
                        t2=$(grep ${MATRIX_NAME} ${taufile} | grep "${B},${b}" | cut -d',' -f4)
					else
						export t=0
					fi
					if [ $t != -1 ];then #this is a special flag for BCSR_reord experiments that are not to be run
						export ARGS="-f ${fullpath} -b ${b} -B ${B} -t ${t} -o ${OUTFILE} -n ${EXP_NAME}"
						export BASIC_ARGS
						export EXP_ARGS=${experiments[$exp]}
# 						create_launch
					fi
				fi
			done
		done
	done
done
