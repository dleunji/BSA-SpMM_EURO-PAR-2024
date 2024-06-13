# Accelerated Block-Sparsity-Aware Matrix Reordering for Leveraging Tensor Cores in Sparse Matrix-Multivector Multiplication
## Prerequisites
- g++ $\ge$ 11
- cmake $\ge$ 3.14
- git
- python $\ge$ 3.9
- CUDA $=$ 12.1
- NVIDIA GPU with sm $\ge$ 80

## Step 1. Setup and Download
### Setup the environmental variable
Change some variables, `CUDA_PATH` and `CUDA_ARCH`, in the env.sh file according to your computer.
`CUDA_PATH` denotes the path where nvcc is installed.
And change `CUDA_ARCH` following the [specification](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
Other environmental variables will be setup automatically.

```bash
export CUDA_PATH=/usr/local/cuda-12.1
export CUDA_ARCH=86
```

And then, execute the env.sh file with `source` command to export the environmental variables and install python packages.

```bash
source env.sh
```

### Install one of the baselines, Sputnik
```bash
bash install_sputnik.sh
```

### Download the dataset
```bash
bash download_data.sh
```

### Install required package (for Debian)
The Debian user should install the bc package as shown below because the bc package is not pre-installed in the Debian system.

```bash
sudo apt-get install bc
```


## Step 2. Compile and run the experiments
After running the shell script, The each figure file is generated and located in `plots` directory.

### Compile the source codes
```bash
bash build.sh
```

### To reproduce the figure 4
Benchmarking all algorithms in Figure 4 on the large DLMC dataset takes more than 5 hours.
The paper includes ASpT-RR as a benchmark baseline in figure 4, but as it is not currently open-source, we are unable to provide it.
Therefore, we ask for your understanding that it is not included in the released artifact.

```bash
bash run_fig4_dlmc_sh       
```

If you want to shorten the execution time and conduct a brief experiment, just run `run_fig4_dlmc_short.sh`.

This script conducts the experiment on just 2 matrices for each sparsity in a subfigure.


```bash   
bash run_fig4_dlmc_short.sh # Brief version
```

### To reproduce the figure 5
It will take about 30 minutes to run and plot the figure.

```bash
bash run_fig5_dlmc_sh
```
Similar to Figure 4, there is a brief version of Figure 5 that requires about 5 minutes to execute.

```bash
bash run_fig5_dlmc_short.sh # Brief version
```


### To reproduce the figure 6
It will take about 30 minutes to run and plot the figure.
```bash
bash run_fig6_dlmc_sh
```