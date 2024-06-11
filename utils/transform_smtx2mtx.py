import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="the directory for input")
parser.add_argument("-o", help="the directory for output of transformation")

args = parser.parse_args()


def transform_matrix(matrix, out_matrix):
    f = open(matrix, mode="r")
    output_f = open(out_matrix, mode="w")
    meta = f.readline()
    print(meta)
    meta_arr = meta.split(",")
    nrows = int(meta_arr[0])
    ncols = int(meta_arr[1])
    nzs = int(meta_arr[2])

    rowptr = f.readline().split(" ")
    colidx = f.readline().split(" ")
    output_f.write(f"{nrows} {ncols} {nzs}\n")
    for r in range(nrows):
        start_pos = int(rowptr[r])
        end_pos = int(rowptr[r + 1])
        for nz in range(start_pos, end_pos):
            c = int(colidx[nz])
            output_f.write(f"{r + 1} {c + 1} 1\n")

    f.close()
    output_f.close()

if __name__ == "__main__":
    input_dir = args.i
    output_dir = args.o
    models = os.listdir(input_dir)
    for model in models:
        cur_dir1 = input_dir + "/" + model
        if not os.path.isdir(cur_dir1):
            continue
        out_dir1 = output_dir + "/" + model
        prunings = os.listdir(cur_dir1)
        for pruning in prunings:
            cur_dir2 = cur_dir1 + "/" + pruning
            out_dir2 = out_dir1 + "/" + pruning
            sparsities = os.listdir(cur_dir2)
            for sparsity in sparsities:
                cur_dir3 = cur_dir2 + "/" + sparsity
                out_dir3 = out_dir2 + "/" + sparsity
                os.makedirs(out_dir3, exist_ok=True)
                for mtx in os.listdir(cur_dir3):
                    print(mtx)
                    transform_matrix(cur_dir3 + "/" + mtx, out_dir3 + "/" + mtx)
