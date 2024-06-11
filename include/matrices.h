#include "definitions.h"
#include "option.h"
#include <string.h>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <random>
#include "logger.h"

using namespace std;

#ifndef MATRICES_H
#define MATRICES_H
class CSR
{
public:
    intT rows;
    intT cols;
    // for zero paading
    intT original_rows;
    intT original_cols;
    intT total_nonzeros;
    bool pattern_only;

    intT *rowptr;
    // intT *nzcount;
    intT *colidx;
    DataT *values;

    void read_from_mtx(std::ifstream &fin, Option option, bool zero_base);
    void read_from_smtx(std::ifstream &fin, Option option, bool zero_base);

    CSR(Option &option)
    {
        FileFormatType format = static_cast<FileFormatType>(option.input_format);
        std::ifstream fin;
        pattern_only = option.pattern_only;
        fin.open(option.input_filename);
        if (format == mtx)
        {
            read_from_mtx(fin, option, true);
        }
        else if (format == smtx)
        {
            read_from_smtx(fin, option, true);
        }
    }

    ~CSR()
    {
        delete[] rowptr;
        // printf("delete rowptr\n");
        // delete nzcount;
        // printf("delete nzcount\n");
        delete[] colidx;
        // printf("delete colidx\n");
        if (not pattern_only)
            delete[] values;
    }
};

class BSA_HYBRID
{
public:
    intT rows;
    intT cols;
    // for zero paading
    intT original_rows;
    intT original_cols;

    bool pattern_only;

    // The result of reordering
    vector<intT> row_permutation;

    // BELL PACK
    intT bellpack_total_nonzeros = 0;
    intT block_size;
    intT *ellColInd;
    DataT *ellValues;
    DataT_H *h_ellValues;

    intT ell_rows;
    intT ell_cols;

    // CSR
    intT csr_total_nonzeros = 0;
    intT *csr_rowptr;
    intT *csr_colidx;
    DataT *csr_values;
    DataT_H *csv_h_values;

    BSA_HYBRID(CSR &csr, LOGGER &logger, intT block_size_, float delta, vector<intT> row_permutation_)
    {
        // printf("delta: %f\n", delta);
        rows = csr.rows;
        cols = csr.cols;
        block_size = block_size_;

        original_rows = csr.original_rows;
        original_cols = csr.original_cols;
        pattern_only = csr.pattern_only;
        row_permutation = row_permutation_;

        assert(rows % block_size == 0);
        assert(cols % block_size == 0);

        ell_rows = rows / block_size;
        ell_cols = 0;
        vector<vector<intT>> nnz(ell_rows, vector<intT>(cols / block_size, 0));
        vector<vector<intT>> dense_tiles(ell_rows);
        intT offset = 0;

        logger.rows = original_rows;
        logger.cols = original_cols;
        logger.block_size = block_size;
        // logger.method = method;

// for (int i = 0; i <= rows; i++)
// {
//     printf("%d ", csr.rowptr[i]);
// }
// printf("\n");
// for (int i = 0; i < csr.total_nonzeros; i++)
// {
//     printf("%d ", csr.colidx[i]);
// }
// printf("\n");
#pragma omp parallel for num_threads(12)
        for (intT i = 0; i < rows; i++)
        {
            intT r = row_permutation[i];
            intT start_pos = csr.rowptr[r];
            intT end_pos = csr.rowptr[r + 1];
            // printf("%d th row: %d ~ %d \n", i, start_pos, end_pos);
            for (intT nz = start_pos; nz < end_pos; nz++)
            {
                intT col = csr.colidx[nz];
                // printf("col: %d\n", col);
                // printf("%d %d\n", i / block_size, col / block_size);
                nnz[i / block_size][col / block_size]++;
            }
        }

        // for (int i = 0; i < nnz.size(); i++)
        // {
        //     for (int j = 0; j < nnz[i].size(); j++)
        //     {
        //         printf("%d ", nnz[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("here1\n");
        // vector<intT> csr_rowptr(rows + 1);
        vector<vector<intT>> vec_csr_colidx(rows);
        vector<vector<DataT>> vec_csr_values(rows);

        for (intT row_panel_id = 0; row_panel_id < nnz.size(); row_panel_id++)
        {
            for (intT j = 0; j < nnz[row_panel_id].size(); j++)
            {
                // printf("nnz [%d][%d]: %d\n", row_panel_id, j, nnz[row_panel_id][j]);
                if (nnz[row_panel_id][j] > block_size * block_size * delta)
                {
                    // BELL PACK
                    // printf("%d th panel | %d th block inserted.\n", row_panel_id, j);
                    dense_tiles[row_panel_id].push_back(j);
                    logger.num_tiles++;
                    // printf("density: %.3f\n", (float)nnz[row_panel_id][j] / (block_size * block_size));
                    logger.avg_density_of_tiles += (float)nnz[row_panel_id][j] / (block_size * block_size);
                }
                else
                {
                    // intT max_row = (row_panel_id + 1) * block_size - 1 < original_rows ? (row_panel_id + 1) * block_size - 1 : original_rows;
                    if (nnz[row_panel_id][j] == 0)
                        continue;

                    // printf("row panel id: %d j: %d\n", row_panel_id, j);
                    // CSR
                    for (intT row = row_panel_id * block_size; row < (row_panel_id + 1) * block_size; row++)
                    {
                        // printf("row: %d\n", row);
                        intT row_id = row_permutation[row];
                        // printf("[][]row: %d => row_id: %d\n", row, row_id);
                        if (row_id >= original_rows)
                            continue;
                        intT start_pos = csr.rowptr[row_id];
                        intT end_pos = csr.rowptr[row_id + 1];
                        for (intT nz = start_pos; nz < end_pos; nz++)
                        {
                            // printf("row: %d => row_id: %d col: %d ell_col: %d\n", row, row_id, csr.colidx[nz], j);
                            if (csr.colidx[nz] / block_size != j)
                                continue;
                            // printf("push row: %d col: %d\n", row, csr.colidx[nz]);

                            vec_csr_colidx[row].push_back(csr.colidx[nz]);
                            csr_total_nonzeros++;
                            if (pattern_only)
                                vec_csr_values[row].push_back(1);
                            else
                                vec_csr_values[row].push_back(csr.values[nz]);
                        }
                    }
                }
            }
            ell_cols = max(ell_cols, (intT)dense_tiles[row_panel_id].size());
        }
        // printf("here2\n");
        ellColInd = new intT[ell_rows * ell_cols];
        ellValues = new DataT[ell_rows * ell_cols * block_size * block_size];
        h_ellValues = new DataT_H[ell_rows * ell_cols * block_size * block_size];

        memset(ellColInd, -1, ell_rows * ell_cols * sizeof(intT));
        memset(ellValues, 0, ell_rows * ell_cols * block_size * block_size * sizeof(DataT));
        memset(h_ellValues, 0, ell_rows * ell_cols * block_size * block_size * sizeof(DataT_H));
        // printf("ellcolind\n");
        for (int i = 0; i < ell_rows; i++)
        {
            for (int j = 0; j < dense_tiles[i].size(); j++)
            {
                ellColInd[i * ell_cols + j] = dense_tiles[i][j];
            }
        }
        // printf("ellcolind\n");
        // // printf("here3\n");
        // for (int i = 0; i < dense_tiles.size(); i++)
        // {
        //     for (int j = 0; j < dense_tiles[i].size(); j++)
        //     {
        //         printf("%d ", dense_tiles[i][j]);
        //     }
        //     printf("\n");
        // }
        for (int i = 0; i < rows; i++)
        {
            intT r = row_permutation[i];
            // printf("%d => %d\n", i, r);
            intT start_pos = csr.rowptr[r];
            intT end_pos = csr.rowptr[r + 1];
            for (int nz = start_pos; nz < end_pos; nz++)
            {
                bellpack_total_nonzeros++;
                intT row_panel_id = i / block_size;
                intT original_col = csr.colidx[nz];
                // printf("col: %d\n", original_col);
                // printf("size of dense tiles: %d\n", dense_tiles.size());
                auto it_ellblock_col = find(dense_tiles[row_panel_id].begin(), dense_tiles[row_panel_id].end(), (original_col / block_size));
                // assert(it_ellblock_col != dense_tiles[i].end());
                if (it_ellblock_col == dense_tiles[row_panel_id].end())
                    continue;
                // printf("here4\n");
                intT ellblock_col = it_ellblock_col - dense_tiles[row_panel_id].begin();
                intT inner_col = original_col % block_size;
                // intT inner_row = i % block_size;
                intT idx = (i * ell_cols * block_size) + (ellblock_col * block_size) + inner_col;
                // intT idx = row_panel_id * ell_cols * block_size * block_size + (ellblock_col * block_size * block_size) + (inner_row * block_size) + inner_col;
                // printf("nz:%d row:%d col: %d idx: %d inner_row: %d inner_col:%d\n", nz, row_panel_id, ellblock_col, idx, inner_row, inner_col);
                if (not pattern_only)
                {
                    ellValues[idx] = csr.values[offset + nz];
                    h_ellValues[idx] = __float2half(csr.values[offset + nz]);
                }
                else
                {
                    ellValues[idx] = 1;
                    h_ellValues[idx] = __float2half(1);
                }
            }
        }
        // printf("total bellpack nonzeros: %d\n", bellpack_total_nonzeros);
        // printf("here5\n");
        csr_rowptr = new intT[rows + 1];
        csr_colidx = new intT[csr_total_nonzeros];
        csr_values = new DataT[csr_total_nonzeros];
        csr_rowptr[0] = 0;
        offset = 0;
        for (intT r = 0; r < rows; r++)
        {
            // intT r = row_permutation[i];
            // printf("r: %d nnz: %d\n", r, vec_csr_colidx[r].size());
            std::copy(vec_csr_colidx[r].begin(), vec_csr_colidx[r].end(), csr_colidx + offset);
            std::copy(vec_csr_values[r].begin(), vec_csr_values[r].end(), csr_values + offset);
            csr_rowptr[r + 1] = csr_rowptr[r] + (intT)vec_csr_colidx[r].size();
            offset += vec_csr_colidx[r].size();
        }

        logger.nnz_in_csr = csr_total_nonzeros;
        logger.nnz_in_bellpack = bellpack_total_nonzeros;
        if (logger.num_tiles)
            logger.avg_density_of_tiles /= logger.num_tiles;
    }

    ~BSA_HYBRID()
    {
        // printf("start deleting bsa hybrid\n");
        delete[] ellColInd;
        // printf("delete colind\n");
        delete[] ellValues;
        delete[] h_ellValues;
        delete[] csr_rowptr;
        delete[] csr_colidx;
        delete[] csr_values;
        // printf("delete values\n");
    }
};

class ARR
{
public:
    intT original_rows;
    intT rows;
    intT cols;
    DataT *mat;
    DataT_H *h_mat;
    bool with_half;

    ARR(intT original_rows_, intT rows_, intT cols_, bool with_half_)
    {
        // printf("here1\n");
        original_rows = original_rows_;
        rows = rows_;
        cols = cols_;
        // printf("rows: %d cols: %d\n", rows, cols);
        with_half = with_half_;

        mat = new DataT[rows * cols];
        if (with_half)
            h_mat = new DataT_H[rows * cols];

        memset(mat, 0, rows * cols * sizeof(DataT));
        if (with_half)
            memset(h_mat, 0, rows * cols * sizeof(DataT_H));
    }

    void fill_random(bool zero_padding)
    {
        random_device rd;
        mt19937 e2(rd());
        uniform_real_distribution<> dist(0, 1);
        if (zero_padding)
        {

            for (int n = 0; n < rows * cols; n++)
            {
                int r = n / cols;
                if (r < original_rows)
                {
                    // mat[n] = dist(e2);
                    mat[n] = 1;
                    if (with_half)
                        h_mat[n] = __float2half(mat[n]);
                }
                else
                {
                    mat[n] = 0.0;
                    if (with_half)
                        h_mat[n] = __float2half(0.0);
                }
            }
        }
        else
        {
            for (int n = 0; n < rows * cols; n++)
            {
                // mat[n] = dist(e2);
                mat[n] = 1;
                if (with_half)
                    h_mat[n] = __float2half(mat[n]);
            }
        }
    }

    ~ARR()
    {
        delete[] mat;
        if (with_half)
            delete[] h_mat;
    }
};

#endif