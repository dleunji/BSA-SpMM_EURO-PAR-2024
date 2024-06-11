#include <string>
#include <unistd.h>
#include <iostream>

#ifndef OPTION_H
#define OPTION_H

class Option
{
public:
    std::string input_filename = "data/test.smtx";
    std::string output_filename = "";

    int input_format = 0;
    bool pattern_only = true;
    bool zero_padding = true;
    int n_cols = 128;
    int method = 0;

    int repetitions = 10;
    int spmm = 0;

    int block_size = 1;
    float delta = 0;
    float alpha = 0;
    bool compress_rows = true;
    bool valid = false;

    Option(int argc, char *argv[])
    {
        parse(argc, argv);
    }

    void parse(int argc, char *argv[])
    {
        char param_opt;
        while ((param_opt = getopt(argc, argv, "d:t:c:n:x:p:f:o:i:b:z:a:v:m:s:")) != -1)
        {
            switch (param_opt)
            {
            case 'a':
                alpha = std::stof(optarg);
                break;
            case 'd':
                delta = std::stof(optarg);
                break;
            case 'c':
                compress_rows = (std::stoi(optarg) == 1);
                break;
            case 'n':
                n_cols = std::stoi(optarg);
                break;
            case 'x':
                repetitions = std::stoi(optarg);
                break;
            case 'p':
                pattern_only = (std::stoi(optarg) == 1);
                break;
            case 'f':
                input_filename = std::string(optarg);
                break;
            case 'o':
                output_filename = std::string(optarg);
                break;
            case 'i':
                input_format = std::stoi(optarg);
                break;
            case 'b':
                block_size = std::stoi(optarg);
                break;
            case 'v':
                valid = (std::stoi(optarg) == 1);
                break;
            case 'z':
                zero_padding = (std::stoi(optarg) == 1);
                break;
            case 'm':
                method = std::stoi(optarg);
                break;
            case 's':
                spmm = std::stoi(optarg);
                break;
            }
        }
    }
};

#endif