#include <sys/stat.h>

#include "src/custom/mxnet/mxnet.h"


int file_size(char* filename)
{
    struct stat statbuf;
    stat(filename, &statbuf);
    int size = statbuf.st_size;
    return size;
}


int readfile(char* path, const int length, char* content){
    FILE *fp;
    fp = fopen(path, "rt");
    if(fp == NULL){
        return -1
    }
    
    size_t cnt = fread(content, sizeof(char), length, fp);
    if (int(cnt) != length){
        printf("There is err when reading model file.\n");
        exit(1);
    }

    fclose(fp);
    return 0;
}

int create_predictor(char* json_path, char* param_path, mx_uint num_input_nodes,
    const char** input_keys, const mx_uint* input_shape_indptr, 
    const mx_uint* input_shape_data, int dev_type, int dev_id,
    PredictorHandle* hnd)
{
    int json_length = file_size(json_path);
    char* json_content = (char *)malloc(json_length*sizeof(char));
    int err = readfile(json_path, json_length, json_content);
    if (err!=0){
        return err;
    }

    int param_length = file_size(param_path);
    char* param_content = (char *)malloc(param_length*sizeof(char));
    err = readfile(param_path, param_length, param_content);
    if (err != 0){
        return err;
    }

    int success = MXPredCreate(static_cast<const char*>(json_content),
               static_cast<const char*>(param_content),
               param_length,
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               hnd);
    return success;
}

int set_input(PredictorHandle* hnd, const char* key, const float* data, mx_uint size){
    int success = MXPredSetInput(*hnd, key, data, size);
    return success;
}

int get_output_shape(PredictorHandle* hnd, mx_uint index, mx_uint** shape_data, mx_uint* shape_dim){
    int success = MXPredGetOutputShape(*hnd, index, shape_data, shape_dim);
    return success;
}

int get_output(PredictorHandle* hnd, uint32_t index, float* data, uint32_t size){
    int success = MXPredGetOutput(hnd, index, data, size);
    return success;
}

int forward(PredictorHandle* hnd){
    int success = MXPredForward(*hnd);
    return success;
}
