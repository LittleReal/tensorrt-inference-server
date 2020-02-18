// Path for c_predict_api
#include <mxnet/c_predict_api.h>

// struct MXNetPredictor {
//     PredictorHandle* hnd;
//     char* json_content;
//     char* param_content;
// }

int set_input(PredictorHandle* hnd, const char* key,const float* data, mx_uint size);

int create_predictor(char* json_path, char* param_path, mx_uint num_input_nodes,
    const char** input_keys, const mx_uint* input_shape_indptr, 
    const mx_uint* input_shape_data, int dev_type, int dev_id,
    PredictorHandle* hnd);

int get_output_shape(PredictorHandle* hnd, mx_uint index, mx_uint** shape_data, mx_uint* shape_dim);

int get_output(PredictorHandle* hnd, uint32_t index, float* data, uint32_t size);


