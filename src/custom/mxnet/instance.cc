#include "src/custom/mxnet/instance.h"
#include "src/core/filesystem.h"
#include "src/core/status.h"

using namespace std;

MXNetInstance::MXNetInstance(
    const string& instance_name, const ModelConfig& model_config,
    int gpu_device)
    :model_config_(model_config), instance_name_(instance_name), gpu_device_(gpu_device)
{
    datatype_ = model_config_.input(0).data_type();
}

int MXNetInstance::Create(
    MXNetInstance** instance, const string& instance_name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
    MXNetInstance* mxnet_instance = MXNetInstance::MXNetInstance(
        instance_name, model_config, data->gpu_device_id);
    *instance = mxnet_instance;

    if (gpu_device == BackendContext::NO_GPU_DEVICE){
        return ErrorCodes::CreationFailure;
    }

    vector<string> server_params = data->server_parameters;
    if (server_params.size() != CUSTOM_SERVER_PARAMETER_CNT){
        return ErrorCodes::CreationFailure;
    }

    string inference_version = server_params[CustomServerParameter::INFERENCE_SERVER_VERSION];
    string model_repository_path = server_params[CustomServerParameter::MODEL_REPOSITORY_PATH];

    int version_int = stoi(inference_version);
    version = uint32(version_int);

    const auto path =
        JoinPath({model_repository_path, model_config.name(), inference_version});
    
    bool* is_dir;
    RETURN_IF_ERROR(IsDirectory(path, is_dir));
    if (! *is_dir){
        return ErrorCodes::CreationFailure;
    }

    if (model_config.parameters_size() < 2){
        return ErrorCodes::CreationFailure;
    }

    auto itr = model_config.parameters().find(MODEL_SYSTEM);
    if (itr == model_config.parameters().end()){
        return ErrorCodes::CreationFailure;
    }

    string model_system = itr->second.string_value();
    string system_path = JoinPath({path, model_system});
    bool* exist;
    RETURN_IF_ERROR(FileExists(system_path, exist));
    if (!*exist){
        return ErrorCodes::CreationFailure;
    }

    itr = model_config.parameters().find(MODEL_PARAM);
    RETURN_IF_ERROR(itr != model_config.parameters().end());
    string model_param = itr->second.string_value();
    string param_path = JoinPath({path, model_param});
    RETURN_IF_ERROR(FileExists(param_path, exist));
    if(!*exist){
        return ErrorCodes::CreationFailure;
    }

    int batch_size = model_config.max_batch_size();
    const auto input_size = model_config.input_size();
    char* input_keys[MAX_INPUT_SIZE];
    mx_uint input_shapes[MAX_INPUT_SIZE][4];
    mx_uint* input_shape_indptr;
    int shape_indptr = 0;
    for (int i =0; i<input_size; i++){
        input_keys[i]=model_config.input(i).name();

        *input_shape_indptr++ = mx_uint(i);
        *input_shape_indptr++ = mx_uint(shape_indptr + model_config.input(i).dims());

        shape_indptr = shape_indptr + model_config.input(i).dims().size() + 1;

        input_shapes[i][0] = batch_size;
        for (int j=0; j < model_config.input(i).dims().size()){
            input_shapes[i][j+1] = model_config.input(i).dims(j);
        }
    }
    
    int success = create_predictor(system_path, param_path, static_cast<mx_uint>(input_size),
        static_cast<const char**>(&input_keys), const mx_uint* input_shape_indptr, 
        static_cast<const mx_uint*>(input_shape_data), int dev_type, int dev_id,
        &pred_hnd_)
}

int get_contiguous_input_tensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, float* input)
{
    uint64_t content_byte_size = expected_byte_size;
    if (!input_fn(input_context, name, static_cast<void**>(&input), &content_byte_size)) {
      return ErrorCodes::GetInputFailed;
    }

    if (content_byte_size!=expected_byte_size){
        return ErrorCodes::GetInputFailed;
    }

    return ErrorCodes::Success;
}

int get_contiguous_output_tensor(
    CustomGetOutputFn_t output_fn, void* output_context, const char* name,
    size_t shape_dim_cnt, int64_t* shape_dims, uint64_t expected_byte_size, float* output)
{
    bool ok = output_fn(output_context, name, shape_dim_cnt, shape_dims, 
        &expected_byte_size, static_cast<void**>(&output));
    if (!ok) {
      return ErrorCodes::GetOutputFailed;
    }
    return ErrorCodes::Success;
}

int MXNetInstance::ExecuteV1(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
    if (payload_cnt == 0) {
        return ErrorCodes::Success;
    }

    uint64_t batch1_element_count;
    uint64_t batch1_byte_size;
    uint64_t batchn_element_count;
    uint64_t batchn_byte_size;
    int err;
    size_t data_type_byte_size = GetDataTypeByteSize(datatype_);

    for (uint32_t pidx = 0; pidx < payload_cnt; pidx++){
        CustomPayload& payload = payloads[pidx];

        for (uint32_t input_idx = 0; input_idx < payload.input_cnt; input_idx++){
            batch1_element_count = GetElementCount(payload.input_shape_dims[input_idx]);
            batch1_byte_size = batch1_element_count * data_type_byte_size;
            batchn_element_count = payload.batch_size * batch1_element_count;
            batchn_byte_size = payload.batch_size * batch1_byte_size;

            float* input;
            const char* input_name = payload.input_names[input_idx];
            err = get_contiguous_input_tensor(
                input_fn, payload.input_context, input_name, batchn_byte_size, &input);
            if (err!=0){
                payload.error_code = err;
                return err;
            }

            err = set_input(&pred_hnd_, input_name, static_cast<const float*>data, batchn_element_count);
            if (err!=0){
                err = ErrorCodes::MXNetSetInputFailed;
                payload.error_code = err;
                return err;
            }
        }

        err = forward(&pred_hnd_);
        if (err != 0){
            err = ErrorCodes::MXNetForwardFailed;
            payload.error_code = err;
            return err;
        }

        for (uint32_t out_idx = 0; out_idx < payload.output_cnt; out_idx++){
            uint32_t* shape_data;
            uint32_t shape_dim;
            err = get_output_shape(&pred_hnd_, out_idx, &shape_data, &shape_dim);
            if (err!=0){
                err = ErrorCodes::MXNetGetOutputFailed;
                payload.error_code = err;
                return err;
            }


            size_t output_ele_cnt=1;
            size_t shape_dim_cnt = size_t(shape_dim);
            vector<int64_t> output_shape;
            for (uint32_t z=0; z < shape_dim; z++){
                int64_t shape_x = int64_t(shape_data[z]);
                out_ele_cnt = out_ele_cnt * size_t(shape_x);
                output_shape.push_back(shape_x);
            }

            const char* out_name = payload.required_output_names[out_index];
            float* output;
            err = get_contiguous_output_tensor(output_fn, payload.output_context, out_name, 
                shape_dim_cnt, &output_shape[0], output_ele_cnt, output);
            if (err!=ErrorCodes::Success){
                payload.error_code = err;
                return err;
            }
            err = get_output(&pred_hnd_, out_idx, output, output_ele_cnt);
            if (err!=0){
                err = ErrorCodes::MXNetGetOutputFailed;
                payload.error_code = err;
                return err;
            }
        }
    }

    return ErrorCodes::Success;
}

int MXNetPredictor::Finalize(){
    int success = MXPredFree(pred_hnd_);
    if (success != 0){
        return ErrorCodes::MXNetFreeFailed
    }
}