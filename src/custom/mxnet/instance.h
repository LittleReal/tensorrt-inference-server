#pragma once

#include "src/backends/custom/custom.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/backend_context.h"
#include "src/custom/mxnet/error_codes.h"
#include "src/custom/mxnet/mxnet.h"

// Path for c_predict_api
#include <mxnet/c_predict_api.h>

#define MODEL_SYSTEM = "model_system";
#define MODEL_PARAM = "model_param";

#define MAX_INPUT_SIZE 5;


using namespace std;

class MXNetInstance{
    public:
        static uint32_t Version = 0;

        MXNetInstance(
        const string& instance_name, const ModelConfig& model_config,
        int gpu_device);

        ~MXNetInstance() = default;

        int Create(
        MXNetInstance** instance, const string& name,
        const ModelConfig& model_config, int gpu_device,
        const CustomInitializeData* data);

        int ExecuteV1(
            const uint32_t payload_cnt, CustomPayload* payloads,
            CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
        {
            return ErrorCodes::InvalidInvocationV1;
        }

        int ExecuteV2(
            const uint32_t payload_cnt, CustomPayload* payloads,
            CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn)
        {
            return ErrorCodes::InvalidInvocationV2;
        }

        const char* ErrorString(uint32_t error) const
        {
            return errors_.ErrorString(error);
        }
    
        int Finalize(){};

    private:

        PredictorHandle pred_hnd_;

        string instance_name_;
        
        ModelConfig model_config_;

        int gpu_device_;

        DataType data_type_;
        
        ErrorCodes errors_{};
}