#pragma once

#include "src/backends/custom/custom.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"

// Path for c_predict_api
#include <mxnet/c_predict_api.h>


class MxnetInstance{
    public:

        static int Create(
        MxnetInstance** instance, const string& name,
        const ModelConfig& model_config, int gpu_device,
        const CustomInitializeData* data);

        MxnetInstance(
        const string& instance_name, const ModelConfig& model_config,
        int gpu_device);

        virtual ~MxnetInstance() = default;

        virtual int ExecuteV1(
            const uint32_t payload_cnt, CustomPayload* payloads,
            CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
        {
            return ErrorCodes::InvalidInvocationV1;
        }

        virtual int ExecuteV2(
            const uint32_t payload_cnt, CustomPayload* payloads,
            CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn)
        {
            return ErrorCodes::InvalidInvocationV2;
        }
    

    private:

        PredictorHandle pred_hnd_;

        const string instance_name_;
        
        const ModelConfig model_config_;

        const int gpu_device_;
        
        ErrorCodes errors_{};
}