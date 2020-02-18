#include "src/custom/mxnet/instance.h"

extern "C" {

uint32_t CustomVersion()
{
    return MXNetInstance::Version;
}

int CustomInitialize(const CustomInitializeData* data, void** instance)
{
    // Convert the serialized model config to a ModelConfig object.
    ModelConfig model_config;
    if (!model_config.ParseFromString(std::string(
            data->serialized_model_config, data->serialized_model_config_size))) {
        return ErrorCodes::InvalidModelConfig;
    }

    // Create the instance and validate that the model configuration is
    // something that we can handle.
    int err = MXNetInstance::Create(
        (MXNetInstance**)instance, std::string(data->instance_name),
        model_config, data->gpu_device_id, data);

    if (ErrorCodes::Success != err) {
        return err;
    }

    if (instance == nullptr) {
        return ErrorCodes::CreationFailure;
    }

    return ErrorCodes::Success;
}

int CustomFinalize(void* instance)
{
    int err = ErrorCodes::Success;
    if (instance != nullptr) {
        MXNetInstance* instance = static_cast<MXNetInstance*>(instance);
        err = instance->Finalize();
        delete instance;
    }

    return err;
}

const char* CustomErrorString(void* instance, int errcode)
{
    MXNetInstance* instance = static_cast<MXNetInstance*>(instance);

    return instance->ErrorString(errcode);
}

int CustomExecute(
    void* instance, const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
    if (instance == nullptr) {
        return ErrorCodes::Unknown;
    }

    MXNetInstance* instance = static_cast<MXNetInstance*>(instance);
    return instance->ExecuteV1(payload_cnt, payloads, input_fn, output_fn);
}

int CustomExecuteV2(
    void* instance, const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn)
{
    if (instance == nullptr) {
        return ErrorCodes::Unknown;
    }

    MXNetInstance* instance = static_cast<MXNetInstance*>(instance);
    return instance->ExecuteV2(payload_cnt, payloads, input_fn, output_fn);
}

}  // extern "C"