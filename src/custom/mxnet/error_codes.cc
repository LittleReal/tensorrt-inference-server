#include "src/custom/mxnet/error_codes.h"


using namespace std;

ErrorCodes::ErrorCodes()
{
    RegisterError(Success, "success");
    RegisterError(CreationFailure, "failed to create instance");
    RegisterError(InvalidModelConfig, "invalid model configuration");
    RegisterError(
        InvalidInvocationV1,
        "invalid V1 function invocation while the custom backend is not V1");
    RegisterError(
        InvalidInvocationV2,
        "invalid V2 function invocation while the custom backend is not V2");
    RegisterError(Unknown, "unknown error");
}

const char* ErrorCodes::ErrorString(int error) const
{
    if (ErrorIsRegistered(error)) {
        return err_messages_[error].c_str();
    }

    return err_messages_[Unknown].c_str();
}

int ErrorCodes::RegisterError(const string& error_string)
{
    err_messages_.push_back(error_string);
    return static_cast<int>(err_messages_.size() - 1);
}

void ErrorCodes::RegisterError(int error, const string& error_string)
{
    if (ErrorIsRegistered(error)) {
        err_messages_[error] = error_string;
    }
}
