
#pragma once

#include <string>
#include <vector>


using namespace std;

class ErrorCodes {
    public:
        static const int Success = 0;
        
        static const int CreationFailure = 1;
        
        static const int InvalidModelConfig = 2;
        
        static const int InvalidInvocationV1 = 3;

        static const int InvalidInvocationV2 = 4;

        static const int ReadModelFileError = 5;

        static const int GetInputFailed = 6;

        static const int GetOutputFailed = 7;

        static const int MXNetCreateFailed = 8;

        static const int MXNetSetInputFailed = 9;

        static const int MXNetForwardFailed = 10;

        static const int MXNetGetOutputFailed = 11;

        static const int MXNetFreeFailed = 12;

        static const int VersionGetFailed = 13;

        /// Error code for an unknown error.
        static const int Unknown = 14;

        ErrorCodes();
        ~ErrorCodes() = default;

        /// Get the string for an error code.
        ///
        /// /param error Error code returned by a CustomInstance function
        /// /return Descriptive error message for a specific error code.
        const char* ErrorString(int error) const;

        /// Register a custom error and error message.
        ///
        /// \param error_message A descriptive error message string
        /// \return The unique error code registered to this error message
        int RegisterError(const string& error_string);

    private:
        /// List of error messages indexed by the error codes
        std::vector<string> err_messages_{Unknown + 1};

        /// Register a specific error. This is use for internal class registration
        /// only.
        ///
        /// \param error The error code
        /// \param error_string The error message
        void RegisterError(int error, const string& error_string);

        /// \param error Error code.
        /// \return True if error code is registered
        inline bool ErrorIsRegistered(int error) const
        {
            return (error >= 0) && (error < static_cast<int>(err_messages_.size()));
        }
};
