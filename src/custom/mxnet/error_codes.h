
#pragma once

#include <string>
#include <vector>


class ErrorCodes {
    public:
        /// Error code for success
        static const int Success = 0;

        /// Error code for creation failure.
        static const int CreationFailure = 1;

        /// Error code when instance failed to load the model configuration.
        static const int InvalidModelConfig = 2;

        /// Error code when V1 version of a function is called
        /// while the custom backend is not V1.
        static const int InvalidInvocationV1 = 3;

        /// Error code when V2 version of a function is called
        /// while the custom backend is not V2.
        static const int InvalidInvocationV2 = 4;

        /// Error code for an unknown error.
        static const int Unknown = 5;

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
