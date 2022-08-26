#include "precomp.hpp"
#include <opencv2/core/types_c.h>

#if CV_MAJOR_VERSION < 5
#if !defined(GAPI_STANDALONE)
#  include "opencv2/core/cvdef.h"
#  include "opencv2/core/utils/logger.hpp"
#  define GAPI_LOG_INFO(tag, ...)    CV_LOG_INFO(tag, __VA_ARGS__)
#  define GAPI_LOG_WARNING(tag, ...) CV_LOG_WARNING(tag, __VA_ARGS__)
#  define GAPI_LOG_DEBUG(tag, ...)    CV_LOG_DEBUG(tag, __VA_ARGS__)
#  define GAPI_LOG_FATAL(tag, ...)   CV_LOG_FATAL(tag, __VA_ARGS__)
#else
#  define GAPI_LOG_INFO(tag, ...)
#  define GAPI_LOG_WARNING(tag, ...)
#  define GAPI_LOG_DEBUG(tag, ...)
#  define GAPI_LOG_FATAL(tag, ...)
#endif //  !defined(GAPI_STANDALONE)
#endif

namespace cv {
#if CV_MAJOR_VERSION < 5

    enum class MatrixType
    {
        AUTO = 0,
        DENSE = 1,
        SPARSE = 2
    };

    /** @brief Type of variables used in LevMarq solver

    Variables can be linear, rotation (SO(3) group) or rigid transformation (SE(3) group) with corresponding jacobians and exponential updates.

    Note: only linear variables are now supported
    */
    enum class VariableType
    {
        LINEAR = 0,
        SO3 = 1,
        SE3 = 2
    };
    class CV_EXPORTS LevMarq
    {
    public:
        /** @brief Optimization report

        The structure is returned when optimization is over.
        */
        struct CV_EXPORTS Report
        {
            Report(bool isFound, int nIters, double finalEnergy) :
                    found(isFound), iters(nIters), energy(finalEnergy)
            { }
            // true if the cost function converged to a local minimum which is checked by check* fields, thresholds and other options
            // false if the cost function failed to converge because of error, amount of iterations exhausted or lambda explosion
            bool found;
            // amount of iterations elapsed until the optimization stopped
            int iters;
            // energy value reached by the optimization
            double energy;
        };

        /** @brief Structure to keep LevMarq settings

        The structure allows a user to pass algorithm parameters along with their names like this:
        @code
        MySolver solver(nVars, callback, MySolver::Settings().geodesicS(true).geoScale(1.0));
        @endcode
        */
        struct CV_EXPORTS Settings
        {
            Settings();

            inline Settings& setJacobiScaling          (bool   v) { jacobiScaling = v; return *this; }
            inline Settings& setUpDouble               (bool   v) { upDouble = v; return *this; }
            inline Settings& setUseStepQuality         (bool   v) { useStepQuality = v; return *this; }
            inline Settings& setClampDiagonal          (bool   v) { clampDiagonal = v; return *this; }
            inline Settings& setStepNormInf            (bool   v) { stepNormInf = v; return *this; }
            inline Settings& setCheckRelEnergyChange   (bool   v) { checkRelEnergyChange = v; return *this; }
            inline Settings& setCheckMinGradient       (bool   v) { checkMinGradient = v; return *this; }
            inline Settings& setCheckStepNorm          (bool   v) { checkStepNorm = v; return *this; }
            inline Settings& setGeodesic               (bool   v) { geodesic = v; return *this; }
            inline Settings& setHGeo                   (double v) { hGeo = v; return *this; }
            inline Settings& setGeoScale               (double v) { geoScale = v; return *this; }
            inline Settings& setStepNormTolerance      (double v) { stepNormTolerance = v; return *this; }
            inline Settings& setRelEnergyDeltaTolerance(double v) { relEnergyDeltaTolerance = v; return *this; }
            inline Settings& setMinGradientTolerance   (double v) { minGradientTolerance = v; return *this; }
            inline Settings& setSmallEnergyTolerance   (double v) { smallEnergyTolerance = v; return *this; }
            inline Settings& setMaxIterations          (int    v) { maxIterations = (unsigned int)v; return *this; }
            inline Settings& setInitialLambda          (double v) { initialLambda = v; return *this; }
            inline Settings& setInitialLmUpFactor      (double v) { initialLmUpFactor = v; return *this; }
            inline Settings& setInitialLmDownFactor    (double v) { initialLmDownFactor = v; return *this; }

            // normalize jacobian columns for better conditioning
            // slows down sparse solver, but maybe this'd be useful for some other solver
            bool jacobiScaling;
            // double upFactor until the probe is successful
            bool upDouble;
            // use stepQuality metrics for steps down
            bool useStepQuality;
            // clamp diagonal values added to J^T*J to pre-defined range of values
            bool clampDiagonal;
            // to use squared L2 norm or Inf norm for step size estimation
            bool stepNormInf;
            // to use relEnergyDeltaTolerance or not
            bool checkRelEnergyChange;
            // to use minGradientTolerance or not
            bool checkMinGradient;
            // to use stepNormTolerance or not
            bool checkStepNorm;
            // to use geodesic acceleration or not
            bool geodesic;
            // second directional derivative approximation step for geodesic acceleration
            double hGeo;
            // how much of geodesic acceleration is used
            double geoScale;
            // optimization stops when norm2(dx) drops below this value
            double stepNormTolerance;
            // optimization stops when relative energy change drops below this value
            double relEnergyDeltaTolerance;
            // optimization stops when max gradient value (J^T*b vector) drops below this value
            double minGradientTolerance;
            // optimization stops when energy drops below this value
            double smallEnergyTolerance;
            // optimization stops after a number of iterations performed
            unsigned int maxIterations;

            // LevMarq up and down params
            double initialLambda;
            double initialLmUpFactor;
            double initialLmDownFactor;
        };

        /** "Long" callback: f(param, &err, &J) -> bool
        Computes error and Jacobian for the specified vector of parameters,
        returns true on success.

        param: the current vector of parameters
        err: output vector of errors: err_i = actual_f_i - ideal_f_i
        J: output Jacobian: J_ij = d(err_i)/d(param_j)

        Param vector values may be changed by the callback only if they are fixed.
        Changing non-fixed variables may lead to incorrect results.
        When J=noArray(), it means that it does not need to be computed.
        Dimensionality of error vector and param vector can be different.
        The callback should explicitly allocate (with "create" method) each output array
        (unless it's noArray()).
        */
        typedef std::function<bool(InputOutputArray, OutputArray, OutputArray)> LongCallback;

        /** Normal callback: f(param, &JtErr, &JtJ, &errnorm) -> bool

            Computes squared L2 error norm, normal equation matrix J^T*J and J^T*err vector
            where J is MxN Jacobian: J_ij = d(err_i)/d(param_j)
            err is Mx1 vector of errors: err_i = actual_f_i - ideal_f_i
            M is a number of error terms, N is a number of variables to optimize.
            Make sense to use this class instead of usual Callback if the number
            of error terms greatly exceeds the number of variables.

            param: the current Nx1 vector of parameters
            JtErr: output Nx1 vector J^T*err
            JtJ: output NxN matrix J^T*J
            errnorm: output total error: dot(err, err)

            Param vector values may be changed by the callback only if they are fixed.
            Changing non-fixed variables may lead to incorrect results.
            If JtErr or JtJ are empty, they don't have to be computed.
            The callback should explicitly allocate (with "create" method) each output array
            (unless it's noArray()).
        */
        typedef std::function<bool(InputOutputArray, OutputArray, OutputArray, double&)> NormalCallback;

        /**
            Creates a solver

            @param nvars Number of variables in a param vector
            @param callback "Long" callback, produces jacobian and residuals for each energy term, returns true on success
            @param settings LevMarq settings structure, see LevMarqBase class for details
            @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
            @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
            @param paramType Type of optimized parameters; only LINEAR is supported now
            @param nerrs Energy terms amount. If zero, callback-generated jacobian size is used instead
            @param solveMethod What method to use for linear system solving
        */
        LevMarq(int nvars, LongCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
                MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, int nerrs = 0, int solveMethod = DECOMP_SVD);
        /**
            Creates a solver

            @param nvars Number of variables in a param vector
            @param callback Normal callback, produces J^T*J and J^T*b directly instead of J and b, returns true on success
            @param settings LevMarq settings structure, see LevMarqBase class for details
            @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
            @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
            @param paramType Type of optimized parameters; only LINEAR is supported now
            @param LtoR Indicates what part of symmetric matrix to copy to another part: lower or upper. Used only with alt. callback
            @param solveMethod What method to use for linear system solving
        */
        LevMarq(int nvars, NormalCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
                MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, bool LtoR = false, int solveMethod = DECOMP_SVD);

        /**
            Creates a solver

            @param param Input/output vector containing starting param vector and resulting optimized params
            @param callback "Long" callback, produces jacobian and residuals for each energy term, returns true on success
            @param settings LevMarq settings structure, see LevMarqBase class for details
            @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
            @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
            @param paramType Type of optimized parameters; only LINEAR is supported now
            @param nerrs Energy terms amount. If zero, callback-generated jacobian size is used instead
            @param solveMethod What method to use for linear system solving
        */
        LevMarq(InputOutputArray param, LongCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
                MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, int nerrs = 0, int solveMethod = DECOMP_SVD);
        /**
            Creates a solver

            @param param Input/output vector containing starting param vector and resulting optimized params
            @param callback Normal callback, produces J^T*J and J^T*b directly instead of J and b, returns true on success
            @param settings LevMarq settings structure, see LevMarqBase class for details
            @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
            @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
            @param paramType Type of optimized parameters; only LINEAR is supported now
            @param LtoR Indicates what part of symmetric matrix to copy to another part: lower or upper. Used only with alt. callback
            @param solveMethod What method to use for linear system solving
        */
        LevMarq(InputOutputArray param, NormalCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
                MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, bool LtoR = false, int solveMethod = DECOMP_SVD);

        /**
            Runs Levenberg-Marquadt algorithm using current settings and given parameters vector.
            The method returns the optimization report.
        */
        Report optimize();

        /** @brief Runs optimization using the passed vector of parameters as the start point.

            The final vector of parameters (whether the algorithm converged or not) is stored at the same
            vector.
            This method can be used instead of the optimize() method if rerun with different start points is required.
            The method returns the optimization report.

            @param param initial/final vector of parameters.

            Note that the dimensionality of parameter space is defined by the size of param vector,
            and the dimensionality of optimized criteria is defined by the size of err vector
            computed by the callback.
        */
        Report run(InputOutputArray param);

    private:
        class Impl;
        Ptr<Impl> pImpl;
    };

    namespace detail {
        class CV_EXPORTS LevMarqBackend
        {
        public:
            virtual ~LevMarqBackend() { }

            // enables geodesic acceleration support in a backend, returns true on success
            virtual bool enableGeo() = 0;

            // calculates an energy and/or jacobian at probe param vector
            virtual bool calcFunc(double& energy, bool calcEnergy = true, bool calcJacobian = false) = 0;

            // adds x to current variables and writes the sum to probe var
            // or to geodesic acceleration var if geo flag is set
            virtual void currentOplusX(const Mat_<double>& x, bool geo = false) = 0;

            // allocates jtj, jtb and other resources for objective function calculation, sets probeX to current X
            virtual void prepareVars() = 0;
            // returns a J^T*b vector (aka gradient)
            virtual const Mat_<double> getJtb() = 0;
            // returns a J^T*J diagonal vector
            virtual const Mat_<double> getDiag() = 0;
            // sets a J^T*J diagonal
            virtual void setDiag(const Mat_<double>& d) = 0;
            // performs jacobi scaling if the option is turned on
            virtual void doJacobiScaling(const Mat_<double>& di) = 0;

            // decomposes LevMarq matrix before solution
            virtual bool decompose() = 0;
            // solves LevMarq equation (J^T*J + lmdiag) * x = -right for current iteration using existing decomposition
            // right can be equal to J^T*b for LevMarq equation or J^T*rvv for geodesic acceleration equation
            virtual bool solveDecomposed(const Mat_<double>& right, Mat_<double>& x) = 0;

            // calculates J^T*f(geo) where geo is geodesic acceleration variable
            // this is used for J^T*rvv calculation for geodesic acceleration
            // calculates J^T*rvv where rvv is second directional derivative of the function in direction v
            // rvv = (f(x0 + v*h) - f(x0))/h - J*v)/h
            // where v is a LevMarq equation solution
            virtual bool calcJtbv(Mat_<double>& jtbv) = 0;

            // sets current params vector to probe params
            virtual void acceptProbe() = 0;
        };

    class CV_EXPORTS LevMarqBase
    {
    public:
        virtual ~LevMarqBase() { }

        // runs optimization using given termination conditions
        virtual LevMarq::Report optimize();

        LevMarqBase(const Ptr<LevMarqBackend>& backend_, const LevMarq::Settings& settings_):
                backend(backend_), settings(settings_)
        { }

        Ptr<LevMarqBackend> backend;
        LevMarq::Settings settings;
    };
    }
#endif

void projectPoints5x(
                    InputArray objectPoints,
                    InputArray rvec, InputArray tvec,
                    InputArray cameraMatrix, InputArray distCoeffs,
                    OutputArray imagePoints, OutputArray dpdr,
                    OutputArray dpdt, OutputArray dpdf=noArray(),
                    OutputArray dpdc=noArray(), OutputArray dpdk=noArray(),
                    OutputArray dpdo=noArray(), double aspectRatio=0.);

template <typename FLOAT>
void computeTiltProjectionMatrix(FLOAT tauX,
    FLOAT tauY,
    Matx<FLOAT, 3, 3>* matTilt = 0,
    Matx<FLOAT, 3, 3>* dMatTiltdTauX = 0,
    Matx<FLOAT, 3, 3>* dMatTiltdTauY = 0,
    Matx<FLOAT, 3, 3>* invMatTilt = 0)
{
    FLOAT cTauX = cos(tauX);
    FLOAT sTauX = sin(tauX);
    FLOAT cTauY = cos(tauY);
    FLOAT sTauY = sin(tauY);
    Matx<FLOAT, 3, 3> matRotX = Matx<FLOAT, 3, 3>(1,0,0,0,cTauX,sTauX,0,-sTauX,cTauX);
    Matx<FLOAT, 3, 3> matRotY = Matx<FLOAT, 3, 3>(cTauY,0,-sTauY,0,1,0,sTauY,0,cTauY);
    Matx<FLOAT, 3, 3> matRotXY = matRotY * matRotX;
    Matx<FLOAT, 3, 3> matProjZ = Matx<FLOAT, 3, 3>(matRotXY(2,2),0,-matRotXY(0,2),0,matRotXY(2,2),-matRotXY(1,2),0,0,1);
    if (matTilt)
    {
        // Matrix for trapezoidal distortion of tilted image sensor
        *matTilt = matProjZ * matRotXY;
    }
    if (dMatTiltdTauX)
    {
        // Derivative with respect to tauX
        Matx<FLOAT, 3, 3> dMatRotXYdTauX = matRotY * Matx<FLOAT, 3, 3>(0,0,0,0,-sTauX,cTauX,0,-cTauX,-sTauX);
        Matx<FLOAT, 3, 3> dMatProjZdTauX = Matx<FLOAT, 3, 3>(dMatRotXYdTauX(2,2),0,-dMatRotXYdTauX(0,2),
          0,dMatRotXYdTauX(2,2),-dMatRotXYdTauX(1,2),0,0,0);
        *dMatTiltdTauX = (matProjZ * dMatRotXYdTauX) + (dMatProjZdTauX * matRotXY);
    }
    if (dMatTiltdTauY)
    {
        // Derivative with respect to tauY
        Matx<FLOAT, 3, 3> dMatRotXYdTauY = Matx<FLOAT, 3, 3>(-sTauY,0,-cTauY,0,0,0,cTauY,0,-sTauY) * matRotX;
        Matx<FLOAT, 3, 3> dMatProjZdTauY = Matx<FLOAT, 3, 3>(dMatRotXYdTauY(2,2),0,-dMatRotXYdTauY(0,2),
          0,dMatRotXYdTauY(2,2),-dMatRotXYdTauY(1,2),0,0,0);
        *dMatTiltdTauY = (matProjZ * dMatRotXYdTauY) + (dMatProjZdTauY * matRotXY);
    }
    if (invMatTilt)
    {
        FLOAT inv = 1./matRotXY(2,2);
        Matx<FLOAT, 3, 3> invMatProjZ = Matx<FLOAT, 3, 3>(inv,0,inv*matRotXY(0,2),0,inv,inv*matRotXY(1,2),0,0,1);
        *invMatTilt = matRotXY.t()*invMatProjZ;
    }
}
}
