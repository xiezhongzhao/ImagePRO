#pragma once
#include <videodenoise/common.hpp>
#include <videodenoise/DISOpticalFlowV2.hpp>

namespace cv {

    class VariationalRefinementV2{
    public:
        virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow) = 0;

        virtual void collectGarbage() = 0;

        /** @brief @ref calc function overload to handle separate horizontal (u) and vertical (v) flow components
        (to avoid extra splits/merges) */
        virtual void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v) = 0;

        /** @brief Number of outer (fixed-point) iterations in the minimization procedure.
        @see setFixedPointIterations */
        virtual int getFixedPointIterations() const = 0;
        /** @copybrief getFixedPointIterations @see getFixedPointIterations */
        virtual void setFixedPointIterations(int val) = 0;

        /** @brief Number of inner successive over-relaxation (SOR) iterations
            in the minimization procedure to solve the respective linear system.
        @see setSorIterations */
        virtual int getSorIterations() const = 0;
        /** @copybrief getSorIterations @see getSorIterations */
        virtual void setSorIterations(int val) = 0;

        /** @brief Relaxation factor in SOR
        @see setOmega */
        virtual float getOmega() const = 0;
        /** @copybrief getOmega @see getOmega */
        virtual void setOmega(float val) = 0;

        /** @brief Weight of the smoothness term
        @see setAlpha */
        virtual float getAlpha() const = 0;
        /** @copybrief getAlpha @see getAlpha */
        virtual void setAlpha(float val) = 0;

        /** @brief Weight of the color constancy term
        @see setDelta */
        virtual float getDelta() const = 0;
        /** @copybrief getDelta @see getDelta */
        virtual void setDelta(float val) = 0;

        /** @brief Weight of the gradient constancy term
        @see setGamma */
        virtual float getGamma() const = 0;
        /** @copybrief getGamma @see getGamma */
        virtual void setGamma(float val) = 0;

        /** @brief Creates an instance of VariationalRefinement
        */
        static Ptr<VariationalRefinementV2> create();
    };


}




