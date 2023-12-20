#pragma once
#include <videodenoise/common.hpp>

namespace cv{

    class DISOpticalFlowV2 {
    public:
        enum{
            PRESET_ULTRAFAST = 0,
            PRESET_FAST = 1,
            PRESET_MEDIUM = 2
        };

        virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow) = 0;
        virtual void collectGarbage() = 0;

        /** @brief Finest level of the Gaussian pyramid on which the flow is computed (zero level
            corresponds to the original image resolution). The final flow is obtained by bilinear upscaling.
            @see setFinestScale */
        virtual int getFinestScale() const = 0;
        /** @copybrief getFinestScale @see getFinestScale */
        virtual void setFinestScale(int val) = 0;

        /** @brief Size of an image patch for matching (in pixels). Normally, default 8x8 patches work well
            enough in most cases.
            @see setPatchSize */
        virtual int getPatchSize() const = 0;
        /** @copybrief getPatchSize @see getPatchSize */
        virtual void setPatchSize(int val) = 0;

        /** @brief Stride between neighbor patches. Must be less than patch size. Lower values correspond
            to higher flow quality.
            @see setPatchStride */
        virtual int getPatchStride() const = 0;
        /** @copybrief getPatchStride @see getPatchStride */
        virtual void setPatchStride(int val) = 0;

        /** @brief Maximum number of gradient descent iterations in the patch inverse search stage. Higher values
            may improve quality in some cases.
            @see setGradientDescentIterations */
        virtual int getGradientDescentIterations() const = 0;
        /** @copybrief getGradientDescentIterations @see getGradientDescentIterations */
        virtual void setGradientDescentIterations(int val) = 0;

        /** @brief Number of fixed point iterations of variational refinement per scale. Set to zero to
            disable variational refinement completely. Higher values will typically result in more smooth and
            high-quality flow.
        @see setGradientDescentIterations */
        virtual int getVariationalRefinementIterations() const = 0;
        /** @copybrief getGradientDescentIterations @see getGradientDescentIterations */
        virtual void setVariationalRefinementIterations(int val) = 0;

        /** @brief Weight of the smoothness term
        @see setVariationalRefinementAlpha */
        virtual float getVariationalRefinementAlpha() const = 0;
        /** @copybrief getVariationalRefinementAlpha @see getVariationalRefinementAlpha */
        virtual void setVariationalRefinementAlpha(float val) = 0;

        /** @brief Weight of the color constancy term
        @see setVariationalRefinementDelta */
        virtual float getVariationalRefinementDelta() const = 0;
        /** @copybrief getVariationalRefinementDelta @see getVariationalRefinementDelta */
        virtual void setVariationalRefinementDelta(float val) = 0;

        /** @brief Weight of the gradient constancy term
        @see setVariationalRefinementGamma */
        virtual float getVariationalRefinementGamma() const = 0;
        /** @copybrief getVariationalRefinementGamma @see getVariationalRefinementGamma */
        virtual void setVariationalRefinementGamma(float val) = 0;


        /** @brief Whether to use mean-normalization of patches when computing patch distance. It is turned on
            by default as it typically provides a noticeable quality boost because of increased robustness to
            illumination variations. Turn it off if you are certain that your sequence doesn't contain any changes
            in illumination.
        @see setUseMeanNormalization */
        virtual bool getUseMeanNormalization() const = 0;
        /** @copybrief getUseMeanNormalization @see getUseMeanNormalization */
        virtual void setUseMeanNormalization(bool val) = 0;

        /** @brief Whether to use spatial propagation of good optical flow vectors. This option is turned on by
            default, as it tends to work better on average and can sometimes help recover from major errors
            introduced by the coarse-to-fine scheme employed by the DIS optical flow algorithm. Turning this
            option off can make the output flow field a bit smoother, however.
        @see setUseSpatialPropagation */
        virtual bool getUseSpatialPropagation() const = 0;
        /** @copybrief getUseSpatialPropagation @see getUseSpatialPropagation */
        virtual void setUseSpatialPropagation(bool val) = 0;

        /** @brief Creates an instance of DISOpticalFlow

        @param preset one of PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
        */
        static Ptr<DISOpticalFlowV2> create(int preset = DISOpticalFlowV2::PRESET_FAST);
    };
}




