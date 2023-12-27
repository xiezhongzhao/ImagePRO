#pragma once
#include <videodenoise/common.hpp>
#include <videodenoise/VariationalRefinementImplV2.hpp>

namespace cv{

    class DISOpticalFlowImplV2
    {
    public:
        enum{
            PRESET_ULTRAFAST = 0,
            PRESET_FAST = 1,
            PRESET_MEDIUM = 2
        };

        DISOpticalFlowImplV2();

        void calc(InputArray I0, InputArray I1, InputOutputArray flow);
        void collectGarbage();

    protected: //!< algorithm parameters
        bool is_first_frame;
        int finest_scale, coarsest_scale;
        int patch_size;
        int patch_stride;
        int grad_descent_iter;
        int variational_refinement_iter;
        float variational_refinement_alpha;
        float variational_refinement_gamma;
        float variational_refinement_delta;
        bool use_mean_normalization;
        bool use_spatial_propagation;

    protected: //!< some auxiliary variables
        int border_size;
        int w, h;   //!< flow buffer width and height on the current scale
        int ws, hs; //!< sparse flow buffer width and height on the current scale

    public:
        int getFinestScale() const { return finest_scale; }
        void setFinestScale(int val) { finest_scale = val; }
        int getPatchSize() const { return patch_size; }
        void setPatchSize(int val) { patch_size = val; }
        int getPatchStride() const { return patch_stride; }
        void setPatchStride(int val) { patch_stride = val; }
        int getGradientDescentIterations() const { return grad_descent_iter; }
        void setGradientDescentIterations(int val) { grad_descent_iter = val; }
        int getVariationalRefinementIterations() const { return variational_refinement_iter; }
        void setVariationalRefinementIterations(int val) { variational_refinement_iter = val; }
        float getVariationalRefinementAlpha() const { return variational_refinement_alpha; }
        void setVariationalRefinementAlpha(float val) { variational_refinement_alpha = val; }
        float getVariationalRefinementDelta() const { return variational_refinement_delta; }
        void setVariationalRefinementDelta(float val) { variational_refinement_delta = val; }
        float getVariationalRefinementGamma() const { return variational_refinement_gamma; }
        void setVariationalRefinementGamma(float val) { variational_refinement_gamma = val; }

        bool getUseMeanNormalization() const { return use_mean_normalization; }
        void setUseMeanNormalization(bool val) { use_mean_normalization = val; }
        bool getUseSpatialPropagation() const { return use_spatial_propagation; }
        void setUseSpatialPropagation(bool val) { use_spatial_propagation = val; }

        static Ptr<DISOpticalFlowImplV2> create(int preset = DISOpticalFlowImplV2::PRESET_FAST);

    protected:
        vector<Mat_<uchar> > I0s;     //!< Gaussian pyramid for the current frame
        vector<Mat_<uchar> > I1s;     //!< Gaussian pyramid for the next frame
        vector<Mat_<uchar> > I1s_ext; //!< I1s with borders

        vector<Mat_<short> > I0xs; //!< Gaussian pyramid for the x gradient of the current frame
        vector<Mat_<short> > I0ys; //!< Gaussian pyramid for the y gradient of the current frame

        vector<Mat_<float> > Ux; //!< x component of the flow vectors
        vector<Mat_<float> > Uy; //!< y component of the flow vectors

        vector<Mat_<float> > initial_Ux; //!< x component of the initial flow field, if one was passed as an input
        vector<Mat_<float> > initial_Uy; //!< y component of the initial flow field, if one was passed as an input

        Mat_<Vec2f> U; //!< a buffer for the merged flow

        Mat_<float> Sx; //!< intermediate sparse flow representation (x component)
        Mat_<float> Sy; //!< intermediate sparse flow representation (y component)

        /* Structure tensor components: */
        Mat_<float> I0xx_buf; //!< sum of squares of x gradient values
        Mat_<float> I0yy_buf; //!< sum of squares of y gradient values
        Mat_<float> I0xy_buf; //!< sum of x and y gradient products

        /* Extra buffers that are useful if patch mean-normalization is used: */
        Mat_<float> I0x_buf; //!< sum of x gradient values
        Mat_<float> I0y_buf; //!< sum of y gradient values

        /* Auxiliary buffers used in structure tensor computation: */
        Mat_<float> I0xx_buf_aux;
        Mat_<float> I0yy_buf_aux;
        Mat_<float> I0xy_buf_aux;
        Mat_<float> I0x_buf_aux;
        Mat_<float> I0y_buf_aux;

        vector<Ptr<VariationalRefinementImplV2>> variational_refinement_processors;

    private: //!< private methods and parallel sections
        void prepareBuffers(Mat &I0, Mat &I1, Mat &flow, bool use_flow);
        void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy,
                                       Mat &dst_I0x, Mat &dst_I0y, Mat &I0x, Mat &I0y);
        int autoSelectCoarsestScale(int img_width);
        void autoSelectPatchSizeAndScales(int img_width);

        struct PatchInverseSearch_ParBody : public ParallelLoopBody
        {
            DISOpticalFlowImplV2 *dis;
            int nstripes, stripe_sz;
            int hs;
            Mat *Sx, *Sy, *Ux, *Uy, *I0, *I1, *I0x, *I0y;
            int num_iter, pyr_level;

            PatchInverseSearch_ParBody(DISOpticalFlowImplV2 &_dis, int _nstripes, int _hs,
                                       Mat &dst_Sx, Mat &dst_Sy,
                                       Mat &src_Ux, Mat &src_Uy,
                                       Mat &_I0, Mat &_I1, Mat &_I0x, Mat &_I0y,
                                       int _num_iter, int _pyr_level);
            void operator()(const Range &range) const override;
        };

        struct Densification_ParBody : public ParallelLoopBody
        {
            DISOpticalFlowImplV2 *dis;
            int nstripes, stripe_sz;
            int h;
            Mat *Ux, *Uy, *Sx, *Sy, *I0, *I1;

            Densification_ParBody(DISOpticalFlowImplV2 &_dis, int _nstripes, int _h,
                                  Mat &dst_Ux, Mat &dst_Uy,
                                  Mat &src_Sx, Mat &src_Sy,
                                  Mat &_I0, Mat &_I1);
            void operator()(const Range &range) const override;
        };
    };

}




