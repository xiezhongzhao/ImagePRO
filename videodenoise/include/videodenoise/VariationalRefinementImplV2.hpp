#pragma once
#include <videodenoise/common.hpp>

namespace cv {

    class VariationalRefinementImplV2
    {
    public:
        VariationalRefinementImplV2();

        void calc(InputArray I0, InputArray I1, InputOutputArray flow) ;
        void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v);
        void collectGarbage();

    protected: //!< algorithm parameters
        int fixedPointIterations, sorIterations;
        float omega;
        float alpha, delta, gamma;
        float zeta, epsilon;

    public:
        int getFixedPointIterations() const { return fixedPointIterations; }
        void setFixedPointIterations(int val){ fixedPointIterations = val; }
        int getSorIterations() const { return sorIterations; }
        void setSorIterations(int val) { sorIterations = val; }
        float getOmega() const { return omega; }
        void setOmega(float val) { omega = val; }
        float getAlpha() const { return alpha; }
        void setAlpha(float val) { alpha = val; }
        float getDelta() const { return delta; }
        void setDelta(float val) { delta = val; }
        float getGamma() const { return gamma; }
        void setGamma(float val) { gamma = val; }

        static Ptr<VariationalRefinementImplV2> create();

    protected: //!< internal buffers
        /* This struct defines a special data layout for Mat_<float>. Original buffer is split into two: one for "red"
         * elements (sum of indices is even) and one for "black" (sum of indices is odd) in a checkerboard pattern. It
         * allows for more efficient processing in SOR iterations, more natural SIMD vectorization and parallelization
         * (Red-Black SOR). Additionally, it simplifies border handling by adding repeated borders to both red and
         * black buffers.
         */
        struct RedBlackBuffer
        {
            Mat_<float> red;   //!< (i+j)%2==0
            Mat_<float> black; //!< (i+j)%2==1

            /* Width of even and odd rows may be different */
            int red_even_len, red_odd_len;
            int black_even_len, black_odd_len;

            RedBlackBuffer();
            void create(Size s);
            void release();
        };

        Mat_<float> Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;                            //!< image derivative buffers
        RedBlackBuffer Ix_rb, Iy_rb, Iz_rb, Ixx_rb, Ixy_rb, Iyy_rb, Ixz_rb, Iyz_rb; //!< corresponding red-black buffers

        RedBlackBuffer A11, A12, A22, b1, b2; //!< main linear system coefficients
        RedBlackBuffer weights;               //!< smoothness term weights in the current fixed point iteration

        Mat_<float> mapX, mapY; //!< auxiliary buffers for remapping

        RedBlackBuffer tempW_u, tempW_v; //!< flow buffers that are modified in each fixed point iteration
        RedBlackBuffer dW_u, dW_v;       //!< optical flow increment
        RedBlackBuffer W_u_rb, W_v_rb;   //!< red-black-buffer version of the input flow

    private: //!< private methods and parallel sections
        void splitCheckerboard(RedBlackBuffer &dst, Mat &src);
        void mergeCheckerboard(Mat &dst, RedBlackBuffer &src);
        void updateRepeatedBorders(RedBlackBuffer &dst);
        void warpImage(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v);
        void prepareBuffers(Mat &I0, Mat &I1, Mat &W_u, Mat &W_v);

        /* Parallelizing arbitrary operations with 3 input/output arguments */
        typedef void (VariationalRefinementImplV2::*Op)(void *op1, void *op2, void *op3);
        struct ParallelOp_ParBody : public ParallelLoopBody
        {
            VariationalRefinementImplV2 *var;
            vector<Op> ops;
            vector<void *> op1s;
            vector<void *> op2s;
            vector<void *> op3s;

            ParallelOp_ParBody(VariationalRefinementImplV2 &_var, vector<Op> _ops, vector<void *> &_op1s,
                               vector<void *> &_op2s, vector<void *> &_op3s);
            void operator()(const Range &range) const CV_OVERRIDE;
        };
        void gradHorizAndSplitOp(void *src, void *dst, void *dst_split)
        {
//            CV_INSTRUMENT_REGION();

            Sobel(*(Mat *)src, *(Mat *)dst, -1, 1, 0, 1, 1, 0.00, BORDER_REPLICATE);
            splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
        }
        void gradVertAndSplitOp(void *src, void *dst, void *dst_split)
        {
//            CV_INSTRUMENT_REGION();

            Sobel(*(Mat *)src, *(Mat *)dst, -1, 0, 1, 1, 1, 0.00, BORDER_REPLICATE);
            splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
        }
        void averageOp(void *src1, void *src2, void *dst)
        {
//            CV_INSTRUMENT_REGION();

            addWeighted(*(Mat *)src1, 0.5, *(Mat *)src2, 0.5, 0.0, *(Mat *)dst, CV_32F);
        }
        void subtractOp(void *src1, void *src2, void *dst)
        {
//            CV_INSTRUMENT_REGION();

            subtract(*(Mat *)src1, *(Mat *)src2, *(Mat *)dst, noArray(), CV_32F);
        }

        struct ComputeDataTerm_ParBody : public ParallelLoopBody
        {
            VariationalRefinementImplV2 *var;
            int nstripes, stripe_sz;
            int h;
            RedBlackBuffer *dW_u, *dW_v;
            bool red_pass;

            ComputeDataTerm_ParBody(VariationalRefinementImplV2 &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                                    RedBlackBuffer &_dW_v, bool _red_pass);
            void operator()(const Range &range) const CV_OVERRIDE;
        };

        struct ComputeSmoothnessTermHorPass_ParBody : public ParallelLoopBody{
            VariationalRefinementImplV2 *var;
            int nstripes, stripe_sz;
            int h;
            RedBlackBuffer *W_u, *W_v, *curW_u, *curW_v;
            bool red_pass;

            ComputeSmoothnessTermHorPass_ParBody(VariationalRefinementImplV2 &_var, int _nstripes, int _h,
                                                 RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, RedBlackBuffer &_tempW_u,
                                                 RedBlackBuffer &_tempW_v, bool _red_pass);
            void operator()(const Range &range) const CV_OVERRIDE;
        };

        struct ComputeSmoothnessTermVertPass_ParBody : public ParallelLoopBody
        {
            VariationalRefinementImplV2 *var;
            int nstripes, stripe_sz;
            int h;
            RedBlackBuffer *W_u, *W_v;
            bool red_pass;

            ComputeSmoothnessTermVertPass_ParBody(VariationalRefinementImplV2 &_var, int _nstripes, int _h,
                                                  RedBlackBuffer &W_u, RedBlackBuffer &_W_v, bool _red_pass);
            void operator()(const Range &range) const CV_OVERRIDE;
        };

        struct RedBlackSOR_ParBody : public ParallelLoopBody
        {
            VariationalRefinementImplV2 *var;
            int nstripes, stripe_sz;
            int h;
            RedBlackBuffer *dW_u, *dW_v;
            bool red_pass;

            RedBlackSOR_ParBody(VariationalRefinementImplV2 &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                                RedBlackBuffer &_dW_v, bool _red_pass);
            void operator()(const Range &range) const CV_OVERRIDE;
        };
    };

}




