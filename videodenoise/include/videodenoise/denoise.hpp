// MIT License
// Copyright (c) 2022 - xiezhongzhao
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// @Author:  xiezhongzhao
// @Email:   2234309583@qq.com
// @Data:    2023/11/27 13:14
// @Version: 1.0

#ifndef IMAGEPRO_DENOISE_HPP
#define IMAGEPRO_DENOISE_HPP

#include <videodenoise/common.hpp>
#include <videodenoise/DISOpticalFlowV2.hpp>
#include <videodenoise/VariationalRefinementImplV2.hpp>

namespace denoise{

class VideoDenoise{
private:
    const int PB = 16;
    const int PE = 32;
    const float WB = 0.5;
    const float WE = 1.0;

    cv::Size y_size;
    cv::Size uv_size;
    cv::Size y_small_size;
    cv::Size uv_small_size;

    vector<cv::Mat> yuv_pre;
    vector<cv::Mat> yuv;

    vector<cv::Mat> yuv_small_pre;
    vector<cv::Mat> yuv_small;

    cv::Ptr<cv::DISOpticalFlowImplV2> Dis;
    std::unique_ptr<Timer::Timer> timer;

    cv::Mat flow_op, flow_op_small;
    cv::Mat flow_op_displacement, flow_op_small_displacement;
    cv::Mat flow_op2_displacement, flow_op2_small_displacement;

    cv::Mat y_remap, u_remap, v_remap;
    cv::Mat y_remap_small, u_remap_small, v_remap_small;

    cv::Mat ypre_original, upre_original, vpre_original;

    cv::Mat denoised_y, denoised_u, denoised_v;
    cv::Mat denoised_y_small, denoised_u_small, denoised_v_small;

public:
    VideoDenoise(int width, int height);
    ~VideoDenoise();

    // relative motion
    void EstimateMotion(vector<cv::Mat>& yuv_pre, vector<cv::Mat>& yuv_cur, bool is_first_frame);

    void GetYUVAbsoluteMotion();

    void RemapYUV();

    void Fusion(cv::Mat& pre, cv::Mat& cur, int multiple);

    void YUVFusion();

    void FilterYUV();

    void DenoiseProcess(vector<cv::Mat>& yuv_pre,
                        vector<cv::Mat>& yuv);

    void GetDenoisedYUV(cv::Mat& denoised_y,
                        cv::Mat& denoised_u,
                        cv::Mat& denoised_v);

};

} // namespace denoise

#endif //IMAGEPRO_DENOISE_HPP
