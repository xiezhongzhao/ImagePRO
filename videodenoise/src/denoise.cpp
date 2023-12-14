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
// @Data:    2023/11/27 14:41
// @Version: 1.0

#include <videodenoise/denoise.hpp>

namespace denoise{

    VideoDenoise::VideoDenoise(int width, int height) {
        width = width;
        height = height;
        y_size = cv::Size(width, height);
        uv_size = cv::Size(width/2, height/2);

        y_small_size = cv::Size(width/2, height/2);
        uv_small_size = cv::Size(width/4, height/4);

        yuv_pre = vector<cv::Mat>(3);
        yuv = vector<cv::Mat>(3);

        yuv_pre[0] = cv::Mat(y_size, CV_8UC1);
        yuv_pre[1] = cv::Mat(uv_size, CV_8UC1);
        yuv_pre[2] = cv::Mat(uv_size, CV_8UC1);

        yuv[0] = cv::Mat(y_size, CV_8UC1);
        yuv[1] = cv::Mat(uv_size, CV_8UC1);
        yuv[2] = cv::Mat(uv_size, CV_8UC1);

        yuv_small_pre = vector<cv::Mat>(3);
        yuv_small = vector<cv::Mat>(3);

        yuv_small_pre[0] = cv::Mat(y_small_size, CV_8UC1);
        yuv_small_pre[1] = cv::Mat(uv_small_size, CV_8UC1);
        yuv_small_pre[2] = cv::Mat(uv_small_size, CV_8UC1);

        yuv_small[0] = cv::Mat(y_small_size, CV_8UC1);
        yuv_small[1] = cv::Mat(uv_small_size, CV_8UC1);
        yuv_small[2] = cv::Mat(uv_small_size, CV_8UC1);

        y_remap = cv::Mat(y_size, CV_8UC1);
        u_remap = cv::Mat(uv_size, CV_8UC1);
        v_remap = cv::Mat(uv_size, CV_8UC1);

        y_remap_small =  cv::Mat(y_small_size, CV_8UC1);
        u_remap_small =  cv::Mat(uv_small_size, CV_8UC1);
        v_remap_small =  cv::Mat(uv_small_size, CV_8UC1);

        flow_op = cv::Mat(y_size, CV_32FC2); // Δ
        flow_op_displacement = cv::Mat(y_size, CV_32FC2); // (x+Δx, y+Δy)
        flow_op2_displacement = cv::Mat(uv_size, CV_32FC2);

        flow_op_small = cv::Mat(y_small_size, CV_32FC2);
        flow_op_small_displacement = cv::Mat(y_small_size, CV_32FC2);
        flow_op2_small_displacement = cv::Mat(uv_small_size, CV_32FC2);

        denoised_y = cv::Mat(y_size, CV_8UC1);
        denoised_u = cv::Mat(uv_size, CV_8UC1);
        denoised_v = cv::Mat(uv_size, CV_8UC1);

        denoised_y_small = cv::Mat(y_small_size, CV_8UC1);
        denoised_u_small = cv::Mat(uv_small_size, CV_8UC1);
        denoised_v_small = cv::Mat(uv_small_size, CV_8UC1);

        Dis =  cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);

    }

    VideoDenoise::~VideoDenoise() = default;

    void VideoDenoise::EstimateMotion(vector<cv::Mat> &yuv_pre_, vector<cv::Mat> &yuv_cur_) {

        yuv_pre = yuv_pre_;
        yuv = yuv_cur_;

        cv::resize(yuv_pre[0], yuv_small_pre[0], y_small_size,2, 2, cv::INTER_NEAREST);
        cv::resize(yuv_pre[1], yuv_small_pre[1], uv_small_size,2, 2, cv::INTER_NEAREST);
        cv::resize(yuv_pre[2], yuv_small_pre[2], uv_small_size,2, 2, cv::INTER_NEAREST);

        cv::resize(yuv[0], yuv_small[0], y_small_size,2, 2, cv::INTER_NEAREST);
        cv::resize(yuv[1], yuv_small[1], uv_small_size,2, 2, cv::INTER_NEAREST);
        cv::resize(yuv[2], yuv_small[2], uv_small_size,2, 2, cv::INTER_NEAREST);

        Dis->calc(yuv_small_pre[0], yuv_small[0], flow_op_small); // 5 ms
    }

    void VideoDenoise::GetYUVAbsoluteMotion() {  // 1 ms
        // Y, U, V displacement
        for(int i=0; i<flow_op_small.rows; ++i){
            cv::Vec2f* pData_src = flow_op_small.ptr<cv::Vec2f>(i);
            cv::Vec2f* pData_dst = flow_op_small_displacement.ptr<cv::Vec2f>(i);
            cv::Vec2f* pData2_dst = flow_op2_small_displacement.ptr<cv::Vec2f>(i/2);
            for(int j=0; j<flow_op_small.cols; ++j){
                pData_dst[j][0] = pData_src[j][0] + j;
                pData_dst[j][1] = pData_src[j][1] + i;
                pData2_dst[j/2][0] = pData_src[j][0]/2 + j/2;
                pData2_dst[j/2][1] = pData_src[j][1]/2 + i/2;
            }
        }
    }

    void VideoDenoise::RemapYUV() { // 2.2ms

        // split y displacement
        Mat flow_y_small_displacement[2];
        split(flow_op_small_displacement, flow_y_small_displacement);

        // split uv displacemnet
        cv::Mat flow_uv_small_displacement[2];
        split(flow_op2_small_displacement, flow_uv_small_displacement);

        // remap
        remap(yuv_small_pre[0], y_remap_small, flow_y_small_displacement[0], flow_y_small_displacement[1],cv::INTER_NEAREST);
        remap(yuv_small_pre[1], u_remap_small, flow_uv_small_displacement[0], flow_uv_small_displacement[1],cv::INTER_NEAREST);
        remap(yuv_small_pre[2], v_remap_small, flow_uv_small_displacement[0], flow_uv_small_displacement[1],cv::INTER_NEAREST);

        // enlarger 2×
        cv::resize(y_remap_small, y_remap, y_size, 0.5, 0.5, cv::INTER_NEAREST);
    }

    void VideoDenoise::Blend(cv::Mat& pre, cv::Mat& cur, int multiple){ // 12ms->3ms ???
        int pb = PB / multiple;  // 16
        int pe = PE / multiple;  // 32
        int wb = WB;  //  0.5
        int we = WE;  //  1.0

        float ratio_weight1 = (we - wb) / (pe - pb);  // 0.5/16
        float ratio_weight2 = pb * we - pe * wb; // 16 - 16
        cv::Mat diff;
        cv::absdiff(cur, pre, diff);
        const int height = diff.rows;
        const int width = diff.cols;

        float ratio[256], ratio_mi[256];
        for(int delta=0; delta<=255; ++delta){
            float weight = 0.f;
            if(delta <= pb){
                weight = wb;
            }else if(delta > pe){
                weight = we;
            }else{
                weight = ratio_weight1 * (static_cast<float>(delta) + ratio_weight2);
            }
            ratio[delta] = weight;
            ratio_mi[delta] = 1.f - weight;
        }

        cv::Mat curCopy;
        cur.copyTo(curCopy);

        unsigned char* pDiff = diff.data;
        unsigned char* pPre = pre.data;
        unsigned char* pCur = cur.data;
        unsigned char* pCurCopy = curCopy.data;

#ifdef SSE   // 6.5 ms good !!!
        for(int y=0; y<height; ++y){
            unsigned char* linePDiff = pDiff + y * width;
            unsigned char* linePPre = pPre + y * width;
            unsigned char* linePCur = pCur + y * width;
            unsigned char* linePCurCopy = pCurCopy + y * width;
            int x = 0;
            for(x=0; x<width-16; x+=16){
                // 计算权重
                __m128 weight1 = _mm_set_ps(ratio[linePDiff[x+3]], ratio[linePDiff[x+2]],
                                            ratio[linePDiff[x+1]], ratio[linePDiff[x]]);
                __m128 weight_mi1 = _mm_set_ps(ratio_mi[linePDiff[x+3]], ratio_mi[linePDiff[x+2]],
                                               ratio_mi[linePDiff[x+1]], ratio_mi[linePDiff[x]]);

                __m128 weight2 = _mm_set_ps(ratio[linePDiff[x+7]], ratio[linePDiff[x+6]],
                                            ratio[linePDiff[x+5]], ratio[linePDiff[x+4]]);
                __m128 weight_mi2 = _mm_set_ps(ratio_mi[linePDiff[x+7]], ratio_mi[linePDiff[x+6]],
                                               ratio_mi[linePDiff[x+5]], ratio_mi[linePDiff[x+4]]);

                __m128 weight3 = _mm_set_ps(ratio[linePDiff[x+11]], ratio[linePDiff[x+10]],
                                            ratio[linePDiff[x+9]], ratio[linePDiff[x+8]]);
                __m128 weight_mi3 = _mm_set_ps(ratio_mi[linePDiff[x+11]], ratio_mi[linePDiff[x+10]],
                                               ratio_mi[linePDiff[x+9]], ratio_mi[linePDiff[x+8]]);

                __m128 weight4 = _mm_set_ps(ratio[linePDiff[x+15]], ratio[linePDiff[x+14]],
                                            ratio[linePDiff[x+13]], ratio[linePDiff[x+12]]);
                __m128 weight_mi4 = _mm_set_ps(ratio_mi[linePDiff[x+15]], ratio_mi[linePDiff[x+14]],
                                               ratio_mi[linePDiff[x+13]], ratio_mi[linePDiff[x+12]]);

                // load the data
                __m128i pre_data = _mm_loadu_si128((__m128i*)(linePPre+x));
                __m128i cur_data = _mm_loadu_si128((__m128i*)(linePCur+x));

                __m128 pre_val1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(pre_data));
                __m128 cur_val1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(cur_data));
                __m128i result1 = _mm_cvtps_epi32(_mm_add_ps(
                                _mm_mul_ps(weight_mi1,pre_val1),
                                _mm_mul_ps(weight1, cur_val1)));

                __m128i shuffle_mask2 = _mm_setr_epi8(
                        4, 5, 6, 7,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1
                );
                __m128i pre_data2 = _mm_shuffle_epi8(pre_data, shuffle_mask2);
                __m128i cur_data2 = _mm_shuffle_epi8(cur_data, shuffle_mask2);
                __m128 pre_val2 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(pre_data2));
                __m128 cur_val2 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(cur_data2));
                __m128i result2 = _mm_cvtps_epi32(_mm_add_ps(
                        _mm_mul_ps(weight_mi2,pre_val2),
                        _mm_mul_ps(weight2, cur_val2)));

                __m128i shuffle_mask3 = _mm_setr_epi8(
                        8, 9, 10, 11,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1
                );
                __m128i pre_data3 = _mm_shuffle_epi8(pre_data, shuffle_mask3);
                __m128i cur_data3 = _mm_shuffle_epi8(cur_data, shuffle_mask3);
                __m128 pre_val3 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(pre_data3));
                __m128 cur_val3 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(cur_data3));
                __m128i result3 = _mm_cvtps_epi32(_mm_add_ps(
                        _mm_mul_ps(weight_mi3,pre_val3),
                        _mm_mul_ps(weight3, cur_val3)));

                __m128i shuffle_mask4 = _mm_setr_epi8(
                        12, 13, 14, 15,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1
                );
                __m128i pre_data4 = _mm_shuffle_epi8(pre_data, shuffle_mask4);
                __m128i cur_data4 = _mm_shuffle_epi8(cur_data, shuffle_mask4);
                __m128 pre_val4 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(pre_data4));
                __m128 cur_val4 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(cur_data4));
                __m128i result4 = _mm_cvtps_epi32(_mm_add_ps(
                        _mm_mul_ps(weight_mi4,pre_val4),
                        _mm_mul_ps(weight4, cur_val4)));

                __m128i shft1 = _mm_setr_epi8(
                        0, 4, 8, 12,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1
                );
                __m128i shft2 = _mm_setr_epi8(
                        -1, -1, -1, -1,
                        0, 4, 8, 12,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1
                );
                __m128i shft3 = _mm_setr_epi8(
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        0, 4, 8, 12,
                        -1, -1, -1, -1
                );
                __m128i shft4 = _mm_setr_epi8(
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        -1, -1, -1, -1,
                        0, 4, 8, 12
                );
                __m128i res1 = _mm_shuffle_epi8(result1, shft1);
                __m128i res2 = _mm_shuffle_epi8(result2, shft2);
                __m128i res3 = _mm_shuffle_epi8(result3, shft3);
                __m128i res4 = _mm_shuffle_epi8(result4, shft4);

                __m128i res = _mm_or_si128(
                        _mm_or_si128(res1, res2),
                        _mm_or_si128(res3, res4)
                        );

                // 将结果写回内存
                _mm_storeu_si128((__m128i*)(linePCurCopy+x), res);
            }

            for(; x<width; ++x){
                int delta = linePDiff[x];
                linePCurCopy[x] = ratio_mi[delta] * linePPre[x] + ratio[delta] * linePCur[x];
            }
        }
        curCopy.copyTo(cur);

#elif ORIG  // 10~13ms -> 7~10ms
         for(int y=0; y<height; ++y){
            unsigned char* linePDiff = pDiff + y * width;
            unsigned char* linePPre = pPre + y * width;
            unsigned char* linePCur = pCur + y * width;
            for(int x=0; x<width; ++x){
                int delta = linePDiff[x];
                linePCur[x] = ratio_mi[delta] * linePPre[x] + ratio[delta] * linePCur[x];
            }
         }
#endif
    }

    void VideoDenoise::YUVFusion(){ // 12.5ms ->
        // fusion y of the original image -> ok !!!
        cv::Mat y_src;
        yuv[0].copyTo(y_src);
        Blend(y_remap, yuv[0], 1); // fusion y (src size)
        cv::Mat y_src_denoised;
        cv::addWeighted(y_src, 0.33,
                        yuv[0], 0.67, 0, y_src_denoised);

        // fusion y of the small image -> ok !!!
        cv::Mat y_small;
        yuv_small[0].copyTo(y_small);
        Blend(y_remap_small, yuv_small[0], 2); // fusion y (small size)
        Blend(u_remap_small, yuv_small[1], 4); // fusion u (small size)
        Blend(v_remap_small, yuv_small[2], 4); // fusion v (small size)

        cv::addWeighted(y_small, 0.33,
                        yuv_small[0], 0.67, 0, yuv_small[0]);
        cv::Mat y_small_resize_denoised;
        cv::resize(yuv_small[0], y_small_resize_denoised, y_size, 0.5, 0.5, cv::INTER_NEAREST);

        // fusion the two branches
        cv::addWeighted(y_small_resize_denoised, 0.5,
                        y_src_denoised, 0.5 , 0, denoised_y);

        // fusion the current and previous denoised frame
        cv::addWeighted(yuv_pre[0], 0.33,
                        denoised_y, 0.67, 0, denoised_y);// fusion from two branches

    }

    void VideoDenoise::FilterYUV(){

//        cv::GaussianBlur(denoised_y, denoised_y, Size(3, 3), 0.6, 0.6);

        cv::GaussianBlur(yuv_small[1], denoised_u_small, Size(3, 3), 1, 1);
        cv::GaussianBlur(yuv_small[2], denoised_v_small, Size(3, 3), 1, 1);
        cv::resize(denoised_u_small, denoised_u, uv_size, 0.5, 0.5, cv::INTER_AREA);
        cv::resize(denoised_v_small, denoised_v, uv_size, 0.5, 0.5, cv::INTER_AREA);

        // fusion the previous(denoised) and current u
        cv::addWeighted(yuv_pre[1], 0.33,
                        denoised_u, 0.67, 0, denoised_u);// fusion from two branches

        // fusion the previous(denoised) and current v
        cv::addWeighted(yuv_pre[2], 0.33,
                        denoised_v, 0.67, 0, denoised_v);// fusion from two branches
    }

    void VideoDenoise::DenoiseProcess(vector<cv::Mat> &yuv_pre,
                                      vector<cv::Mat> &yuv) {
//        EstimateMotion(yuv_pre, yuv);
//        GetYUVAbsoluteMotion();
//        RemapYUV();
//        YUVFusion();
//        FilterYUV();

        timer = std::make_unique<Timer::Timer>("EstimateMotion");
        EstimateMotion(yuv_pre, yuv);
        timer->stop();

        timer = std::make_unique<Timer::Timer>("GetYUVAbsoluteMotion");
        GetYUVAbsoluteMotion();
        timer->stop();

        timer = std::make_unique<Timer::Timer>("RemapYUV");
        RemapYUV();
        timer->stop();

        timer = std::make_unique<Timer::Timer>("YUVFusion");
        YUVFusion();
        timer->stop();

        timer = std::make_unique<Timer::Timer>("FilterYUV");
        FilterYUV();
        timer->stop();
    }

    void VideoDenoise::GetDenoisedYUV(cv::Mat& y,
                                      cv::Mat& u,
                                      cv::Mat& v){
        denoised_y.copyTo(y);
        denoised_u.copyTo(u);
        denoised_v.copyTo(v);

        denoised_y.copyTo(yuv_pre[0]);
        denoised_u.copyTo(yuv_pre[1]);
        denoised_v.copyTo(yuv_pre[2]);

    }
}


