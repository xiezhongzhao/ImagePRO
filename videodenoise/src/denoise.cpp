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
        raw_size = cv::Size(width, height);
        small_size = cv::Size(width/2, height/2);

        yuv_pre = vector<cv::Mat>(3);
        yuv = vector<cv::Mat>(3);

        yuv_pre[0] = cv::Mat(raw_size, CV_8UC1);
        yuv_pre[1] = cv::Mat(small_size, CV_8UC1);
        yuv_pre[2] = cv::Mat(small_size, CV_8UC1);

        yuv[0] = cv::Mat(raw_size, CV_8UC1);
        yuv[1] = cv::Mat(small_size, CV_8UC1);
        yuv[2] = cv::Mat(small_size, CV_8UC1);

        flow_op = cv::Mat(raw_size, CV_32FC2); // Δ
        flow_op_displacement = cv::Mat(raw_size, CV_32FC2); // (x+Δx, y+Δy)
        flow_op2_displacement = cv::Mat(small_size, CV_32FC2);

        Dis =  cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);

    }

    VideoDenoise::~VideoDenoise() {

    }

    void VideoDenoise::EstimateMotion(vector<cv::Mat> &yuv_pre_, vector<cv::Mat> &yuv_cur_) {
        yuv_pre[0] = yuv_pre_[0].clone();
        yuv_pre[1] = yuv_pre_[1].clone();
        yuv_pre[2] = yuv_pre_[2].clone();

        yuv[0] = yuv_cur_[0];
        yuv[1] = yuv_cur_[1];
        yuv[2] = yuv_cur_[2];

        Dis->calc(yuv_pre[0], yuv[0], flow_op); // 15.9ms
    }

    void VideoDenoise::GetYUVAbsoluteMotion() {  // 3.78ms ->
        // Y, U, V displacement
        for(int i=0; i<flow_op.rows; ++i){
            cv::Vec2f* pData_src = flow_op.ptr<cv::Vec2f>(i);
            cv::Vec2f* pData_dst = flow_op_displacement.ptr<cv::Vec2f>(i);
            cv::Vec2f* pData2_dst = flow_op2_displacement.ptr<cv::Vec2f>(i/2);
            for(int j=0; j<flow_op.cols; ++j){
                pData_dst[j][0] = pData_src[j][0] + j;
                pData_dst[j][1] = pData_src[j][1] + i;
                pData2_dst[j/2][0] = pData_src[j][0]/2 + j/2;
                pData2_dst[j/2][1] = pData_src[j][1]/2 + i/2;
            }
        }
    }

    void VideoDenoise::RemapYUV() { // 8ms ->

        // split y displacement
        Mat flow_y_displacement[2];
        split(flow_op_displacement, flow_y_displacement);

        // split uv displacemnet
        cv::Mat flow_uv_displacement[2];
        split(flow_op2_displacement, flow_uv_displacement);

        // remap
        remap(yuv_pre[0], denoised_y, flow_y_displacement[0], flow_y_displacement[1],cv::INTER_NEAREST);
        remap(yuv_pre[1], denoised_u, flow_uv_displacement[0], flow_uv_displacement[1],cv::INTER_NEAREST);
        remap(yuv_pre[2], denoised_v, flow_uv_displacement[0], flow_uv_displacement[1],cv::INTER_NEAREST);
    }

    void VideoDenoise::Fusion(){ // 12.5ms ->
        // fusion y, u, v
        float ratio;
        cv::Mat y_diff;

        cv::absdiff(yuv[0], denoised_y, y_diff);
        for(int i=0; i<y_diff.rows; ++i){
            for(int j=0; j<y_diff.cols; ++j){
                int delta = y_diff.at<uchar>(i, j);
                if(delta <= PB){ // <=15
                    ratio = WB; // 0.5
                }else if(delta > PE){  // >=30
                    ratio = WE; // 1
                }else{
                    ratio = (WE-WB)/(PE-PB)*(delta+PB*WE-PE*WB);
                }
                denoised_y.at<uchar>(i,j) = (1-ratio) * denoised_y.at<uchar>(i,j)
                                           + ratio * yuv[0].at<uchar>(i,j);
            }
        }
        cv::Mat u_diff;
        cv::absdiff(yuv[1], denoised_u, u_diff);
        for(int i=0; i<u_diff.rows; ++i){
            for(int j=0; j<u_diff.cols; ++j){
                int delta = u_diff.at<uchar>(i, j);
                if(delta <= PB){ // <=15
                    ratio = WB; // 0.5
                }else if(delta > PE){  // >=30
                    ratio = WE; // 1
                }else{
                    ratio = (WE-WB)/(PE-PB)*(delta+PB*WE-PE*WB);
                }
                denoised_u.at<uchar>(i,j) = (1-ratio) * denoised_u.at<uchar>(i,j)
                                           + ratio * yuv[1].at<uchar>(i,j);
            }
        }

        cv::Mat v_diff;
        cv::absdiff(yuv[2], denoised_v, v_diff);
        for(int i=0; i<v_diff.rows; ++i){
            for(int j=0; j<v_diff.cols; ++j){
                int delta = v_diff.at<uchar>(i, j);
                if(delta <= PB){ // <=15
                    ratio = WB; // 0.5
                }else if(delta > PE){  // >=30
                    ratio = WE; // 1
                }else{
                    ratio = (WE-WB)/(PE-PB)*(delta+PB*WE-PE*WB);
                }
                denoised_v.at<uchar>(i,j) = (1-ratio) * denoised_v.at<uchar>(i,j)
                                           + ratio * yuv[2].at<uchar>(i,j);
            }
        }
    }

    void VideoDenoise::FilterYUV(){
        cv::GaussianBlur(denoised_y, denoised_y, Size(5, 5), 1, 1);
        cv::GaussianBlur(denoised_u, denoised_u, Size(5, 5), 1, 1);
        cv::GaussianBlur(denoised_v, denoised_v, Size(5, 5), 1, 1);
    }

    void VideoDenoise::DenoiseProcess(vector<cv::Mat> &yuv_pre,
                                      vector<cv::Mat> &yuv) {

        timer = std::make_unique<Timer::Timer>("EstimateMotion");
        EstimateMotion(yuv_pre, yuv);
        timer->stop();

        timer = std::make_unique<Timer::Timer>("GetYUVAbsoluteMotion");
        GetYUVAbsoluteMotion();
        timer->stop();

        timer = std::make_unique<Timer::Timer>("RemapYUV");
        RemapYUV();
        timer->stop();

        timer = std::make_unique<Timer::Timer>("Fusion");
        Fusion();
        timer->stop();

        timer = std::make_unique<Timer::Timer>("FilterYUV");
        FilterYUV();
        timer->stop();
        cout << endl;
    }

    void VideoDenoise::GetDenoisedYUV(cv::Mat& y,
                                      cv::Mat& u,
                                      cv::Mat& v){
        y = denoised_y;
        u = denoised_u;
        v = denoised_v;
    }

}


