#include <videodenoise/VideoIO.hpp>
#include <videodenoise/parallel.hpp>
#include <videodenoise/DISOpticalFlowV2.hpp>
#include <videodenoise/VariationalRefinementImplV2.hpp>
#include <videodenoise/denoise.hpp>

#include <contrast/secedct.hpp>
#include <timer/timer.hpp>

#define WIDTH 1920
#define HEIGHT 1080

void denoiseTest(){
    // read the yuv stream
    string prefix = "E:/WorkSpace/CPlusPlus/ImagePRO/data/";
    string fileFolder = prefix + "data6/";
    string fileExtension = "*.yuv";
    string saveVideoPath = prefix + "tcl_v2.2.avi";

    // save the denoised video
    cv::VideoWriter writer;
    Size size = Size(WIDTH, HEIGHT);
    int fourcc = writer.fourcc('X', 'V', 'I', 'D'); // x v i D
    writer.open(saveVideoPath, fourcc, 30.0, size);

    // load the all yuv filenames
    vector<string> v_filenames = denoise::getFiles(fileFolder, fileExtension);
    std::cout << "Frames: " << v_filenames.size() << std::endl;

    int frameCount = 0;
    vector<Mat> yuv_pre(3);
    vector<Mat> yuv(3);

    std::unique_ptr<Timer::Timer> timer;
    std::unique_ptr<denoise::VideoDenoise> video_denoise
            = std::make_unique<denoise::VideoDenoise>(WIDTH, HEIGHT);
    while(frameCount < v_filenames.size()){

        yuv = denoise::readYUV(fileFolder, v_filenames[frameCount++], WIDTH, HEIGHT);

        if(yuv_pre[0].empty()){
            yuv[0].copyTo(yuv_pre[0]);
            cv::GaussianBlur(yuv[1], yuv[1], Size(5, 5), 1, 1);
            cv::GaussianBlur(yuv[2], yuv[2], Size(5, 5), 1, 1);
            yuv[1].copyTo(yuv_pre[1]);
            yuv[2].copyTo(yuv_pre[2]);

            Mat bgr = denoise::yuv2bgr(yuv[0], yuv[1], yuv[2]);
            writer.write(bgr);
            continue;
        }

        // contrast enhancement
        //timer = std::make_unique<Timer::Timer>("SECE");
        //contrast::SECE(yuv[0]);
        //timer->stop();

        //timer = std::make_unique<Timer::Timer>("denoise");
        video_denoise->DenoiseProcess(yuv_pre, yuv); // 5ms
        //timer->stop();

        cv::Mat denoisedY, denoisedU, denoisedV;
        video_denoise->GetDenoisedYUV(denoisedY, denoisedU, denoisedV);
        cv::imshow("y_channel", denoisedY);
        cv::waitKey(1);
        cout << endl;

        // IIR
        denoisedY.copyTo(yuv_pre[0]);
        denoisedU.copyTo(yuv_pre[1]);
        denoisedV.copyTo(yuv_pre[2]);

        // convert yuv into bgr
        Mat bgr = denoise::yuv2bgr(denoisedY, denoisedU, denoisedV);
        cv::imshow("bgr", bgr);
        cv::waitKey(1);

        // write the denoised video
        writer.write(bgr);
    }
    writer.release();
}

int main(){
    denoiseTest();
    return 0;
}
