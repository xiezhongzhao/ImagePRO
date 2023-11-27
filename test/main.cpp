#include <videodenoise/VideoIO.hpp>
#include <videodenoise/parallel.hpp>
#include <videodenoise/DISOpticalFlowV2.hpp>
#include <videodenoise/VariationalRefinementImplV2.hpp>
#include <videodenoise/denoise.hpp>

#include <contrast/secedct.hpp>
#include <timer/timer.hpp>

#define WIDTH 1920
#define HEIGHT 1080

int main(){
    // read the yuv stream
    string prefix = "E:/WorkSpace/CPlusPlus/ImagePRO/data/";
    string fileFolder = prefix + "data6/";
    string fileExtension = "*.yuv";
    string saveVideoPath = prefix + "tcl.avi";
    Size small_size = Size(WIDTH/2, HEIGHT/2);

    // save the denoised video
    cv::VideoWriter writer;
    Size size = Size(WIDTH, HEIGHT);
    int fourcc = writer.fourcc('x', 'V', 'I', 'D');
    writer.open(saveVideoPath, fourcc, 30.0, size);

    // load the all yuv filenames
    vector<string> v_filenames = denoise::getFiles(fileFolder, fileExtension);
    std::cout << "Frames: " << v_filenames.size() << std::endl;

    int frameCount = 0;
    vector<Mat> yuv_pre(3);
    vector<Mat> yuv(3);

    std::unique_ptr<denoise::VideoDenoise> video_denoise
            = std::make_unique<denoise::VideoDenoise>(WIDTH, HEIGHT);
    while(frameCount < v_filenames.size()){
        yuv = denoise::readYUV(fileFolder, v_filenames[frameCount++], WIDTH, HEIGHT);
        if(yuv_pre[0].empty()){
            yuv_pre[0] = yuv[0].clone();
            yuv_pre[1] = yuv[1].clone();
            yuv_pre[2] = yuv[2].clone();

            Mat bgr = denoise::yuv2bgr(yuv[0], yuv[1], yuv[2]);
            writer.write(bgr);
            continue;
        }

        video_denoise->DenoiseProcess(yuv_pre, yuv); // 46ms

        cv::Mat denoisedY, denoisedU, denoisedV;
        video_denoise->GetDenoisedYUV(denoisedY, denoisedU, denoisedV);

        // IIR
        denoisedY.copyTo(yuv_pre[0]);
        denoisedU.copyTo(yuv_pre[1]);
        denoisedV.copyTo(yuv_pre[2]);

        // convert yuv into bgr`
        Mat bgr = denoise::yuv2bgr(denoisedY, denoisedU, denoisedV);
        cv::imshow("bgr", bgr);
        cv::waitKey(1);

        // write the denoised video
        writer.write(bgr);
    }
    writer.release();
    return 0;
}
