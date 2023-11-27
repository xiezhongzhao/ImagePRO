#include <videodenoise/common.hpp>

namespace denoise{

    vector<string> getFiles(string fileFolder, string& fileExtension);

    vector<Mat> readYUV(string fileFolder, string& filename, int width, int height);

    Mat yuv2bgr(Mat& y, Mat& u, Mat& v);

} // namespace denoise







