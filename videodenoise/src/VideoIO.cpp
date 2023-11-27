#include <videodenoise/VideoIO.hpp>

namespace denoise{

    vector<string> getFiles(string fileFolder, string& fileExtension){
        struct _finddata_t FileInfo;
        intptr_t Handle;
        string dir = fileFolder.append(fileExtension);
        vector<string> filenames;
        Handle = _findfirst(dir.c_str(), &FileInfo);

        if(Handle == -1L){
            printf("no matched files\n");
        }
        else{
            filenames.emplace_back(FileInfo.name);
            while(_findnext(Handle, &FileInfo) == 0){
                filenames.emplace_back(FileInfo.name);
            }
            _findclose(Handle);
        }
        return filenames;
    }

    vector<Mat> readYUV(string fileFolder, string& filename, int width, int height){
        int u_width = width >> 1;
        int u_height = height >> 1;
        vector<Mat> YVU;
        Mat Y(Size(width, height), CV_8UC1);
        Mat U(Size(u_width, u_height), CV_8UC1);
        Mat V(Size(u_width, u_height), CV_8UC1);

        FILE* f;
        string full_filename = fileFolder.append(filename);
//        std::cout << full_filename << std::endl;
        if(!(f = fopen(full_filename.c_str(), "rb"))){
            std::cout << "can not open the file" << std::endl;
            return YVU;
        }

        for(int h=0; h<height; ++h){
            for(int w=0; w<width; ++w){
                uchar p[1];
                fread(p, sizeof(uchar), 1, f);
                Y.at<uchar>(h, w) = *p;
            }
        }
        for(int h=0; h<8; ++h){
            for(int w=0; w<width; ++w){
                uchar p[1];
                fread(p, sizeof(uchar), 1, f);
            }
        }
        for(int h=0; h<u_height; ++h){
            for(int w=0; w<width; ++w){
                uchar p[1];
                fread(p, sizeof(uchar), 1, f);
                if(w%2 == 0)
                    V.at<uchar>(h, w/2) = *p;
                if(w%2 == 1)
                    U.at<uchar>(h, w/2) = *p;
            }
        }
        YVU.push_back(Y);
        YVU.push_back(U);
        YVU.push_back(V);
        fclose(f);
        return YVU;
    }

    Mat yuv2bgr(Mat& y, Mat& v, Mat& u){
        int height = y.rows;
        y = y.reshape(0, 1);
        v = v.reshape(0, 1);
        u = u.reshape(0, 1);

        Mat yv, yvu;
        hconcat(y, v, yv);
        hconcat(yv, u, yvu);
        yvu = yvu.reshape(0, height*3/2);

        Mat bgr_img;
        cvtColor(yvu, bgr_img, cv::COLOR_YUV2BGR_YV12);
        return bgr_img;
    }

} // namespace denoise
