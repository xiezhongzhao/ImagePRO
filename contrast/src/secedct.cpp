#include <contrast/secedct.hpp>

#ifdef ORIG
#define original
#elif defined(ARM_NEON)
#include <arm_neon.h>
#elif defined(SSE)
#include <immintrin.h>
#include <emmintrin.h>
#endif

namespace contrast{

    struct fkCDF {
        std::map<int, double> FkNorm;
        cv::Mat V;
    };

    cv::Mat readImg(string dir) {
        /*
        * This function will read image from the directory
        *
        * @param[in] dir the directory
        * @return the image, other is error
        */
        cv::Mat img = cv::imread(dir);

        /* check for failure */
        if (img.empty()) {
            printf("Could not open or find the image\n");
            /* wait for any key press */
            exit(0);
        }
        return img;
    }

    cv::Mat hsvChannels(cv::Mat& img) {
        /*
        * This function will get the three channels of HSV
        * @return
        */
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        return hsv;

    }

    fkCDF spatialHistgram(cv::Mat& hsv) {
        /*
        * This function will return 2D spatial histogram
        *
        * @param[in] img_hsv: the HSV image
        * @return the 2D spatial histogram
        */
        int height = hsv.rows;
        int width = hsv.cols;
        float ratio = height / float(width);
        int k = 256;

        int M = round(pow(k * ratio, 0.5));
        int N = round(pow(k / ratio, 0.5));
        int rows = M;
        int cols = N;
        cv::Mat raw(height, width, CV_8UC1);
        clock_t start, end;
        // v copy: 17ms -> 15ms -> 12ms
        unsigned char* ori = hsv.data;
        unsigned char* dst = raw.data;
        for(int y=0; y<height; ++y){
            unsigned char* linePS = ori + y * width * 3;
            unsigned char* linePD = dst + y * width;
            int x = 0;
#ifdef SSE
            for(x=0; x<width-16; x+=16, linePS+=48){  // 8k 11ms
            __m128i p1v =  _mm_loadu_si128((__m128i* )(linePS));
            __m128i shft1v = _mm_shuffle_epi8(p1v, _mm_setr_epi8(2, 5, 8, 11,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1));

            __m128i p2v = _mm_loadu_si128((__m128i*)(linePS+12));
            __m128i shft2v = _mm_shuffle_epi8(p2v, _mm_setr_epi8(2, 5, 8, 11,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1));
            __m128i shftv2 = _mm_shuffle_epi8(shft2v, _mm_setr_epi8(-1, -1, -1, -1,
                                                                    0, 1, 2, 3,
                                                                    -1, -1, -1, -1,
                                                                    -1, -1, -1, -1));

            __m128i p3v = _mm_loadu_si128((__m128i*)(linePS+24));
            __m128i shft3v = _mm_shuffle_epi8(p3v, _mm_setr_epi8(2, 5, 8, 11,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1));
            __m128i shftv3 = _mm_shuffle_epi8(shft3v, _mm_setr_epi8(-1, -1, -1, -1,
                                                                    -1, -1, -1, -1,
                                                                    0, 1, 2, 3,
                                                                    -1, -1, -1, -1));

            __m128i p4v = _mm_loadu_si128((__m128i*)(linePS+36));
            __m128i shft4v = _mm_shuffle_epi8(p4v, _mm_setr_epi8(2, 5, 8, 11,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1,
                                                                 -1, -1, -1, -1));
            __m128i shftv4 = _mm_shuffle_epi8(shft4v, _mm_setr_epi8(-1, -1, -1, -1,
                                                                    -1, -1, -1, -1,
                                                                    -1, -1, -1, -1,
                                                                    0, 1, 2, 3));
            __m128i res = _mm_or_si128(_mm_or_si128(shft1v, shftv2),
                                     _mm_or_si128(shftv3, shftv4));

            _mm_storeu_si128((__m128i *)(linePD+x), res);
        }
#elif ORIG
            for(x=0; x<width-4; x+=4, linePS+=12){  // 8k 17ms
                linePD[x] = linePS[2];
                linePD[x+1] = linePS[5];
                linePD[x+2] = linePS[8];
                linePD[x+3] = linePS[11];
            }
#endif
            for(; x<width; ++x, linePS+=3){
                linePD[x] = linePS[2];
            }
        }

        cv::Mat src;
        resize(raw, src,
               Size(hsv.cols/2, hsv.rows/2),
               0, 0, cv::INTER_LINEAR);

        /* split the image into m*n blocks */
        vector<Mat> imgParts;
        int irows = src.rows, icols = src.cols; /* the rows and columns of the image*/
        int dr = irows / rows, dc = icols / cols; /* the rows and columns of the split image*/

        int delty = (irows % rows) / 2, deltx = (icols % cols) / 2;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int x = j * dc + deltx, y = i * dr + delty;
                imgParts.push_back(src(cv::Rect(x, y, dc, dr)));
            }
        }

        /* the value k occurrences  */
        const int length = imgParts.size();
        std::unordered_map<int, Mat> histogram;
        unsigned int index = -1;
        int kk = 0;
        for (int i = 0; i < length; i++) {
            int rowRegion = imgParts[i].rows;
            int colRegion = imgParts[i].cols * imgParts[i].channels();
            ++index;
            if (imgParts[i].isContinuous()) {
                rowRegion = 1;
                colRegion = colRegion * imgParts[i].rows;
            }

            int histSize = 256;
            float range[] = { 0,256 };
            const float* histRange = { range };
            bool uniform = true, accumulate = false;
            Mat hist;
            calcHist(&imgParts[i], 1, 0,
                     Mat(), hist, 1, &histSize,
                     &histRange, uniform, accumulate);

            histogram[i] = hist;
        }

        std::unordered_map<int, vector<int>> histBlocks;
        for (auto& it : histogram) {
            int indexBlock = it.first;
            Mat block = it.second;
            for (int i = 0; i < 256; i++) {
                histBlocks[indexBlock].push_back(block.at<float>(i));
            }
        }
        histogram.clear();

        std::unordered_map<int, vector<int>> hist;
        for (int i = 0; i < 256; i++) {
            int index = 0;
            for (auto& it : histBlocks) {
                hist[i].push_back(it.second[i]);
            }
        }
        histBlocks.clear();

        double eps = 0.00001;
        std::map<int, double> entropy;
        std::unordered_map<int, vector<int>>::iterator it;
        for (it = hist.begin(); it != hist.end(); it++) {
            int key = it->first;
            vector<int> vals = it->second;
            double sum = accumulate(vals.begin(), vals.end(), 0);
            double sk = 0.0;
            for (int ele : vals) {
                double val = ele;
                val = val / (sum + eps); /* normalize */
                if (val != 0) {
                    sk += -(val * (log(val) / log(2)));
                }
            }
            entropy.insert(std::make_pair(key, sk));
        }

        double entropySum = 0.0;
        for (auto& it : entropy) { /* calculate the sum entropy */
            entropySum += it.second;
        }

        std::map<int, double> fk;
        for (auto& it : entropy) {
            double value = it.second / (entropySum - it.second + eps); /* compute a discrete function fk */
            fk.insert(std::make_pair(it.first, value));
        }
        double fkSum = 0.0;
        for (auto& it : fk) {
            fkSum += it.second;
        }

        std::map<int, double> fkNorm;
        for (auto& it : fk) {
            double value = it.second / (fkSum + eps);
            fkNorm.insert(std::make_pair(it.first, value));
        }

        int zeroNum = 0;
        for (auto& it : fkNorm) {
            int key = it.first;
            double val = it.second;
            if (val == 0) {
                ++zeroNum;
            }
        }

        if (zeroNum >= 120 ) {
            fkCDF res;
            res.FkNorm = fkNorm;
            res.V = raw;
            return res;
        }

        std::map<int, double> cdf;
        double val = 0.0000;
        for (auto& it : fkNorm) {
            int key = it.first;
            val = val + it.second;
            cdf.insert(std::make_pair(key, val));
        }

        /* mapping function: using the cumulative distribution function */
        int yu = 255;
        int yd = 0;
        std::map<int, int> ymap;
        for (auto& it : cdf) {
            int value = round(it.second * (yu - yd) + yd);
            ymap.insert(std::make_pair(it.first, value));
        }

        /* get the globally enhanced image */
        cv::Mat image;
        image.create(raw.rows, raw.cols, CV_8UC1);

        uchar lutData[256];
        int i = 0;
        for (auto& it : ymap) {
            lutData[i] = it.second;
            i++;
        }

        Mat lut(1, 256, CV_8UC1, lutData);
        cv::LUT(raw, lut, image);

        fkCDF res;
        res.FkNorm = fkNorm;
        res.V = image;
        return res;
    }

    cv::Mat dctTransform(cv::Mat& globalImg) {
        /*
        * This function will transform image into the directory
        *
        * @param[in] dir the directory
        * @return
        */
        globalImg.convertTo(globalImg, CV_64F);

        cv::Mat imgDCT;
        cv::dct(globalImg, imgDCT);

        return imgDCT;
    }

    cv::Mat domainCoefweight(cv::Mat& imgDct, std::map<int, double> fkNorm) {
        /*
        * This function will transform domain coefficient weighting
        *
        * @return
        */
        int H = imgDct.rows;
        int W = imgDct.cols;
        double sum = 0.0;
        for (auto& it : fkNorm) {
            if (it.second != 0) {
                sum += -it.second * (log(it.second) / log(2));
            }
        }

        cv::Mat imgWeight;
        imgWeight.create(H, W, CV_64F);
        double alpha = pow(sum, 0.20);

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                double weight = (1.0 + (alpha - 1.0) * i / float(H - 1.0)) *
                                (1.0 + (alpha - 1.0) * j / float(W - 1.0));
                imgWeight.at<double>(i, j) = weight * imgDct.at<double>(i, j);
            }
        }

        return imgWeight;
    }

    cv::Mat inverseDct(cv::Mat& imgWeight) {
        /*
        * This function will perform the inverse 2D-DCT transform
        * @return
        */
        imgWeight.convertTo(imgWeight, CV_64FC1);

        cv::Mat imgIDCT;
        cv::idct(imgWeight, imgIDCT);
        imgIDCT.convertTo(imgIDCT, CV_8U);

        return imgIDCT;
    }

    cv::Mat colorRestoration(cv::Mat& hsv, cv::Mat& V) {
        /*
        * This function will perform the color restoration
        * @return
        */

        int height = V.rows;
        int width = V.cols;

        unsigned char* src = V.data;
        unsigned char* dst = hsv.data;
        clock_t start, end;

        // 8k 14ms -> 11ms -> 37ms
        for (int y = 0; y < height; ++y) {
            unsigned char* linePS = src + y * width;
            unsigned char* linePD = dst + y * width * 3;
            int x = 0;

            for(x=0; x<width-8; x+=8, linePD+=24){ // 8k 10ms-11ms
                linePD[2] = linePS[x];
                linePD[5] = linePS[x+1];
                linePD[8] = linePS[x+2];
                linePD[11] = linePS[x+3];
                linePD[14] = linePS[x+4];
                linePD[17] = linePS[x+5];
                linePD[20] = linePS[x+6];
                linePD[23] = linePS[x+7];
            }
            for(; x<width; ++x, linePD+=3)
                linePD[2] = linePS[x];
        }
        cv::Mat hsvCopy(height, width, CV_8UC3, dst);

        cv::Mat newImg;
        // 8k 14ms -> 11ms ->
        start = clock();
        cv::cvtColor(hsvCopy, newImg, cv::COLOR_HSV2BGR); /* convert the hsv to rgb */
        return newImg;
    }

    void contrastEnhancement(cv::Mat& input, cv::Mat& output) {
        /*
        * This function will enhance contrast of the input image
        * @return: the enhanced image
        */
        /* image contrast enhancement */

        cv::Mat hsv = hsvChannels(input);  // 8k 13ms

        fkCDF res = spatialHistgram(hsv);  // 8k 32ms

        //cv::Mat imgDCT = dctTransform(res.V);
        //cv::Mat imgWeight = domainCoefweight(imgDCT, res.FkNorm);
        //cv::Mat imgIDCT = inverseDct(imgWeight);
        output = colorRestoration(hsv, res.V); // 8k 30ms
    }

    void SECE(cv::Mat& y){

        int height = y.rows;
        int width = y.cols;
        float ratio = height / float(width);
        int k = 256;

        int M = round(pow(k * ratio, 0.5));
        int N = round(pow(k / ratio, 0.5));
        int rows = M;
        int cols = N;
        cv::Mat raw(height, width, CV_8UC1);
        y.copyTo(raw);
        clock_t start, end;

        cv::Mat src;
        resize(raw, src,
               Size(y.cols/2, y.rows/2),
               0, 0, cv::INTER_LINEAR);

        /* split the image into m*n blocks */
        vector<Mat> imgParts;
        int irows = src.rows, icols = src.cols; /* the rows and columns of the image*/
        int dr = irows / rows, dc = icols / cols; /* the rows and columns of the split image*/

        int delty = (irows % rows) / 2, deltx = (icols % cols) / 2;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int x = j * dc + deltx, y = i * dr + delty;
                imgParts.push_back(src(cv::Rect(x, y, dc, dr)));
            }
        }

        /* the value k occurrences  */
        const int length = imgParts.size();
        std::unordered_map<int, Mat> histogram;
        unsigned int index = -1;
        int kk = 0;
        for (int i = 0; i < length; i++) {
            int rowRegion = imgParts[i].rows;
            int colRegion = imgParts[i].cols * imgParts[i].channels();
            ++index;
            if (imgParts[i].isContinuous()) {
                rowRegion = 1;
                colRegion = colRegion * imgParts[i].rows;
            }

            int histSize = 256;
            float range[] = { 0,256 };
            const float* histRange = { range };
            bool uniform = true, accumulate = false;
            Mat hist;
            calcHist(&imgParts[i], 1, 0,
                     Mat(), hist, 1, &histSize,
                     &histRange, uniform, accumulate);

            histogram[i] = hist;
        }

        std::unordered_map<int, vector<int>> histBlocks;
        for (auto& it : histogram) {
            int indexBlock = it.first;
            Mat block = it.second;
            for (int i = 0; i < 256; i++) {
                histBlocks[indexBlock].push_back(block.at<float>(i));
            }
        }
        histogram.clear();

        std::unordered_map<int, vector<int>> hist;
        for (int i = 0; i < 256; i++) {
            int index = 0;
            for (auto& it : histBlocks) {
                hist[i].push_back(it.second[i]);
            }
        }
        histBlocks.clear();

        double eps = 0.00001;
        std::map<int, double> entropy;
        std::unordered_map<int, vector<int>>::iterator it;
        for (it = hist.begin(); it != hist.end(); it++) {
            int key = it->first;
            vector<int> vals = it->second;
            double sum = accumulate(vals.begin(), vals.end(), 0);
            double sk = 0.0;
            for (int ele : vals) {
                double val = ele;
                val = val / (sum + eps); /* normalize */
                if (val != 0) {
                    sk += -(val * (log(val) / log(2)));
                }
            }
            entropy.insert(std::make_pair(key, sk));
        }

        double entropySum = 0.0;
        for (auto& it : entropy) { /* calculate the sum entropy */
            entropySum += it.second;
        }

        std::map<int, double> fk;
        for (auto& it : entropy) {
            double value = it.second / (entropySum - it.second + eps); /* compute a discrete function fk */
            fk.insert(std::make_pair(it.first, value));
        }
        double fkSum = 0.0;
        for (auto& it : fk) {
            fkSum += it.second;
        }

        std::map<int, double> fkNorm;
        for (auto& it : fk) {
            double value = it.second / (fkSum + eps);
            fkNorm.insert(std::make_pair(it.first, value));
        }

        int zeroNum = 0;
        for (auto& it : fkNorm) {
            int key = it.first;
            double val = it.second;
            if (val == 0) {
                ++zeroNum;
            }
        }

        if (zeroNum >= 120 ) {
            raw.copyTo(y);
        }

        std::map<int, double> cdf;
        double val = 0.0000;
        for (auto& it : fkNorm) {
            int key = it.first;
            val = val + it.second;
            cdf.insert(std::make_pair(key, val));
        }

        /* mapping function: using the cumulative distribution function */
        int yu = 255;
        int yd = 0;
        std::map<int, int> ymap;
        for (auto& it : cdf) {
            int value = round(it.second * (yu - yd) + yd);
            ymap.insert(std::make_pair(it.first, value));
        }

        /* get the globally enhanced image */
        cv::Mat image;
        image.create(raw.rows, raw.cols, CV_8UC1);

        uchar lutData[256];
        int i = 0;
        for (auto& it : ymap) {
            lutData[i] = it.second;
            i++;
        }

        Mat lut(1, 256, CV_8UC1, lutData);
        cv::LUT(raw, lut,y);

    }
}
