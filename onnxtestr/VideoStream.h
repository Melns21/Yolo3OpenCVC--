#ifndef VIDEOSTREAM_H
#define VIDEOSTREAM_H

#include <opencv2/opencv.hpp>

class VideoStream {
public:
    void initialize();
    cv::Mat getFrame();

private:
    cv::VideoCapture cap;
};

#endif // VIDEOSTREAM_H
