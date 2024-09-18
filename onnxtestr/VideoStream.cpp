#include "VideoStream.h"

void VideoStream::initialize() {
    cap.open(0, cv::CAP_DSHOW); // ���������� DSHOW ��� �������
    if (!cap.isOpened()) {
        throw std::runtime_error("Error opening video stream");
    }
}

cv::Mat VideoStream::getFrame() {
    cv::Mat frame;
    cap >> frame;
    return frame;
}