#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <opencv2/opencv.hpp>

class KalmanFilter {
public:
    KalmanFilter();
    void initialize(const cv::Rect& initialMeasurement); // Новый метод для инициализации
    void correct(const cv::Rect& measuredState);
    cv::Rect predict();

private:
    cv::KalmanFilter kf;
    cv::Mat measurement;
    int objectWidth;
    int objectHeight;
};

#endif // KALMANFILTER_H
