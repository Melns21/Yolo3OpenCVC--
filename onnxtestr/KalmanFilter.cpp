#include "KalmanFilter.h"

KalmanFilter::KalmanFilter() : kf(4, 2, 0), measurement(2, 1, CV_32F), objectWidth(100), objectHeight(100) {
    // Инициализация матриц
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    kf.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        1e-2, 0, 0, 0,
        0, 1e-2, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
        1e-1, 0,
        0, 1e-1);

    kf.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
    measurement.setTo(cv::Scalar(0));
}

void KalmanFilter::initialize(const cv::Rect& initialMeasurement) {
    // Инициализация состояния фильтра Калмана
    measurement.at<float>(0) = static_cast<float>(initialMeasurement.x + initialMeasurement.width / 2);
    measurement.at<float>(1) = static_cast<float>(initialMeasurement.y + initialMeasurement.height / 2);

    kf.statePost = (cv::Mat_<float>(4, 1) <<
        measurement.at<float>(0),
        measurement.at<float>(1),
        0,
        0);

    // Сохраняем ширину и высоту от YOLO
    objectWidth = initialMeasurement.width;
    objectHeight = initialMeasurement.height;
}

void KalmanFilter::correct(const cv::Rect& measuredState) {
    // Обновление фильтра Калмана с новыми измерениями
    measurement.at<float>(0) = static_cast<float>(measuredState.x + measuredState.width / 2);
    measurement.at<float>(1) = static_cast<float>(measuredState.y + measuredState.height / 2);

    // Обновляем ширину и высоту объекта
    objectWidth = measuredState.width;
    objectHeight = measuredState.height;

    kf.correct(measurement);
}

cv::Rect KalmanFilter::predict() {
    cv::Mat prediction = kf.predict();
    int centerX = static_cast<int>(prediction.at<float>(0));
    int centerY = static_cast<int>(prediction.at<float>(1));

    // Используем ширину и высоту от последнего обновления
    return cv::Rect(centerX - objectWidth / 2, centerY - objectHeight / 2, objectWidth, objectHeight);
}
