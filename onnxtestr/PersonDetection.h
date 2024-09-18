#ifndef PERSONDETECTION_H
#define PERSONDETECTION_H

#include <opencv2/opencv.hpp>
#include "KalmanFilter.h"

class PersonDetection {
public:
    PersonDetection();
    void detectPerson(const cv::Mat& frame, std::vector<cv::Rect>& persons);
    std::vector<KalmanFilter>& getKalmanFilters(); // Возвращаем вектор фильтров Калмана

private:
    cv::dnn::Net net;
    std::string modelConfiguration;
    std::string modelWeights;

    std::vector<KalmanFilter> kalmanFilters; // Вектор для фильтров Калмана
    std::vector<cv::Rect> trackedPersons;    // Вектор для хранения трекаемых объектов

    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    void postprocess(const cv::Mat& frame, const cv::Mat& output, std::vector<cv::Rect>& persons);
    bool isDuplicate(const cv::Rect& newRect, const std::vector<cv::Rect>& existingRects);
};

#endif // PERSONDETECTION_H
