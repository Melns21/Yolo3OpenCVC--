#include "PersonDetection.h"

PersonDetection::PersonDetection() {
    modelConfiguration = "C:/Users/shmon/source/repos/test/test/yolov3.cfg"; // Путь к конфигурационному файлу YOLOv4
    modelWeights = "C:/Users/shmon/source/repos/test/test/yolov3.weights";   // Путь к весам YOLOv4

    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    if (net.empty()) {
        throw std::runtime_error("Ошибка загрузки модели YOLOv4");
    }
}

void PersonDetection::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(190, 190), cv::Scalar(0, 0, 0), true, false);
}

bool PersonDetection::isDuplicate(const cv::Rect& newRect, const std::vector<cv::Rect>& existingRects) {
    for (const auto& rect : existingRects) {
        double intersectionArea = (newRect & rect).area();
        double unionArea = newRect.area() + rect.area() - intersectionArea;
        if (intersectionArea / unionArea > 0.5) { // Порог перекрытия 50%
            return true;
        }
    }
    return false;
}

void PersonDetection::postprocess(const cv::Mat& frame, const cv::Mat& output, std::vector<cv::Rect>& persons) {
    for (int i = 0; i < output.rows; ++i) {
        float confidence = output.at<float>(i, 5);
        if (confidence > 0.5) { // Порог уверенности без ограничения на количество объектов
            int centerX = static_cast<int>(output.at<float>(i, 0) * frame.cols);
            int centerY = static_cast<int>(output.at<float>(i, 1) * frame.rows);
            int width = static_cast<int>(output.at<float>(i, 2) * frame.cols);
            int height = static_cast<int>(output.at<float>(i, 3) * frame.rows);

            int x = centerX - width / 2;
            int y = centerY - height / 2;

            cv::Rect personRect(x, y, width, height);

            if (!isDuplicate(personRect, persons)) {
                persons.push_back(personRect);

                if (kalmanFilters.size() < persons.size()) {
                    KalmanFilter newKalmanFilter;
                    newKalmanFilter.initialize(personRect); // Инициализация нового фильтра Калмана
                    kalmanFilters.push_back(newKalmanFilter);
                }
                else {
                    kalmanFilters[persons.size() - 1].correct(personRect);
                }
            }
        }
    }
}

void PersonDetection::detectPerson(const cv::Mat& frame, std::vector<cv::Rect>& persons) {
    cv::Mat blob;
    preprocess(frame, blob);

    net.setInput(blob);
    std::vector<cv::Mat> outs;
    std::vector<cv::String> outLayerNames = net.getUnconnectedOutLayersNames();
    net.forward(outs, outLayerNames);

    // Очистка предыдущих обнаруженных объектов и фильтров Калмана
    persons.clear();

    // Обработаем вывод сети
    for (const auto& out : outs) {
        postprocess(frame, out, persons);
    }
}

std::vector<KalmanFilter>& PersonDetection::getKalmanFilters() {
    return kalmanFilters; // Возвращаем вектор фильтров Калмана
}
