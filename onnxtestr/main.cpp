#include "VideoStream.h"
#include "PersonDetection.h"
#include <opencv2/opencv.hpp>
#include <string>

int main() {
    VideoStream videoStream;
    PersonDetection personDetection;

    videoStream.initialize();

    cv::namedWindow("Video Stream", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Stream", 1280, 720);

    int frameCount = 0;
    std::vector<cv::Rect> detectedRects;

    while (true) {
        cv::Mat frame = videoStream.getFrame();
        if (frame.empty()) break;

        std::vector<cv::Rect> people;

        if (frameCount % 2 == 0) {
            personDetection.detectPerson(frame, people);
            detectedRects = people;
        }
        else {
            auto& kalmanFilters = personDetection.getKalmanFilters();
            for (size_t i = 0; i < detectedRects.size(); ++i) {
                if (i < kalmanFilters.size()) {
                    cv::Rect predictedRect = kalmanFilters[i].predict();
                    if (predictedRect.width > 0 && predictedRect.height > 0) {
                        cv::rectangle(frame, predictedRect, cv::Scalar(255, 0, 0), 2); // Отображение предсказанного объекта синим
                    }
                }
            }
        }

        for (const auto& person : detectedRects) {
            cv::rectangle(frame, person, cv::Scalar(0, 255, 0), 2); // Зеленый прямоугольник от YOLO
        }

        frameCount++;
        cv::imshow("Video Stream", frame);
        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}
