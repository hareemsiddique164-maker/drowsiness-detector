#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <string>
#include <deque>
#include <vector>
#include <memory>
#define NOMINMAX
#include <windows.h>

// =============================================================================
// ABSTRACTION — pure interface for any alert type
// =============================================================================
class Alert {
public:
    virtual void trigger(cv::Mat& frame) = 0;
    virtual void reset()                 = 0;
    virtual ~Alert()                     = default;
};

// =============================================================================
// INHERITANCE + POLYMORPHISM — concrete alert types
// =============================================================================
class SoundAlert : public Alert {
    int cooldown = 0;
public:
    void trigger(cv::Mat& frame) override {
        if (cooldown == 0) {
            MessageBeep(MB_ICONHAND);
            cooldown = 30;
        }
        if (cooldown > 0) cooldown--;
    }
    void reset() override { cooldown = 0; }
};

class VisualAlert : public Alert {
public:
    void trigger(cv::Mat& frame) override {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, {0, 0, frame.cols, frame.rows}, {0, 0, 180}, -1);
        cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame);

        // Outlined text for readability
        cv::putText(frame, "DROWSY! WAKE UP!", {20, 60},
                    cv::FONT_HERSHEY_SIMPLEX, 1.1, {0, 0, 0}, 4);
        cv::putText(frame, "DROWSY! WAKE UP!", {20, 60},
                    cv::FONT_HERSHEY_SIMPLEX, 1.1, {0, 0, 255}, 2);
    }
    void reset() override {}
};

// =============================================================================
// ENCAPSULATION — drowsiness state logic hidden inside a class
// =============================================================================
class EyeStateTracker {
    std::deque<bool> history;
    int  consecClosed = 0;
    int  drowsyLatch  = 0;
    bool drowsy       = false;

    static const int   WINDOW_SIZE      = 20;
    static const int   CONSEC_THRESHOLD = 15;
    static constexpr float DROWSY_RATIO = 0.50f;

public:
    void update(bool faceFound, bool eyesFound) {
        if (faceFound && !eyesFound) consecClosed++;
        else                          consecClosed = 0;

        if (faceFound) {
            history.push_back(eyesFound);
            if ((int)history.size() > WINDOW_SIZE)
                history.pop_front();
        } else {
            history.clear();
        }

        float ratio     = getRatio();
        bool triggered  = (ratio >= DROWSY_RATIO) || (consecClosed >= CONSEC_THRESHOLD);

        if (triggered) {
            drowsy      = true;
            drowsyLatch = 60;
        } else if (drowsyLatch > 0) {
            drowsyLatch--;
            drowsy = (drowsyLatch > 0);
        } else {
            drowsy = false;
        }
    }

    bool  isDrowsy()       const { return drowsy; }
    int   getConsec()      const { return consecClosed; }
    bool  windowFull()     const { return (int)history.size() == WINDOW_SIZE; }

    float getRatio() const {
        if ((int)history.size() < WINDOW_SIZE) return 0.0f;
        int closed = 0;
        for (bool seen : history) if (!seen) closed++;
        return (float)closed / WINDOW_SIZE;
    }

    float getDisplayProgress() const {
        float r = std::max(getRatio(),
                           (float)consecClosed / CONSEC_THRESHOLD);
        return std::min(r, 1.0f);
    }
};

// =============================================================================
// ABSTRACTION — pure interface for any detector
// =============================================================================
class Detector {
public:
    virtual void run()  = 0;
    virtual ~Detector() = default;
};

// =============================================================================
// INHERITANCE — DrowsinessDetector implements Detector
// ENCAPSULATION — camera, cascades, tracker, alerts all private
// =============================================================================
class DrowsinessDetector : public Detector {
    cv::CascadeClassifier        faceCascade, eyeCascade;
    cv::VideoCapture             cap;
    EyeStateTracker              tracker;
    std::vector<std::unique_ptr<Alert>> alerts;  // POLYMORPHISM via base pointer

    static void drawLabel(cv::Mat& frame, const std::string& text,
                          cv::Point pos, cv::Scalar color, double scale = 0.85)
    {
        cv::putText(frame, text, pos, cv::FONT_HERSHEY_SIMPLEX, scale, {0,0,0}, 4);
        cv::putText(frame, text, pos, cv::FONT_HERSHEY_SIMPLEX, scale, color,  2);
    }

    void drawHUD(cv::Mat& frame, bool faceFound, bool eyesFound) {
        if (tracker.isDrowsy()) {
            // alerts triggered via polymorphic call below — just show bar here
        } else if (faceFound && eyesFound) {
            drawLabel(frame, "Alert",           {20, 60}, {0, 220, 0});
        } else if (faceFound) {
            drawLabel(frame, "Eyes closing...", {20, 60}, {0, 165, 255});
        } else {
            drawLabel(frame, "No face detected",{20, 60}, {180, 180, 180});
        }

        // Drowsiness progress bar
        float prog  = tracker.getDisplayProgress();
        int   barW  = 220;
        cv::Scalar barColor = prog >= 1.0f ? cv::Scalar(0,   0, 255)
                            : prog >= 0.5f ? cv::Scalar(0, 165, 255)
                                          : cv::Scalar(0, 200,   0);
        cv::rectangle(frame, {20, 75}, {20 + barW,             92}, {60,60,60},  -1);
        cv::rectangle(frame, {20, 75}, {20 + (int)(barW*prog), 92}, barColor,    -1);
        cv::putText(frame, "Drowsiness level", {20, 107},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {200,200,200}, 1);
    }

    bool detectEyes(const cv::Mat& gray, const cv::Rect& face, cv::Mat& frame) {
        cv::Rect upperFace(face.x, face.y, face.width, (int)(face.height * 0.55));
        cv::Mat  faceROI = gray(upperFace);

        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4,
                                    cv::CASCADE_SCALE_IMAGE, {15, 15});

        if (eyes.size() < 1 || eyes.size() > 2) return false;

        for (const cv::Rect& eye : eyes) {
            cv::Point center(upperFace.x + eye.x + eye.width  / 2,
                             upperFace.y + eye.y + eye.height / 2);
            cv::circle(frame, center, eye.width / 2, {0, 220, 0}, 2);
        }
        return true;
    }

public:
    DrowsinessDetector() {
        if (!faceCascade.load("haarcascade_frontalface_default.xml"))
            throw std::runtime_error("Cannot load face cascade.");
        if (!eyeCascade.load("haarcascade_eye.xml"))
            throw std::runtime_error("Cannot load eye cascade.");

        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        if (!cap.isOpened())
            throw std::runtime_error("Cannot open camera.");
    }

    // Add any alert type — POLYMORPHISM
    void addAlert(std::unique_ptr<Alert> alert) {
        alerts.push_back(std::move(alert));
    }

    void run() override {
        cv::Mat frame, gray;
        std::cout << "Drowsiness Detector running. Press 'q' to quit.\n";

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            auto clahe = cv::createCLAHE(2.0, {8, 8});
            clahe->apply(gray, gray);

            // Face detection
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(gray, faces, 1.1, 3,
                                         cv::CASCADE_SCALE_IMAGE, {50, 50});

            bool faceFound  = !faces.empty();
            bool eyesFound  = false;

            for (const cv::Rect& face : faces) {
                cv::rectangle(frame, face, {255, 180, 0}, 2);
                if (detectEyes(gray, face, frame))
                    eyesFound = true;
            }

            tracker.update(faceFound, eyesFound);

            // Trigger all registered alerts via polymorphic call
            if (tracker.isDrowsy()) {
                for (auto& alert : alerts)
                    alert->trigger(frame);
            } else {
                for (auto& alert : alerts)
                    alert->reset();
            }

            drawHUD(frame, faceFound, eyesFound);

            // Console debug
            std::cout << "\rFace:" << faceFound
                      << "  Eyes:"   << eyesFound
                      << "  Consec:" << tracker.getConsec()
                      << "  Ratio:"  << tracker.getRatio()
                      << "  DROWSY:" << tracker.isDrowsy()
                      << "       "   << std::flush;

            cv::imshow("Drowsiness Detector", frame);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }

        cap.release();
        cv::destroyAllWindows();
    }
};

// =============================================================================
// MAIN
// =============================================================================
int main()
{
    try {
        DrowsinessDetector detector;
        detector.addAlert(std::make_unique<VisualAlert>());
        detector.addAlert(std::make_unique<SoundAlert>());
        detector.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
