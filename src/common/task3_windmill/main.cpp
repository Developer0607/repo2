// main.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// ============================================================
//  目标结构体
// ============================================================
struct Target {
    int id = 0;
    Point2f centroid;
    Rect box;
    bool isCircle = false;
    int missingFrames = 0;   // 活动列表中连续丢失帧数
    int age = 0;
    int lostCounter = 0;     // 在待定池中已保留帧数

    // 外观特征
    Mat hist;                // HSV 颜色直方图
    double area = 0;
    double aspectRatio = 1;
};

// ============================================================
//  圆度：4*pi*Area / Perimeter^2，完美圆=1.0
// ============================================================
double circularity(const vector<Point>& cnt) {
    double area = contourArea(cnt);
    double peri = arcLength(cnt, true);
    if (peri < 1e-6) return 0.0;
    return 4.0 * CV_PI * area / (peri * peri);
}

// ============================================================
//  提取 HSV 颜色直方图（H:30, S:32）
// ============================================================
Mat extractHist(const Mat& frame, const Rect& box) {
    Rect safe = box & Rect(0, 0, frame.cols, frame.rows);
    if (safe.area() <= 0) return Mat();

    Mat roi = frame(safe);
    if (roi.empty()) return Mat();

    Mat hsv;
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    int hBins = 30, sBins = 32;
    int histSize[] = { hBins, sBins };
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };
    int channels[] = { 0, 1 };

    Mat hist;
    calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX);
    return hist;
}

// ============================================================
//  直方图相似度（Bhattacharyya 距离，0=完全相同，1=完全不同）
// ============================================================
double histSimilarity(const Mat& h1, const Mat& h2) {
    if (h1.empty() || h2.empty()) return 1e9;
    return compareHist(h1, h2, HISTCMP_BHATTACHARYYA);
}

// ============================================================
//  目标跟踪器
// ============================================================
class TargetTracker {
public:
    int nextId = 1;
    vector<Target> targets;          // 活动目标
    vector<Target> pendingTargets;   // 待定池（丢失但未超时）

    // 参数
    int maxMissing = 15;             // 活动目标允许连续丢失帧数
    int maxLostFrames = 90;          // 待定池保留帧数
    double maxDist = 80.0;           // 位置匹配最大距离
    double histThreshold = 0.35;     // 直方图相似度阈值
    double areaRatioThreshold = 0.5; // 面积比例下限

    // ------------------------------------------------------------
    //  主更新
    // ------------------------------------------------------------
    void update(const Mat& frame,
                const vector<Rect>& detections,
                const vector<Point2f>& centroids,
                const vector<bool>& isCircleFlags,
                const vector<double>& areas) {

        // ---- 阶段 0：活动目标丢失计数 +1 ----
        for (auto& t : targets) t.missingFrames++;

        vector<char> targetMatched(targets.size(), 0);
        vector<char> detUsed(detections.size(), 0);

        // ---- 阶段 1a：位置匹配活动目标 ----
        for (size_t d = 0; d < detections.size(); ++d) {
            double bestDist = maxDist;
            int bestIdx = -1;
            for (size_t t = 0; t < targets.size(); ++t) {
                if (targetMatched[t]) continue;
                double dist = norm(centroids[d] - targets[t].centroid);
                if (dist < bestDist) { bestDist = dist; bestIdx = (int)t; }
            }
            if (bestIdx >= 0) {
                updateTarget(targets[bestIdx], frame, detections[d],
                             centroids[d], isCircleFlags[d], areas[d]);
                targetMatched[bestIdx] = 1;
                detUsed[d] = 1;
            }
        }

        // ---- 阶段 1b：位置匹配失败的检测，尝试与待定池做特征匹配 ----
        for (size_t d = 0; d < detections.size(); ++d) {
            if (detUsed[d]) continue;

            Mat detHist = extractHist(frame, detections[d]);
            double detArea = areas[d];
            double detAspect = (double)detections[d].width /
                               max(1, detections[d].height);

            int bestPendingIdx = -1;
            double bestScore = 1e9;

            for (size_t p = 0; p < pendingTargets.size(); ++p) {
                const Target& pt = pendingTargets[p];

                // 条件1：形状类别一致
                if (pt.isCircle != isCircleFlags[d]) continue;

                // 条件2：面积比例
                double areaRatio = min(pt.area, detArea) / max(pt.area, detArea);
                if (areaRatio < areaRatioThreshold) continue;

                // 条件3：宽高比
                double aspectRatio = min(pt.aspectRatio, detAspect) /
                                     max(pt.aspectRatio, detAspect);
                if (aspectRatio < 0.5) continue;

                // 条件4：直方图相似度
                double histDist = histSimilarity(pt.hist, detHist);
                if (histDist > histThreshold) continue;

                // 综合评分
                double posDist = norm(centroids[d] - pt.centroid);
                double score = histDist + posDist / 1000.0;

                if (score < bestScore) {
                    bestScore = score;
                    bestPendingIdx = (int)p;
                }
            }

            if (bestPendingIdx >= 0) {
                // 恢复目标，保留原 ID
                Target recovered = pendingTargets[bestPendingIdx];
                recovered.centroid = centroids[d];
                recovered.box = detections[d];
                recovered.isCircle = isCircleFlags[d];
                recovered.area = detArea;
                recovered.aspectRatio = detAspect;
                recovered.hist = detHist;
                recovered.missingFrames = 0;
                recovered.age++;
                targets.push_back(recovered);

                pendingTargets.erase(pendingTargets.begin() + bestPendingIdx);
                detUsed[d] = 1;

                cout << "[Re-ID] 恢复目标 ID=" << recovered.id
                     << "  score=" << bestScore << endl;
            }
        }

        // ---- 阶段 2：剩余未匹配检测 → 新目标 ----
        for (size_t d = 0; d < detections.size(); ++d) {
            if (detUsed[d]) continue;
            Target t;
            t.id = nextId++;
            t.centroid = centroids[d];
            t.box = detections[d];
            t.isCircle = isCircleFlags[d];
            t.area = areas[d];
            t.aspectRatio = (double)detections[d].width /
                            max(1, detections[d].height);
            t.hist = extractHist(frame, detections[d]);
            t.missingFrames = 0;
            t.age = 1;
            targets.push_back(t);
            cout << "[New] 新目标 ID=" << t.id << endl;
        }

        // ---- 阶段 3：活动目标中丢失超限的 → 移入待定池 ----
        vector<Target> stillActive;
        for (auto& t : targets) {
            if (t.missingFrames > maxMissing) {
                t.lostCounter = 0;
                pendingTargets.push_back(t);
            } else {
                stillActive.push_back(t);
            }
        }
        targets.swap(stillActive);

        // ---- 阶段 4：待定池超时清理 ----
        vector<Target> stillPending;
        for (auto& p : pendingTargets) {
            p.lostCounter++;
            if (p.lostCounter <= maxLostFrames) {
                stillPending.push_back(p);
            } else {
                cout << "[Re-ID] 目标 ID=" << p.id
                     << " 超过 " << maxLostFrames
                     << " 帧未找回，永久删除" << endl;
            }
        }
        pendingTargets.swap(stillPending);
    }

    // ------------------------------------------------------------
    //  更新单个目标
    // ------------------------------------------------------------
    void updateTarget(Target& t, const Mat& frame, const Rect& box,
                      const Point2f& centroid, bool isCircle, double area) {
        t.centroid = centroid;
        t.box = box;
        t.isCircle = isCircle;
        t.area = area;
        t.aspectRatio = (double)box.width / max(1, box.height);
        t.hist = extractHist(frame, box);
        t.missingFrames = 0;
        t.age++;
    }

    // ------------------------------------------------------------
    //  绘制
    // ------------------------------------------------------------
    void draw(Mat& frame) {
        if (frame.empty()) return;

        // 活动目标
        for (const auto& t : targets) {
            if (t.missingFrames > 0) continue;
            drawOne(frame, t);
        }

        // 待定池目标（灰色虚线提示）
        for (const auto& p : pendingTargets) {
            Rect safe = p.box & Rect(0, 0, frame.cols, frame.rows);
            if (safe.area() <= 0) continue;
            rectangle(frame, safe, Scalar(128, 128, 128), 1, LINE_AA);
            putText(frame,
                    "LOST #" + to_string(p.id) + " (" +
                    to_string(maxLostFrames - p.lostCounter) + ")",
                    Point(safe.x, safe.y - 8),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(128, 128, 128), 1);
        }
    }

    void drawOne(Mat& frame, const Target& t) {
        Rect safe = t.box & Rect(0, 0, frame.cols, frame.rows);
        if (safe.area() <= 0) return;

        if (t.isCircle) {
            // 圆形目标：红色特殊标记
            rectangle(frame, safe, Scalar(0, 0, 255), 3);
            int r = max(safe.width, safe.height) / 2;
            if (r > 0) circle(frame, t.centroid, r, Scalar(0, 0, 255), 2);
            putText(frame, "CIRCLE #" + to_string(t.id),
                    Point(safe.x, safe.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        } else {
            // 普通目标：绿色框
            rectangle(frame, safe, Scalar(0, 255, 0), 2);
            putText(frame, "ID " + to_string(t.id),
                    Point(safe.x, safe.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
    }

    // 统计信息
    void stats(int& normal, int& circle, int& lost) {
        normal = circle = lost = 0;
        for (const auto& t : targets) {
            if (t.missingFrames > 0) continue;
            if (t.isCircle) circle++; else normal++;
        }
        lost = (int)pendingTargets.size();
    }
};

// ============================================================
//  main
// ============================================================
int main(int argc, char** argv) {
    string videoPath = "task_4.mp4";
    string outputPath = "output.mp4";
    if (argc > 1) videoPath = argv[1];
    if (argc > 2) outputPath = argv[2];

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "无法打开视频: " << videoPath << endl;
        return -1;
    }

    // ---- 读取视频信息，准备 VideoWriter ----
    int fps    = (int)cap.get(CAP_PROP_FPS);
    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);

    if (fps <= 0) fps = 30;  // 某些视频读不到 fps，给个默认值

    // 用 mp4v 编码输出 MP4；如果打不开就退回 MJPG + .avi
    VideoWriter writer(outputPath,
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "mp4v 编码不可用，改用 MJPG + output.avi" << endl;
        outputPath = "output.avi";
        writer.open(outputPath,
                    VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps, Size(width, height));
        if (!writer.isOpened()) {
            cerr << "VideoWriter 打开失败" << endl;
            return -1;
        }
    }

    Ptr<BackgroundSubtractorMOG2> fgbg =
        createBackgroundSubtractorMOG2(500, 40, true);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    TargetTracker tracker;
    const double CIRCLE_THRESHOLD = 0.75;
    const double MIN_AREA = 500;

    Mat frame, fgmask;
    int frameIdx = 0;
    int lastPercent = -1;

    cout << "开始处理: " << videoPath << endl;
    cout << "输出文件: " << outputPath << endl;
    cout << "分辨率: " << width << "x" << height
         << "  FPS: " << fps
         << "  总帧数: " << totalFrames << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameIdx++;

        // 通道兼容
        if (frame.channels() == 1)
            cvtColor(frame, frame, COLOR_GRAY2BGR);
        else if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        // 尺寸兜底（极少数视频中途尺寸变化）
        if (frame.size() != Size(width, height))
            resize(frame, frame, Size(width, height));

        // ---- 前景提取 ----
        fgbg->apply(frame, fgmask);
        if (fgmask.type() != CV_8UC1)
            fgmask.convertTo(fgmask, CV_8UC1);

        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel);
        morphologyEx(fgmask, fgmask, MORPH_CLOSE, kernel);
        dilate(fgmask, fgmask, Mat(), Point(-1, -1), 2);
        threshold(fgmask, fgmask, 250, 255, THRESH_BINARY);

        // ---- 轮廓检测 ----
        vector<vector<Point>> contours;
        findContours(fgmask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Rect> detBoxes;
        vector<Point2f> detCentroids;
        vector<bool> detIsCircle;
        vector<double> detAreas;

        for (const auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area < MIN_AREA) continue;

            Rect box = boundingRect(cnt);
            if (box.width <= 0 || box.height <= 0) continue;

            Point2f c(box.x + box.width / 2.0f,
                      box.y + box.height / 2.0f);
            bool isCircle = circularity(cnt) > CIRCLE_THRESHOLD;

            detBoxes.push_back(box);
            detCentroids.push_back(c);
            detIsCircle.push_back(isCircle);
            detAreas.push_back(area);
        }

        // ---- 跟踪 + 绘制 ----
        tracker.update(frame, detBoxes, detCentroids, detIsCircle, detAreas);
        tracker.draw(frame);

        // ---- 左上角统计 ----
        int normalCnt, circleCnt, lostCnt;
        tracker.stats(normalCnt, circleCnt, lostCnt);
        putText(frame,
                "Frame " + to_string(frameIdx) +
                "  Total:" + to_string(normalCnt + circleCnt) +
                "  Normal:" + to_string(normalCnt) +
                "  Circle:" + to_string(circleCnt) +
                "  Lost:" + to_string(lostCnt),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6,
                Scalar(255, 255, 255), 2);

        // ---- 写入输出视频 ----
        writer.write(frame);

        // ---- 控制台进度 ----
        if (totalFrames > 0) {
            int percent = (int)(100.0 * frameIdx / totalFrames);
            if (percent != lastPercent && percent % 5 == 0) {
                lastPercent = percent;
                printf("\r进度: %3d%%  (%d/%d 帧)",
                       percent, frameIdx, totalFrames);
                fflush(stdout);
            }
        } else {
            // 读不到总帧数，每 30 帧打印一次
            if (frameIdx % 30 == 0) {
                printf("\r已处理 %d 帧", frameIdx);
                fflush(stdout);
            }
        }
    }

    printf("\r进度: 100%%  (%d/%d 帧)\n", frameIdx, totalFrames > 0 ? totalFrames : frameIdx);
    cout << "完成，输出文件: " << outputPath << endl;

    writer.release();
    cap.release();
    return 0;
}