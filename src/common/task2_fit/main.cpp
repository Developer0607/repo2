// Task 2: 旋转视频参数拟合 (C++ / OpenCV / Eigen)
// 模型:
//   omega(t) = b + A * sin(Omega * t + phi)
//   theta(t) = theta0 + b*t + (A/Omega) * (cos(phi) - cos(Omega*t + phi))

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Eigen::VectorXd;
using Eigen::MatrixXd;

// ==================== 配置 ====================
static const std::string VIDEO_PATH = "resources/task_2.mp4";
static const std::string OUT_DIR    = "result/task2_fit";
static const std::string MD_PATH    = "result/task2_fit_result.md";

static const double CX = 480.0;
static const double CY = 360.0;
static const double RADIUS = 220.0;
static const double DEFAULT_FPS = 60.0;

// 青色 HSV 阈值 (OpenCV H: 0-179)
static const int HSV_H_LOW  = 80;
static const int HSV_H_HIGH = 100;
static const int HSV_S_LOW  = 80;
static const int HSV_V_LOW  = 80;

// ==================== 工具 ====================
static double wrapToPi(double x) {
    x = std::fmod(x + M_PI, 2.0 * M_PI);
    if (x < 0) x += 2.0 * M_PI;
    return x - M_PI;
}

// ---- 目标检测 ----
struct DetectResult {
    std::vector<cv::Point2d> centers;
    double fps = DEFAULT_FPS;
    int    nFrames = 0;
    int    width = 0;
    int    height = 0;
};

static DetectResult detectTargetCenters(const std::string& path) {
    DetectResult res;
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        throw std::runtime_error("无法打开视频: " + path);
    }

    res.fps = cap.get(cv::CAP_PROP_FPS);
    if (res.fps <= 0) res.fps = DEFAULT_FPS;
    res.nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    res.width   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    res.height  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat frame, hsv, mask;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv,
                    cv::Scalar(HSV_H_LOW, HSV_S_LOW, HSV_V_LOW),
                    cv::Scalar(HSV_H_HIGH, 255, 255),
                    mask);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double bestScore = -std::numeric_limits<double>::infinity();
        cv::Point2d bestPt(std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN());

        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < 5.0) continue;
            cv::Moments M = cv::moments(c);
            if (M.m00 == 0) continue;
            double x = M.m10 / M.m00;
            double y = M.m01 / M.m00;
            double d = std::hypot(x - CX, y - CY);
            double score = area - 20.0 * std::fabs(d - RADIUS);
            if (score > bestScore) {
                bestScore = score;
                bestPt = cv::Point2d(x, y);
            }
        }

        if (!std::isfinite(bestPt.x)) {
            cv::Moments M = cv::moments(mask, true);
            if (M.m00 > 0) bestPt = cv::Point2d(M.m10 / M.m00, M.m01 / M.m00);
        }

        res.centers.push_back(bestPt);
    }
    cap.release();

    // 线性插值补齐漏检
    for (int j = 0; j < 2; ++j) {
        std::vector<int> bad, good;
        for (int i = 0; i < (int)res.centers.size(); ++i) {
            double v = (j == 0) ? res.centers[i].x : res.centers[i].y;
            if (std::isfinite(v)) good.push_back(i);
            else                  bad.push_back(i);
        }
        if (bad.empty()) continue;
        if (good.empty()) throw std::runtime_error("目标检测失败：全部帧未检测到青色目标");
        for (int i : bad) {
            // 找左右最近有效点
            int lo = -1, hi = -1;
            for (int g : good) { if (g < i) lo = g; else { hi = g; break; } }
            if (lo < 0) lo = hi;
            if (hi < 0) hi = lo;
            double vlo = (j == 0) ? res.centers[lo].x : res.centers[lo].y;
            double vhi = (j == 0) ? res.centers[hi].x : res.centers[hi].y;
            double t = (hi == lo) ? 0.0 : double(i - lo) / double(hi - lo);
            double v = vlo + t * (vhi - vlo);
            if (j == 0) res.centers[i].x = v;
            else        res.centers[i].y = v;
        }
    }
    return res;
}

// ---- 计算角度 ----
static void computeTheta(const std::vector<cv::Point2d>& centers,
                         VectorXd& thetaWrapped,
                         VectorXd& thetaUnwrapped)
{
    const int n = (int)centers.size();
    thetaWrapped.resize(n);
    thetaUnwrapped.resize(n);
    for (int i = 0; i < n; ++i) {
        double x = centers[i].x;
        double y = centers[i].y;
        // 图像 y 向下，cy - yi 表示数学坐标中向上
        thetaWrapped(i) = std::atan2(CY - y, x - CX);
    }
    // unwrap
    thetaUnwrapped(0) = thetaWrapped(0);
    for (int i = 1; i < n; ++i) {
        double d = thetaWrapped(i) - thetaWrapped(i - 1);
        d = wrapToPi(d);
        thetaUnwrapped(i) = thetaUnwrapped(i - 1) + d;
    }
}

// ---- Savitzky-Golay 求导 (居中, 返回 half 用于裁剪) ----
// 简单实现: 使用多项式拟合的局部窗口
static VectorXd savgolDerivative(const VectorXd& y, double dt,
                                 int polyOrder, int maxWin, int& halfWin)
{
    int n = (int)y.size();
    int win = std::min(maxWin, n);
    if (win % 2 == 0) win -= 1;
    if (win <= polyOrder) { win = polyOrder + 2; if (win % 2 == 0) win += 1; }
    if (win > n) { win = n; if (win % 2 == 0) win -= 1; }

    halfWin = win / 2;
    VectorXd dy(n);
    dy.setZero();

    // 构造范德蒙德矩阵的伪逆
    MatrixXd V(win, polyOrder + 1);
    for (int i = 0; i < win; ++i) {
        double x = i - halfWin;
        double p = 1.0;
        for (int k = 0; k <= polyOrder; ++k) {
            V(i, k) = p;
            p *= x;
        }
    }
    // 求导系数: 对多项式 p(x) = sum a_k x^k, p'(0) = a_1
    // 用伪逆得到 a = V^+ y, 一阶导系数就是 a_1 对应的行
    MatrixXd VtV = V.transpose() * V;
    MatrixXd VtV_inv = VtV.inverse();
    MatrixXd Vpinv = VtV_inv * V.transpose();
    // a = Vpinv * y_local, dy = a(1) (polyOrder>=1)
    // 向量 c 使得 dy = c^T * y_local
    VectorXd c = Vpinv.row(1).transpose(); // size win

    for (int i = 0; i < n; ++i) {
        if (i < halfWin || i >= n - halfWin) {
            // 边界: 用中心差分或前向差分
            if (i == 0)             dy(i) = (y(1) - y(0)) / dt;
            else if (i == n - 1)    dy(i) = (y(n - 1) - y(n - 2)) / dt;
            else                    dy(i) = (y(i + 1) - y(i - 1)) / (2.0 * dt);
            continue;
        }
        double acc = 0.0;
        for (int k = 0; k < win; ++k) acc += c(k) * y(i - halfWin + k);
        dy(i) = acc / dt;
    }
    return dy;
}

// ==================== 参数化 ====================
// p = [log A, log(b - A), log Omega, phi, theta0]
struct ThetaParams {
    double A, b, Omega, phi, theta0;
};

static ThetaParams unpackTheta(const VectorXd& p) {
    ThetaParams tp;
    tp.A      = std::exp(p(0));
    tp.b      = tp.A + std::exp(p(1));
    tp.Omega  = std::exp(p(2));
    tp.phi    = p(3);
    tp.theta0 = p(4);
    return tp;
}

static double modelTheta(double t, const ThetaParams& tp) {
    return tp.theta0 + tp.b * t
         + (tp.A / tp.Omega) * (std::cos(tp.phi) - std::cos(tp.Omega * t + tp.phi));
}

static double modelOmega(double t, const ThetaParams& tp) {
    return tp.b + tp.A * std::sin(tp.Omega * t + tp.phi);
}

// ---- 由 omega 序列估计初值 ----
static void initOmegaParams(const VectorXd& t, const VectorXd& y,
                            double& A0, double& b0, double& Omega0, double& phi0)
{
    int n = (int)y.size();
    b0 = y.mean();
    VectorXd yc = y.array() - b0;

    double dt = (n > 1) ? (t(1) - t(0)) : 1.0 / DEFAULT_FPS;

    // FFT (实序列)
    // 直接用 DFT 搜索主频 (n 可能较大, 但一次性可接受; 若慢可换 FFTW)
    // 为了效率, 使用 OpenCV 的 dft
    cv::Mat src(1, n, CV_64F);
    for (int i = 0; i < n; ++i) src.at<double>(0, i) = yc(i);
    cv::Mat planes[] = { src, cv::Mat::zeros(src.size(), CV_64F) };
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg);

    cv::Mat planesOut[2];
    cv::split(complexImg, planesOut);
    cv::Mat mag;
    cv::magnitude(planesOut[0], planesOut[1], mag);

    int nFreq = n / 2 + 1;
    double bestMag = -1.0;
    int bestK = 1;
    for (int k = 1; k < nFreq; ++k) {
        double m = mag.at<double>(0, k);
        if (m > bestMag) { bestMag = m; bestK = k; }
    }
    double freq = bestK / (n * dt); // Hz
    Omega0 = 2.0 * M_PI * freq;
    if (Omega0 < 1e-6) Omega0 = 1e-6;

    // 线性拟合 y ≈ b + C sin(Omega t) + S cos(Omega t)
    MatrixXd X(n, 3);
    VectorXd Y(n);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = std::sin(Omega0 * t(i));
        X(i, 2) = std::cos(Omega0 * t(i));
        Y(i) = y(i);
    }
    VectorXd coef = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
    double bLin = coef(0);
    double C = coef(1);
    double S = coef(2);
    A0 = std::hypot(C, S);
    phi0 = std::atan2(S, C);

    if (!std::isfinite(A0) || A0 < 1e-6) {
        double ptp = y.maxCoeff() - y.minCoeff();
        A0 = std::max(0.1 * ptp, 1e-3);
    }
    if (bLin <= A0) bLin = A0 + std::max(0.1 * std::fabs(bLin), 1e-3);
    b0 = bLin;
}

// ---- 简易 LM / 信赖域: 使用数值雅可比的高斯-牛顿 + 简单阻尼 ----
struct FitResult {
    ThetaParams tp;
    VectorXd thetaPred;
    VectorXd resid;
    double rmse;
    bool success;
    int iterations;
    double finalCost;
};

static FitResult fitThetaModel(const VectorXd& t, const VectorXd& theta, double fps)
{
    int n = (int)t.size();

    // 初值
    int halfWin = 0;
    VectorXd omegaAll = savgolDerivative(theta, 1.0 / fps, 3, 61, halfWin);
    double A0, b0, Omega0, phi0;
    initOmegaParams(t, omegaAll, A0, b0, Omega0, phi0);

    // theta0 初值
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double base = b0 * t(i)
                    + (A0 / Omega0) * (std::cos(phi0) - std::cos(Omega0 * t(i) + phi0));
        sum += (theta(i) - base);
    }
    double theta0_0 = sum / n;

    VectorXd p(5);
    p(0) = std::log(std::max(A0, 1e-6));
    p(1) = std::log(std::max(b0 - A0, 1e-6));
    p(2) = std::log(std::max(Omega0, 1e-6));
    p(3) = phi0;
    p(4) = theta0_0;

    auto residual = [&](const VectorXd& pp) -> VectorXd {
        ThetaParams tp = unpackTheta(pp);
        VectorXd r(n);
        for (int i = 0; i < n; ++i) r(i) = modelTheta(t(i), tp) - theta(i);
        return r;
    };

    // 高斯-牛顿 + Levenberg 阻尼
    double lambda = 1e-3;
    double cost = residual(p).squaredNorm();
    int maxIter = 200;
    bool success = false;
    int iter = 0;
    double eps = 1e-12;

    for (iter = 0; iter < maxIter; ++iter) {
        VectorXd r = residual(p);
        double c = r.squaredNorm();

        // 数值雅可比
        MatrixXd J(n, 5);
        double h = 1e-6;
        for (int j = 0; j < 5; ++j) {
            VectorXd pj = p;
            pj(j) += h;
            VectorXd rj = residual(pj);
            J.col(j) = (rj - r) / h;
        }

        MatrixXd JtJ = J.transpose() * J;
        VectorXd Jtr = J.transpose() * r;

        // 信赖域/阻尼
        MatrixXd H = JtJ;
        for (int k = 0; k < 5; ++k) H(k, k) += lambda * (1.0 + JtJ(k, k));
        VectorXd dp = H.ldlt().solve(-Jtr);

        if (!dp.allFinite()) { lambda *= 10.0; continue; }

        VectorXd pNew = p + dp;
        VectorXd rNew = residual(pNew);
        double cNew = rNew.squaredNorm();

        if (cNew < c) {
            p = pNew;
            double rel = std::fabs(c - cNew) / std::max(c, eps);
            cost = cNew;
            lambda = std::max(lambda * 0.5, 1e-12);
            if (rel < 1e-12) { success = true; break; }
            success = true;
        } else {
            lambda *= 5.0;
        }
    }

    FitResult fr;
    fr.tp = unpackTheta(p);
    fr.tp.phi = wrapToPi(fr.tp.phi);
    fr.thetaPred.resize(n);
    fr.resid.resize(n);
    for (int i = 0; i < n; ++i) {
        fr.thetaPred(i) = modelTheta(t(i), fr.tp);
        fr.resid(i) = theta(i) - fr.thetaPred(i);
    }
    fr.rmse = std::sqrt(fr.resid.squaredNorm() / n);
    fr.success = success;
    fr.iterations = iter;
    fr.finalCost = cost;
    return fr;
}

// ==================== 绘图 ====================
static void plotComparison(const VectorXd& t, const VectorXd& thetaObs,
                           const VectorXd& thetaPred, const std::string& path)
{
    const int W = 1600, H = 700;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    int m = 90;
    int pw = W - 2 * m, ph = H - 2 * m;

    double tmin = t.minCoeff(), tmax = t.maxCoeff();
    double ymin = std::min(thetaObs.minCoeff(), thetaPred.minCoeff());
    double ymax = std::max(thetaObs.maxCoeff(), thetaPred.maxCoeff());
    double dy = std::max(ymax - ymin, 1e-6);
    ymin -= 0.05 * dy; ymax += 0.05 * dy;

    auto X = [&](double tt) { return m + (tt - tmin) / (tmax - tmin) * pw; };
    auto Y = [&](double yy) { return m + ph - (yy - ymin) / (ymax - ymin) * ph; };

    cv::rectangle(img, cv::Rect(m, m, pw, ph), cv::Scalar(0, 0, 0), 1);
    for (int i = 0; i <= 5; ++i) {
        double yy = ymin + (ymax - ymin) * i / 5.0;
        int py = (int)Y(yy);
        cv::line(img, {m, py}, {m + pw, py}, cv::Scalar(220, 220, 220), 1);
        char buf[64]; std::snprintf(buf, sizeof(buf), "%.2f", yy);
        cv::putText(img, buf, {10, py + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    int step = std::max(1, (int)t.size() / 3000);
    for (int i = 0; i < (int)t.size(); i += step)
        cv::circle(img, {(int)X(t(i)), (int)Y(thetaObs(i))}, 1, cv::Scalar(0, 0, 0), -1);
    for (int i = 0; i < (int)t.size() - 1; ++i)
        cv::line(img, {(int)X(t(i)), (int)Y(thetaPred(i))},
                      {(int)X(t(i+1)), (int)Y(thetaPred(i+1))}, cv::Scalar(0, 0, 255), 2);

    cv::putText(img, "observed unwrapped angle (black) / fitted (red)",
                {m, m - 20}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(img, "time (s)", {W / 2 - 30, H - 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::imwrite(path, img);
}

static void plotResidual(const VectorXd& t, const VectorXd& resid, const std::string& path)
{
    const int W = 1600, H = 500;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    int m = 80, pw = W - 2 * m, ph = H - 2 * m;
    double tmin = t.minCoeff(), tmax = t.maxCoeff();
    double rmax = std::max(std::fabs(resid.minCoeff()), std::fabs(resid.maxCoeff()));
    rmax = std::max(rmax, 1e-6);
    auto X = [&](double tt) { return m + (tt - tmin) / (tmax - tmin) * pw; };
    auto Y = [&](double rr) { return m + ph / 2 - rr / rmax * (ph / 2 - 10); };

    cv::rectangle(img, cv::Rect(m, m, pw, ph), cv::Scalar(0, 0, 0), 1);
    cv::line(img, {m, (int)Y(0)}, {m + pw, (int)Y(0)}, cv::Scalar(0, 0, 255), 1);
    int step = std::max(1, (int)t.size() / 3000);
    for (int i = 0; i < (int)t.size(); i += step)
        cv::circle(img, {(int)X(t(i)), (int)Y(resid(i))}, 1, cv::Scalar(0, 0, 0), -1);

    cv::putText(img, "angle residual (rad)", {m, m - 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::imwrite(path, img);
}

static void plotAngularVelocity(const VectorXd& tOmega, const VectorXd& omegaObs,
                                const VectorXd& omegaPred, const std::string& path)
{
    const int W = 1600, H = 500;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    int m = 80, pw = W - 2 * m, ph = H - 2 * m;
    double tmin = tOmega.minCoeff(), tmax = tOmega.maxCoeff();
    double ymin = std::min(omegaObs.minCoeff(), omegaPred.minCoeff());
    double ymax = std::max(omegaObs.maxCoeff(), omegaPred.maxCoeff());
    double dy = std::max(ymax - ymin, 1e-6);
    ymin -= 0.1 * dy; ymax += 0.1 * dy;
    auto X = [&](double tt) { return m + (tt - tmin) / (tmax - tmin) * pw; };
    auto Y = [&](double yy) { return m + ph - (yy - ymin) / (ymax - ymin) * ph; };

    cv::rectangle(img, cv::Rect(m, m, pw, ph), cv::Scalar(0, 0, 0), 1);
    int step = std::max(1, (int)tOmega.size() / 3000);
    for (int i = 0; i < (int)tOmega.size(); i += step)
        cv::circle(img, {(int)X(tOmega(i)), (int)Y(omegaObs(i))}, 1,
                   cv::Scalar(160, 160, 160), -1);
    for (int i = 0; i < (int)tOmega.size() - 1; ++i)
        cv::line(img, {(int)X(tOmega(i)), (int)Y(omegaPred(i))},
                      {(int)X(tOmega(i+1)), (int)Y(omegaPred(i+1))},
                 cv::Scalar(0, 0, 255), 2);

    cv::putText(img, "omega(t): SavGol obs (gray) / model (red)",
                {m, m - 20}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::imwrite(path, img);
}

// ==================== 叠加视频 ====================
static void makeOverlay(const std::string& videoPath,
                        const std::vector<cv::Point2d>& centers,
                        double fps, int width, int height,
                        const ThetaParams& tp,
                        const std::string& outPath)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) throw std::runtime_error("无法打开视频: " + videoPath);

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(outPath, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) throw std::runtime_error("无法打开输出视频: " + outPath);

    std::vector<cv::Point> trail;
    int idx = 0;
    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
        if (idx < (int)centers.size()) {
            cv::Point2d c = centers[idx];
            if (std::isfinite(c.x) && std::isfinite(c.y)) {
                cv::Point p((int)std::lround(c.x), (int)std::lround(c.y));
                trail.push_back(p);
                if ((int)trail.size() > 240) trail.erase(trail.begin());

                for (size_t i = 1; i < trail.size(); ++i)
                    cv::line(frame, trail[i-1], trail[i], cv::Scalar(0, 0, 255), 2);

                cv::circle(frame, p, 9, cv::Scalar(0, 255, 0), 2);
                cv::circle(frame, p, 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::circle(frame, cv::Point((int)CX, (int)CY), 5, cv::Scalar(255, 255, 255), -1);
        cv::circle(frame, cv::Point((int)CX, (int)CY), (int)RADIUS, cv::Scalar(255, 0, 0), 1);

        double tt = idx / fps;
        double om = modelOmega(tt, tp);
        double th = modelTheta(tt, tp);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "frame=%d  t=%.3fs", idx, tt);
        cv::putText(frame, buf, {20, 32}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {255,255,255}, 2);
        std::snprintf(buf, sizeof(buf), "omega_est=%.4f rad/s", om);
        cv::putText(frame, buf, {20, 64}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {255,255,255}, 2);
        std::snprintf(buf, sizeof(buf), "theta_est=%.4f rad", th);
        cv::putText(frame, buf, {20, 96}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {255,255,255}, 2);

        writer.write(frame);
        idx++;
    }
    cap.release();
    writer.release();
}

// ==================== main ====================
int main() {
    try {
        fs::create_directories(OUT_DIR);
        fs::create_directories(fs::path(MD_PATH).parent_path());

        DetectResult det = detectTargetCenters(VIDEO_PATH);
        int n = (int)det.centers.size();
        std::cout << "检测帧数: " << n << ", fps=" << det.fps
                  << ", size=" << det.width << "x" << det.height << std::endl;

        VectorXd t(n);
        for (int i = 0; i < n; ++i) t(i) = i / det.fps;

        VectorXd thetaW, theta;
        computeTheta(det.centers, thetaW, theta);

        // 主拟合: 展开角度
        FitResult fr = fitThetaModel(t, theta, det.fps);

        double A = fr.tp.A, b = fr.tp.b, Omega = fr.tp.Omega;
        double phi = fr.tp.phi, theta0 = fr.tp.theta0;
        double rmseTheta = fr.rmse;

        // 角速度验证
        int halfWin = 0;
        VectorXd omegaAll = savgolDerivative(theta, 1.0 / det.fps, 3, 61, halfWin);
        int i0 = halfWin, i1 = n - halfWin;
        int nOmega = i1 - i0;
        VectorXd tOmega(nOmega), omegaObs(nOmega), omegaPred(nOmega);
        for (int i = 0; i < nOmega; ++i) {
            tOmega(i) = t(i0 + i);
            omegaObs(i) = omegaAll(i0 + i);
            omegaPred(i) = modelOmega(tOmega(i), fr.tp);
        }
        double rmseOmega = std::sqrt((omegaObs - omegaPred).squaredNorm() / nOmega);

        // 绘图
        plotComparison(t, theta, fr.thetaPred, OUT_DIR + "/fit_comparison.png");
        plotResidual(t, fr.resid, OUT_DIR + "/residuals.png");
        plotAngularVelocity(tOmega, omegaObs, omegaPred, OUT_DIR + "/angular_velocity.png");

        // 叠加视频
        makeOverlay(VIDEO_PATH, det.centers, det.fps, det.width, det.height,
                    fr.tp, OUT_DIR + "/tracking_overlay.mp4");

        // 写结果 md
        {
            std::ofstream md(MD_PATH);
            md << std::fixed << std::setprecision(10);
            md << "# Task 2 旋转视频参数拟合结果 (C++ / OpenCV / Eigen)\n\n";
            md << "## 模型\n\n";
            md << "角速度模型：\n\n";
            md << "\\[ \\omega(t) = b + A \\sin(\\Omega t + \\varphi) \\]\n\n";
            md << "角度积分模型：\n\n";
            md << "\\[ \\theta(t) = \\theta_0 + b t + \\frac{A}{\\Omega}"
                  " \\left( \\cos\\varphi - \\cos(\\Omega t + \\varphi) \\right) \\]\n\n";
            md << "时间以视频第 0 帧为原点，角度单位 rad，角速度单位 rad/s。\n\n";

            md << "## 方法\n\n";
            md << "1. 青色目标识别：HSV 阈值 H=[" << HSV_H_LOW << "," << HSV_H_HIGH
               << "], S>=" << HSV_S_LOW << ", V>=" << HSV_V_LOW
               << "，形态学开闭后取轮廓质心；用已知中心 (480,360) 和半径 220 筛选候选。\n";
            md << "2. 角度定义：\\(\\theta_{wrapped,i} = \\operatorname{atan2}(c_y - y_i, x_i - c_x)\\)，"
                  "再用 `unwrap` 展开。\n";
            md << "3. 参数估计：直接拟合展开角度 θ(t)，参数化为 "
                  "\\(A=e^{p_0}, b=A+e^{p_1}, \\Omega=e^{p_2}, \\varphi=p_3, \\theta_0=p_4\\)，"
                  "自动满足 A>0, b>A, Ω>0。使用 Levenberg 阻尼的高斯-牛顿法。\n";
            md << "4. 初值：Savitzky-Golay 对展开角度求导得到角速度，FFT 找主频，"
                  "线性最小二乘估计 A、b、φ。\n";
            md << "5. 主拟合量为角度，故报告角度 RMSE；角速度曲线作为验证。\n\n";

            md << "## 参数估计\n\n";
            md << "| 参数 | 估计值 | 单位 |\n|---|---:|---|\n";
            md << "| A | " << A << " | rad/s |\n";
            md << "| b | " << b << " | rad/s |\n";
            md << "| Ω | " << Omega << " | rad/s |\n";
            md << "| φ | " << phi << " | rad |\n";
            md << "| θ0 | " << theta0 << " | rad |\n\n";
            md << "速度变化周期：\\(T = 2\\pi/\\Omega = "
               << (2.0 * M_PI / Omega) << "\\ \\mathrm{s}\\)\n\n";

            md << "## 误差指标\n\n";
            md << "主拟合量：展开角度 θ(t)。\n\n";
            md << "- 角度 RMSE：" << rmseTheta << " rad，即 "
               << (rmseTheta * 180.0 / M_PI) << " deg\n";
            md << "- 有效样本数：" << n << "\n";
            md << "- 参与计算帧范围：0–" << (n - 1) << "，共 " << det.nFrames << " 帧\n";
            md << "- 视频 FPS：" << det.fps << "\n\n";

            md << "角速度验证：\n\n";
            md << "- 角速度 RMSE：" << rmseOmega << " rad/s\n";
            md << "- 有效样本数：" << nOmega << "\n";
            md << "- 参与计算帧范围：" << i0 << "–" << (i1 - 1) << "，0-based\n";
            md << "- 说明：角速度由 Savitzky-Golay 对展开角度求导得到，"
                  "窗口两端被裁剪。\n\n";

            md << "优化状态：\n\n";
            md << "- success：" << (fr.success ? "true" : "false") << "\n";
            md << "- iterations：" << fr.iterations << "\n";
            md << "- final cost：" << fr.finalCost << "\n\n";

            md << "## 输出文件\n\n";
            md << "- `result/task2_fit/tracking_overlay.mp4`：目标跟踪叠加视频。\n";
            md << "- `result/task2_fit/fit_comparison.png`：展开角度观测与拟合。\n";
            md << "- `result/task2_fit/residuals.png`：角度残差。\n";
            md << "- `result/task2_fit/angular_velocity.png`：估计角速度与 SavGol 观测。\n";
            md << "- `result/task2_fit_result.md`：本文件。\n";
        }

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "A=" << A << ", b=" << b
                  << ", Omega=" << Omega << ", phi=" << phi
                  << ", theta0=" << theta0 << std::endl;
        std::cout << "角度 RMSE=" << rmseTheta << " rad, "
                  << "角速度验证 RMSE=" << rmseOmega << " rad/s" << std::endl;
        std::cout << "输出目录: " << OUT_DIR << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}