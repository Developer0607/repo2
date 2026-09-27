#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void writefile(string path, Mat* src){
    Mat img = *src;
    cout<<"Writing image to "<<path<<endl;
    if(img.empty()){
        cerr<<"Image is empty, cannot write to "<<path<<endl;
        return;
    }
    if(img.type() != CV_8UC1 && img.type() != CV_8UC3){
        cerr<<"Image type is not supported for writing to "<<path<<endl;
        return;
    }
    if(!img.isContinuous()){
        cerr<<"Image is not continuous, making it continuous for writing to "<<path<<endl;
        return;
    }
    if(imwrite(path,img)){
        cout<<path<<" - done.";
    }else{
        cerr<<"Writing " << path << " failed!\n";
    }
    return;
}
void testimg(Mat gray){
    cout<<"rows:"<<gray.rows<<", cols:"<<gray.cols<<endl;
    cout<<"channels:"<<gray.channels()<<", type:"<<gray.type()<<endl;
    cout<<"depth:"<<gray.depth()<<endl;
}

int main(){
    Mat img = imread("example.jpg");
    if(img.empty())
    {
        cerr << "Cannot read image!" << endl;
        return 1;
    }
    cout << "Image loading successful.\n";
    imshow("Original",img);
    
    Mat test(200,200,CV_8UC3,Scalar(0,0,255));
    //Mat img = imread("test.png");
    if(imwrite("test1.png",test)){
        cout<<"test.png - done.\n";
    }else{
        cerr<<"Writing test.png failed!\n";
    }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imshow("Gray",gray);
    cout<<"rows:"<<gray.rows<<", cols:"<<gray.cols<<endl;
    cout<<"channels:"<<gray.channels()<<", type:"<<gray.type()<<endl;
    cout<<"depth:"<<gray.depth()<<endl;
    
    writefile("gray.png", &gray);

    Mat meanImg, gaussianImg, medianImg;
    blur(img, meanImg, Size(5,5));
    GaussianBlur(img, gaussianImg, Size(5,5), 1.5);
    medianBlur(img, medianImg, 5);

    imshow("Mean", meanImg);
    imshow("Gaussian", gaussianImg);
    imshow("Median", medianImg);
    writefile("mean_filter.png", &meanImg);
    writefile("gaussian_filter.png", &gaussianImg);
    writefile("median_filter.png", &medianImg);

    

    Mat  binary, adaptive;
    threshold(gray, binary, 128,255, THRESH_BINARY);
    adaptiveThreshold(gray, adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    imshow("Binary", binary);
    imshow("Binary", adaptive);

    Mat smooth, edges;
    GaussianBlur(img, smooth, Size(5,5), 1.5);
    Canny(smooth, edges, 100, 200);
    imshow("Canny", edges);

    Mat hsv, maskLow, maskHigh, mask;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), maskLow);
    inRange(hsv, Scalar(170, 100, 100), Scalar(179, 255, 255), maskHigh);
    bitwise_or(maskLow, maskHigh, mask);
    imshow("Red Mask", mask);
    writefile("red_mask.png", &mask);
    writefile("hsv.png", &hsv);

    Mat equalized;
    equalizeHist(gray, equalized);
    imshow("Equalized", equalized);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat dilated, eroded, opened, closed;
    dilate(binary, dilated, kernel);
    erode(binary, eroded, kernel);
    morphologyEx(binary, opened, MORPH_OPEN, kernel);
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);
    imshow("Open", opened);
    imshow("Close", closed);
    writefile("dilate.png", &dilated);
    writefile("erode.png", &eroded);
    writefile("open.png", &opened);
    writefile("close.png", &closed);

    Mat gradX, gradY, absX, absY, grad;
    Sobel(gray, gradX, CV_16S, 1, 0);
    Sobel(gray, gradY, CV_16S, 0, 1);
    convertScaleAbs(gradX, absX);
    convertScaleAbs(gradY, absY);
    addWeighted(absX, 0.5, absY, 0.5, 0, grad);
    imshow("Sobel", grad);

    Mat channels[3],colorEdges;
    split(hsv, channels);
    Canny(channels[0], colorEdges, 100, 200);
    imshow("Hue Edges", colorEdges);

    Mat morph;
    Canny(gray, edges, 100, 200);
    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, morph, MORPH_CLOSE, kernel);
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(morph,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    Mat ContourImg = Mat::zeros(img.size(),CV_8UC3);
    drawContours(ContourImg, contours, -1, Scalar(0,255,0),2);
    imshow("Contours",ContourImg);

    Mat result = img.clone();
    for(size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if(area < 500.0)continue;
        Rect box = boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width)/box.height;
        if(ratio < 0.2 || ratio > 5.0)continue;

        rectangle(result, box, Scalar(0,0,255),2);
        drawContours(result,contours,static_cast<int>(i),Scalar(0,255,0),2);
    }
    imshow("filtered contours", result);
    writefile("contours_boxes.png", &ContourImg);

    waitKey(0);
    return 0;
}