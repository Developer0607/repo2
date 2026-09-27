#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main(){
    cout<< getBuildInformation() << endl;
    Mat test(200,200,CV_8UC3,Scalar(0,0,255));
    Mat img = imread("test.png");
    if(imwrite("test1.png",img)){
        cout<<"test.png - done.\n";
    }else{
        cerr<<"Writing test.png failed!\n";
    }
    return 0;
}