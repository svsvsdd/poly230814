#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;
using namespace std;


//void on_threshold(int pos, void* userdata);
void on_trackbar(int pos, void* userdata);


void erode_dilate()
{
	Mat src = imread("milkdrop.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat dst1, dst2;
	erode(bin, dst1, Mat());
	dilate(bin, dst2, Mat());

	imshow("src", src);
	imshow("bin", bin);
	imshow("erode", dst1);
	imshow("dilate", dst2);

}



void open_close()
{
	Mat src = imread("milkdrop.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat dst1, dst2 , dst3;
	morphologyEx(bin, dst1, MORPH_OPEN, Mat());
	morphologyEx(bin, dst2, MORPH_CLOSE, Mat());

	morphologyEx(dst1, dst3, MORPH_CLOSE, Mat());

	imshow("src", src);
	imshow("bin", bin);
	imshow("open", dst1);
	imshow("close", dst2);

	imshow("open-close", dst3);

}



void labeling_basic()
{
	uchar data[] = {
		0,0,1,1,0,0,0,0,
		1,1,1,1,0,0,1,0,
		1,1,1,1,0,0,0,0,
		0,0,0,0,0,1,1,0,
		0,0,0,1,1,1,1,0,
		0,0,0,1,0,0,1,0,
		0,0,1,1,1,1,1,0,
		0,0,0,0,0,0,0,0 };

	Mat src = Mat(8, 8, CV_8UC1, data) * 255;

	Mat labels;
	int cnt = connectedComponents(src, labels);

	cout << "src: \n" << src << endl;
	cout << "labels: \n" << labels << endl;
	cout << "number of labels: \n" << cnt << endl;
}

void labeling_stats()
{
	Mat src = imread("keyboard.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat labels, stats, centroids;
	int cnt= connectedComponentsWithStats(bin, labels, stats, centroids);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int i = 1; i < cnt; i++) {
		int* p = stats.ptr<int>(i);

		if (p[4] < 20) 
			rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 0, 255), 2);
		else 
			rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255), 2);
		//puttext  putText(dst, "FONT_HERSHEY_COMPLEX | FONT_ITALIC", Point(20, 450), FONT_HERSHEY_COMPLEX | FONT_ITALIC, 1, Scalar(255, 0, 0));
	}
	

	imshow("src", src);
	imshow("dst", dst);
}



void contours_basic()
{
	Mat src = imread("contours.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	vector<vector<Point>> contours;
	findContours(src, contours, RETR_LIST, CHAIN_APPROX_NONE);

	Mat dst;
	cvtColor(src, dst, COLOR_BGRA2BGR);

	for (int i = 0; i < contours.size(); i++) {
		Scalar c(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, i, c, 2);
	}

	imshow("src", src);
	imshow("dst", dst);
}


void contours_hier()
{
	Mat src = imread("contours.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	Mat dst;
	cvtColor(src, dst, COLOR_BGRA2BGR);

	for (int idx = 0; idx >= 0; idx=hierarchy[idx][0]) {
		Scalar c(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, idx, c, -1, LINE_8, hierarchy);
	}

	imshow("src", src);
	imshow("dst", dst);
}





void beta_rect_circ()
{
	Mat src = imread("beta2.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	vector<vector<Point>> contours;
	
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat dst, dst2, dst3;
	cvtColor(src, dst, COLOR_BGRA2BGR);
	//cvtColor(src, dst2, COLOR_BGRA2BGR);
	//cvtColor(src, dst3, COLOR_BGRA2BGR);

	Rect boundingRectBox;
	RotatedRect rotBox;
	

	for (int i = 0; i < contours.size(); i++) {
		boundingRectBox = boundingRect(contours[i]);
		rotBox = minAreaRect(contours[i]);
	}

	Point2f vertices[4];
	rotBox.points(vertices);

	rectangle(dst, boundingRectBox, Scalar(0, 255, 255), 2);

	for (int i = 0; i < 4; i++) {
		line(dst, vertices[i], vertices[(i + 1)%4], Scalar(0, 0, 255), 2);	//rot rect는 대각선으로 불가능
	}

	




	Point2f center;
	float radius = 0;
	minEnclosingCircle(contours[0], center, radius);
	circle(dst, center, cvRound(radius), Scalar(255, 0, 0), 1, LINE_AA);



	/*rotatedRect.points(vertices);

	
	Point center;
	float radius;*/



	//for (int i = 0; i < contours.size(); i++) {
	//	RotatedRect rotatedRect = cv::minAreaRect(contours[i]);
	//	Point2f vertices[4];
	//	rotatedRect.points(vertices);
	//	for (int j = 0; j < 4; j++) {
	//		line(dst2, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 2);
	//	}
	//}




	//for (int i = 0; i < contours.size(); i++) {
	//	Scalar c(rand() & 255, rand() & 255, rand() & 255);
	//	drawContours(dst, contours, i, c, 2);

	//	boundingRect(contours);
	//	minAreaRect(contours);
	//	//minEnclosingCircle(contours, Point2f &center, radius);

	//	rectangle(dst, Rect(), Scalar(0, 255, 255), 2);
	//	rectangle(dst, Rect(), Scalar(0, 0, 255), 2);
	//	//circle(dst, Point2f, radius, Scalar(255, 0, 0), 2);

	//}
	

	imshow("src", src);
	imshow("dst", dst);
	//imshow("dst2", dst2);
	//imshow("dst3", dst3);
}


void setLabel(Mat& img, const vector<Point>& pts, const String& label)
{
	Rect rc = boundingRect(pts);
	rectangle(img, rc, Scalar(0, 0, 255), 1);
	putText(img, label, rc.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
}



int main(int argc, char* argv[])
{
	Mat img = imread("polygon.bmp", IMREAD_COLOR);


	if (img.empty()) {
		cerr << "img load fail" << endl;
		return -1;
	}


	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat bin;
	threshold(gray, bin, 200, 255, THRESH_BINARY_INV | THRESH_OTSU);

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	for (vector<Point> pts : contours) {
		if (contourArea(pts) < 400)
			continue;

		vector<Point> approx;
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);

		int vtc = (int)approx.size();

		if (vtc == 3) {
			setLabel(img, pts, "TRI");
		}
		else if (vtc == 4) {
			setLabel(img, pts, "RECT");
		}
		else if (vtc > 4) {
			double len = arcLength(pts, true);
			double area = contourArea(pts);
			double ratio = 4. * CV_PI * area / (len * len);
			
			if (ratio > 0.8) {
				setLabel(img, pts, "CIR");
			}
			else {
				setLabel(img, pts, "ETC");
			}
		}
	}

	imshow("img", img);

	

	waitKey();
	destroyAllWindows();
	return 0;
}







//int main(int argc, char* argv[])	//파일명 뒤에 argv 있으면 
//{
//	Mat src;
//
//	if (argc < 2)
//		src = imread("neutrophils.png", IMREAD_GRAYSCALE);
//		//src = imread("neutrophils.png", IMREAD_GRAYSCALE);
//	else
//		src = imread(argv[1], IMREAD_GRAYSCALE);	//
//
//
//	if (src.empty()) {
//		cerr << "img load fail" << endl;
//		return -1;
//	}
//
//	imshow("src", src);
//
//	namedWindow("dst");
//	createTrackbar("Threshold", "dst", 0, 255, on_threshold, (void*)&src);
//	setTrackbarPos("Threshold", "dst", 128);
//
//
//	waitKey();
//	destroyAllWindows();
//	return 0;
//}

//int main()
//{
//	/*Mat src = imread("sudoku.jpg", IMREAD_GRAYSCALE);
//
//
//	if (src.empty()) {
//		cerr << "img load fail" << endl;
//		return -1;
//	}
//
//	imshow("src", src);
//
//	namedWindow("dst");
//	createTrackbar("Block size", "dst", 0, 200, on_trackbar, (void*)&src);
//	setTrackbarPos("Block size", "dst", 11);*/
//
//
//	//erode_dilate();
//	//open_close();
//	//labeling_basic();
//	//labeling_stats();
//	//contours_basic();
//	//contours_hier();
//	beta_rect_circ();
//	
//
//	waitKey();
//	destroyAllWindows();
//	return 0;
//}


//void on_threshold(int pos, void* userdata)
//{
//	Mat src = *(Mat*)userdata;
//	Mat dst;
//	threshold(src, dst, pos, 255, THRESH_BINARY);
//
//	imshow("dst", dst);
//}



//void on_trackbar(int pos, void* userdata)
//{
//	Mat src = *(Mat*)userdata;
//	
//	int bsize = pos;
//	if (bsize % 2 == 0) bsize--;
//	if (bsize < 3) bsize = 3;
//
//	Mat dst;
//	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bsize, 5);
//
//	imshow("dst", dst);
//}