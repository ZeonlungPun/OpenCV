#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <opencv2\opencv.hpp>
#include<iostream>
#include <algorithm>
#include <time.h>
using namespace std;
using namespace cv;



//������Ӧ
void key_demo(Mat& image)
{
	Mat dst=Mat::zeros(image.size(),image.type());
	while (true)
	{
		int c = waitKey(100);
		if (c == 27)//esc
		{
			break;
		}
		if (c == 49)// key#1
		{
			cout << "#1" << endl;
			//ת�ɻҶ�ͼ
			cvtColor(image,dst,COLOR_BGR2GRAY);

		}
		if (c == 50)// key#2
		{
			cout << "#2" << endl;
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		if (c == 51)// key#3
		{
			cout << "#3" << endl;
			dst = Scalar(50, 50, 50);
			add(image, dst, dst);
		}
		imshow("key reponse", dst);
	}
}

void colour_style_demo(Mat& image)
{
	int colormap[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_JET,
		COLORMAP_WINTER,
		COLORMAP_RAINBOW,
		COLORMAP_OCEAN,
		COLORMAP_SUMMER,
		COLORMAP_SPRING,
		COLORMAP_COOL,
		COLORMAP_PINK,
		COLORMAP_HOT,
		COLORMAP_PARULA,
		COLORMAP_MAGMA,
		COLORMAP_INFERNO,
		COLORMAP_PLASMA,
		COLORMAP_VIRIDIS,
		COLORMAP_CIVIDIS,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED,
	};
	Mat dst;
	int index = 0;
	while (true)
	{
		int c = waitKey(2000);
		if (c == 27)//esc
		{
			break;
		}
		applyColorMap(image, dst, colormap[index%19]);
		index++;
		imshow("colour style", dst);
	}
}


//����ʶ�����
void car_detect()
{
	int effective_area = 250;
	int num = 0;
	int line_base = 500;
	int offset = 10;
	VideoCapture capture("E:\\testcar.mp4");
	//ȥ���������㷨������MOGָ��
	Ptr<BackgroundSubtractorMOG2>MOG=createBackgroundSubtractorMOG2();
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat frame,dst,blur,mask;
	while (true) {
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		//ת�ɻҶ�ͼ
		cvtColor(frame, dst, COLOR_BGR2GRAY);
		//��˹ȥ��
		GaussianBlur(dst,blur,Size(3,3),5);
		//ȥ������
		(*MOG).apply(blur,mask);
		erode(mask, mask, kernel);
		dilate(mask, mask, kernel);
		morphologyEx(mask, mask, MORPH_CLOSE, kernel);
		//��׼��
		Point p1(0,line_base), p2(1400, line_base);
		line(frame, p1, p2, (255, 255, 0), 3);
		
		//������ȡ�����
		vector<vector<Point>>contours;
		vector<vector<Point>>EffectiveContours;
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);

			if (area > effective_area)
			{
				EffectiveContours.push_back(contours[i]);
			}
		}

		//չʾЧ��
		char text[10];
		for (int i = 0; i < EffectiveContours.size(); i++)
		{
			//��ȡ��Ӿ���
			RotatedRect rect = minAreaRect(EffectiveContours[i]);
			Rect box = rect.boundingRect();
			rectangle(frame, Rect(box.x, box.y, box.width, box.height), Scalar(0, 0, 255), 2);
			if (box.y > line_base - offset && box.y < line_base + offset)
			{
				num += 1;
				cout << num << endl;
			}

			sprintf(text, "%s%d", "count:", num);
			putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
		}



		imshow("frame", frame);
		int key = waitKey(10);
		if (key == 27)
		{
			break;
		}
	}



}



void car_detect2()
{
	
	set<int>carset;
	int num = 0;
	int effective_area = 300;

	VideoCapture capture("E:\\YOLOV5+DeepSORT\\yolov5-deepsort\\video\\test_traffic.mp4");

	//ȥ���������㷨������MOGָ��
	Ptr<BackgroundSubtractorMOG2>MOG = createBackgroundSubtractorMOG2();
	
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat frame, dst, blur, mask;
	while (true) {
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		//ת�ɻҶ�ͼ
		cvtColor(frame, dst, COLOR_BGR2GRAY);
		//��˹ȥ��
		GaussianBlur(dst, blur, Size(3, 3), 5);
		//ȥ������
		(*MOG).apply(blur, mask);
		erode(mask, mask, kernel);
		dilate(mask, mask, kernel);
		morphologyEx(mask, mask, MORPH_CLOSE, kernel);
		//������������
		Point p1(500, 500);
		Point p2(300, 505);
		Point p3(1000, 505);
		Point p4(1100, 500);
		vector<Point> pts;
		pts.push_back(p1);
		pts.push_back(p2);
		pts.push_back(p3);
		pts.push_back(p4);

		polylines(frame, pts, true, Scalar(255, 0, 0), 3, 8, 0);
		

		//������ȡ�����
		vector<vector<Point>>contours;
		vector<vector<Point>>EffectiveContours;
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);

			if (area > effective_area)
			{
				EffectiveContours.push_back(contours[i]);
			}
		}

		//չʾЧ��
		char text[10];
		for (int i = 0; i < EffectiveContours.size(); i++)
		{
			//��ȡ��Ӿ���
			RotatedRect rect = minAreaRect(EffectiveContours[i]);
			Rect box = rect.boundingRect();
			rectangle(frame, Rect(box.x, box.y, box.width, box.height), Scalar(0, 0, 255), 2);
			//�������ĵ�
			float cx = box.x + box.width / 2;
			float cy = box.y + box.height / 2;
			circle(frame, Point(cx, cy), 5, (0, 0, 255), -1);

			//������ĵ��Ƿ�����������
			float result=pointPolygonTest(pts, Point(cx, cy), false);
			if (result >= 0)
			{
				num += 1;
				
			}
			carset.insert(num);


			sprintf(text, "%s%d", "count:", carset.size());
			putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
		}



		imshow("frame", frame);
		int key = waitKey(1);
		if (key == 27)
		{
			break;
		}
	}



}




//֡��ֽ�������ʶ��
void people_detect2()
{
	int effective_area = 100;
	int num = 0;
	int line_base = 700;
	int offset = 10;  
	VideoCapture capture("E:\\people.mp4");

	Mat  curFrame, nextFrame,preFrame,frame,diff1,diff2,diff3,temp;
	while (true)
	{
		capture >> frame;
		preFrame = frame.clone();//��һ֡

		capture >> frame;
		curFrame = frame.clone();//�ڶ�֡

		capture >> frame;
		nextFrame = frame.clone();//����֡

		cvtColor(preFrame, preFrame, COLOR_BGR2GRAY);
		cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
		cvtColor(nextFrame, nextFrame, COLOR_BGR2GRAY);
		absdiff(curFrame, preFrame, diff1);
		absdiff(preFrame, nextFrame, diff2);
		bitwise_and(diff1, diff2, diff3);
		threshold(diff3, temp, 20, 255, THRESH_BINARY);
		dilate(temp, temp, Mat());
		erode(temp, temp, Mat());

		//��׼��
		Point p1(line_base, 0), p2(line_base, 1272);
		line(frame, p1, p2, (255, 255, 0), 3);
		vector<vector<Point>>contours;
		vector<vector<Point>>EffectiveContours;
		findContours(temp, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//��ȡ��Ч����
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			RotatedRect rect = minAreaRect(contours[i]);
			Rect box = rect.boundingRect();

			if (area > effective_area)
			{
				EffectiveContours.push_back(contours[i]);
			}

		}
		char text[10];
		for (int j = 0; j < EffectiveContours.size(); j++)
		{
			//��ȡ��Ӿ���

			RotatedRect rect = minAreaRect(EffectiveContours[j]);
			Rect box = rect.boundingRect();
			rectangle(frame, Rect(box.x, box.y, box.width, box.height), Scalar(0, 0, 255), 2);
			if (box.x > line_base - offset && box.x < line_base + offset)
			{
				num += 1;
				cout << num << endl;
			}
			sprintf(text, "%s%d", "count:", num);
			putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

		}


		imshow("detect", frame);
		int key = waitKey(50);
		if (key == 27)
		{
			break;
		}

	}
	capture.release();

}

//������ͼ:��ͼƬ��ֳ�һ��Ļ���ѧϰ������(w,h.c)--> (w*h,c) 
Mat mat_to_samples(Mat& image)
{
	int w = image.cols;
	int h = image.rows;
	int samplenum = w * h;
	int dims = image.channels();
	Mat points(samplenum, dims, CV_32F, Scalar(10));
	int index = 0;
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			index = row * w + col;
			//��ȡ��ͼƬ�ڸ�λ�ö�Ӧ������ֵ
			Vec3b bgr = image.at<Vec3b>(row, col);
			//static_cast< ��Ҫ������> (ԭ����)
			points.at<float>(index, 0) = static_cast<int>(bgr[0]);
			points.at<float>(index, 1) = static_cast<int>(bgr[1]);
			points.at<float>(index, 2) = static_cast<int>(bgr[2]);
		}
	}
	return points;
}

int bgr_replace()
{
	Mat src = imread("E:\\opencv\\shan.jpg");
	namedWindow("input", WINDOW_AUTOSIZE);
	//imshow("input", src);

	//��ȡ������
	Mat points = mat_to_samples(src);
	//����kmeans
	int numCluster = 4;
	//���������
	Mat labels;
	Mat centers;

	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(points, numCluster, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

	//����mask
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	int index = src.rows * 2 + 2;
	//�ҵ������ı��
	int cindex = labels.at<int>(index, 0);
	int height = src.rows;
	int width = src.cols;
	Mat dst;
	src.copyTo(dst);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;
			int label = labels.at<int>(index, 0);
			if (label == cindex)//����
			{
				dst.at<Vec3b>(row, col)[0] = 0;
				dst.at<Vec3b>(row, col)[1] = 0;
				dst.at<Vec3b>(row, col)[2] = 0;
				mask.at<uchar>(row, col) = 0;
			}
			else
			{
				mask.at<uchar>(row, col) = 255;
			}
		}
	}
	imshow("mask", mask);
	imshow("kmenas:", dst);
	//��ʴ+��˹ģ��
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	erode(mask, mask, kernel);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);
	imshow("blur", mask);

	//ͨ�����
	Vec3b color;
	//����������ɫ
	/*color[0] = 255;
	color[1] = 0;
	color[2] = 0;*/
	Mat bgr_pic = imread("E://opencv//snow.jpg");
	resize(bgr_pic, bgr_pic, src.size());
	
	Mat result(src.size(), src.type());

	double w = 0.0;
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int m = mask.at<uchar>(row, col);
			if (m == 255)
			{
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col);//ǰ����ԭ�ⲻ���հ�ԭͼ
			}
			else if (m == 0)
			{
				//result.at<Vec3b>(row, col) = color;//����
				result.at<Vec3b>(row, col) =bgr_pic.at<Vec3b>(row, col);
			}
			//else
			//{
			//	//ԭͼ���±�����Ȩ���
			//	w =2.5* m / 255.0;
			//	b1 = src.at<Vec3b>(row, col)[0];
			//	g1 = src.at<Vec3b>(row, col)[1];
			//	r1 = src.at<Vec3b>(row, col)[2];

			//	/*b2 = color[0];
			//	g2 = color[1];
			//	r2 = color[2];*/
			//	b2= bgr_pic.at<Vec3b>(row, col)[0];
			//	g2= bgr_pic.at<Vec3b>(row, col)[1];
			//	r2= bgr_pic.at<Vec3b>(row, col)[2];


			//	b = b1 * w + b2 * (1.0 - w);
			//	g = g1 * w + g2 * (1.0 - w);
			//	r = r1 * w + r2 * (1.0 - w);

			//	result.at<Vec3b>(row, col)[0] = b;
			//	result.at<Vec3b>(row, col)[1] = g;
			//	result.at<Vec3b>(row, col)[2] = r;

			//}
		}
	}
	imshow("�����滻", result);

	waitKey(0);

	return 0;
}


//������ͼ
// �������
struct Inputparama {
	int thresh = 30;                               // ����ʶ����ֵ����ֵԽС����ʶ��Ǳ��������Խ�����к��ʷ�Χ��ĿǰΪ5-60
	int transparency = 255;                        // �����滻ɫ͸���ȣ�255Ϊʵ��0Ϊ͸��
	int size = 7;                                  // �Ǳ�������Ե�黯��������ֵԽ�����Ե�黯�̶�Խ����
	Point p = Point(0, 0);                 // ����ɫ�����㣬��ͨ���˻�������ȡ��Ҳ����Ĭ��(0,0)����ɫ��Ϊ����ɫ
	Scalar color = Scalar(0, 0, 255);  // ����ɫ
};

// �����ֵ������
int geiDiff(uchar b, uchar g, uchar r, uchar tb, uchar tg, uchar tr)
{
	return  int(sqrt(((b - tb) * (b - tb) + (g - tg) * (g - tg) + (r - tr) * (r - tr)) / 3));
}

// ��������
Mat BackgroundSeparation(Mat src, Inputparama input)
{
	Mat bgra, mask;
	// ת��ΪBGRA��ʽ����͸���ȣ�4ͨ��
	cvtColor(src, bgra, COLOR_BGR2BGRA);
	mask = Mat::zeros(bgra.size(), CV_8UC1);
	int row = src.rows;
	int col = src.cols;

	// �쳣��ֵ����
	input.p.x = max(0, min(col, input.p.x));
	input.p.y = max(0, min(row, input.p.y));
	input.thresh = max(5, min(100, input.thresh));
	input.transparency = max(0, min(255, input.transparency));
	input.size = max(0, min(30, input.size));

	// ȷ������ɫ
	uchar ref_b = src.at<Vec3b>(input.p.y, input.p.x)[0];
	uchar ref_g = src.at<Vec3b>(input.p.y, input.p.x)[1];
	uchar ref_r = src.at<Vec3b>(input.p.y, input.p.x)[2];

	// �����ɰ�������Ĥ��
	for (int i = 0; i < row; ++i)
	{
		uchar* m = mask.ptr<uchar>(i);
		uchar* b = src.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			if ((geiDiff(b[3 * j], b[3 * j + 1], b[3 * j + 2], ref_b, ref_g, ref_r)) > input.thresh)
			{
				m[j] = 255;
			}
		}
	}

	// Ѱ����������������������ںڶ�
	vector<vector<Point>> contour;
	vector<Vec4i> hierarchy;
	// RETR_TREE����״�ṹ��ȡ����������CHAIN_APPROX_NONE��ȡ������ÿ������
	findContours(mask, contour, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(mask, contour, -1, Scalar(255), FILLED, 4);

	// ������
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(mask, mask, MORPH_CLOSE, element);

	// ��Ĥ�˲�����Ϊ�˱�Ե�黯
	blur(mask, mask, Size(2 * input.size + 1, 2 * input.size + 1));

	// ��ɫ
	for (int i = 0; i < row; ++i)
	{
		uchar* r = bgra.ptr<uchar>(i);
		uchar* m = mask.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			// �ɰ�Ϊ0��������Ǳ�׼������
			if (m[j] == 0)
			{
				r[4 * j] = uchar(input.color[0]);
				r[4 * j + 1] = uchar(input.color[1]);
				r[4 * j + 2] = uchar(input.color[2]);
				r[4 * j + 3] = uchar(input.transparency);
			}
			// ��Ϊ0�Ҳ�Ϊ255���������������򣨱�Ե��������Ҫ�黯����
			else if (m[j] != 255)
			{
				// ��Ե����������ɫ
				int newb = (r[4 * j] * m[j] * 0.3 + input.color[0] * (255 - m[j]) * 0.7) / ((255 - m[j]) * 0.7 + m[j] * 0.3);
				int newg = (r[4 * j + 1] * m[j] * 0.3 + input.color[1] * (255 - m[j]) * 0.7) / ((255 - m[j]) * 0.7 + m[j] * 0.3);
				int newr = (r[4 * j + 2] * m[j] * 0.3 + input.color[2] * (255 - m[j]) * 0.7) / ((255 - m[j]) * 0.7 + m[j] * 0.3);
				int newt = (r[4 * j + 3] * m[j] * 0.3 + input.transparency * (255 - m[j]) * 0.7) / ((255 - m[j]) * 0.7 + m[j] * 0.3);
				newb = max(0, min(255, newb));
				newg = max(0, min(255, newg));
				newr = max(0, min(255, newr));
				newt = max(0, min(255, newt));
				r[4 * j] = newb;
				r[4 * j + 1] = newg;
				r[4 * j + 2] = newr;
				r[4 * j + 3] = newt;
			}
		}
	}
	return bgra;
}

int seperate_background()
{
	Mat src = imread("E:\\opencv\\shan3.jpg");
	Inputparama input;
	input.thresh = 55;
	input.transparency = 255;
	input.size = 6;
	input.color = Scalar(255, 255, 255);

	clock_t s, e;
	s = clock();
	Mat result = BackgroundSeparation(src, input);
	e = clock();
	double dif = e - s;
	cout << "time:" << dif << endl;

	imshow("original", src);
	imshow("result", result);
	imwrite("result1.png", result);
	waitKey(0);
	return 0;
}

//����HSV�ռ任�������ҳ�����
//��Ҫ�ǻ���ͼ��ɫ�ʿռ䣬ת��ΪHSVɫ�ʿռ�ʵ��mask����ȡ��Ȼ��ͨ��һЩ�򵥵�ͼ���������̬ѧ���ղ�������˹ģ���ȵ�������mask����
//����mask��������Ȩ��ϵ��������Ļͼ���뱳��ͼ�������ں�����һ�������ͼ����ɿ�ͼ��

//�ںϺ��滻
Mat background_01;
Mat background_02;
Mat replace_and_blend(Mat& src, Mat& mask)
{
	Mat result = Mat::zeros(src.size(), src.type());
	int h = src.rows;
	int w = src.cols;
	int dims = src.channels();
	//replace and blend
	int m = 0;//��¼ͼ���ÿ������ֵ
	double wt = 0;//�ں�Ȩ��
	int r = 0, g = 0, b = 0;
	int r1 = 0, g1 = 0, b1 = 0;
	int r2 = 0, g2 = 0, b2 = 0;

	for (int row = 0; row < h; row++)
	{
		uchar* current = src.ptr<uchar>(row);
		uchar* bgr=background_01.ptr<uchar>(row);
		uchar* maskrow=mask.ptr<uchar>(row);
		uchar* targetrow=result.ptr<uchar>(row);
		for (int col = 0; col < w; col++)
		{
			//�Ƚ�������ָ����һ���洢�� �ҳ�����ֵ
			m = *maskrow++;
			if (m == 255)//����:���滻��
			{
				*targetrow++ = *bgr++;
				*targetrow++ = *bgr++;
				*targetrow++ = *bgr++;
				current += 3;
			}
			else if (m == 0) {//ǰ��������
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				bgr += 3;
			}
			else {
				// ��������
				b1 = *bgr++;
				g1 = *bgr++;
				r1 = *bgr++;

				// Ŀ��ǰ������
				b2 = *current++;
				g2 = *current++;
				r2 = *current++;

				// ���Ȩ��
				wt = m / 255.0;

				// ���
				b = b1 * wt + b2 * (1.0 - wt);
				g = g1 * wt + g2 * (1.0 - wt);
				r = r1 * wt + r2 * (1.0 - wt);

				*targetrow++ = b;
				*targetrow++ = g;
				*targetrow++ = r;
			}
			
		}
	}
	return result;

}


int replace_hsv()
{
	Mat src = imread("E:\\opencv\\shan3.jpg");
	background_01 = imread("E:\\opencv\\snow.jpg");
	resize(background_01, background_01, Size(src.cols, src.rows));
	Mat hsv,mask;
	cvtColor(src, hsv, COLOR_BGR2HSV);
	//�ҵ���ɫ����λ��  mask:����Ϊ0 ��  ǰ��Ϊ1 ��
	inRange(hsv, Scalar(0, 43, 46), Scalar(10, 255, 255), mask);
	//ȡ�� mask:����Ϊ1 ��  ǰ��Ϊ0 ��
	//bitwise_not(mask, mask);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);

	Mat result = replace_and_blend(src, mask);
	imshow("result", result);
	waitKey();
	
	
	



	return 0;

	


}







int main()
{
	seperate_background();
	return 0;
}























int main11()
{
	//����ͼƬ
	string imagename = "E:\\yolov5-tf2-main\\yolov5-tf2-main\\img\\street.jpg";
	Mat img = cv::imread(imagename);
	//�������ͼ��ʧ��
	if (img.empty())
	{
		std::cout << "miss the image file : " + imagename << std::endl;
		return -1;
	}
	//��������
	//namedWindow("image", 1);
	//��ʾͼ��
	imshow("image", img);

	//namedWindow("image1", 1);
	
	//��ʾͼ��
	//imshow("image1", output);
	//imwrite("E://stree.jpg", output);

	//�����հ�ͼ��
	//Mat m1 = Mat::zeros(Size(8, 8), CV_8UC3);
	//cout << m1 << endl;
	//cout << "width:" << m1.cols << "height:" << m1.rows << "channenl:" << m1.channels() << endl;
	//Mat m2= Mat::zeros(Size(8, 8), CV_8UC3);
	//m2 = Scalar(127, 127, 127);
	//imshow("fake", m2);


	//�ȴ����������������������
	//waitKey();
	//key_demo(img);
	colour_style_demo(img);
	


	return 0;
}
