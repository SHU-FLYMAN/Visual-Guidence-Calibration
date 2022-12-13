/* 相机驱动包装器，与渲染器交互 */

#pragma once
#include <iostream>
#include <direct.h>
#include <QThread>
#include <QMutex>
#include <QString>
#include <QDir>
#include <opencv2/opencv.hpp>
#include "Camera_Flycapture.h"
#include "Calibrator.h"



using namespace std;
using namespace cv;


class CameraEngineer:public QThread
{
	Q_OBJECT
public:
	Calibrator *calibrator;  // 检测图像角点
	bool stop_flag = false;  // 记录是否停止（在外部修改）
	bool is_detect = false;  // 是否实时检测
	bool is_guide = false;   // 是否姿态引导
	vector<cv::Point2f> pts2d_nextpose;

protected:
	int img_count = 0;       // 拍照计数
	QMutex mutex1, mutex2;   // 锁
	Camera_Flycapture *cam;  // 相机对象
	Mat img, img_save;                 // 用于保存图像
	// 单间模式：有且仅有一个实例存在
	CameraEngineer(Camera_Flycapture &cam) {
		this->cam = &cam;
		// 启动线程，运行run
		start();
	}; // 保护，外面不可以生成对象了

public:
	static CameraEngineer *Get(Camera_Flycapture &cam) {
		static CameraEngineer imgThread(cam);
		return &imgThread;
	}

	// 实时检测运行
	void run() {
		int wait_time = 20;
		// 主程序
		while (true)
		{
			// 如果相机没有连接，那么等100ms，再次尝试
			// stop_flag 用来判断整个程序是否停止
			if (stop_flag)
			{
				return;
			}
			if (!cam->is_connect)
			{
				msleep(100);
				continue;
			}
			// 如果is_connect连接上来了，但是还是要退出
			mutex1.lock();
			if (stop_flag)
			{
				return;
			}
			// 拍照
			cam->capture(img);
				
			// 检测角点，显示在图像中
			
			if (is_detect)
			{
				// 会改变图像大小
				this->calibrator->detectCornersFast(img, 0.3);
			}
			// 是否显示引导角点
			if (is_guide)
			{
				this->calibrator->show_nextpose(img, pts2d_nextpose);
			}

			// 计算距离
			if (is_detect && is_guide) {
				// 输出两者的距离，并且显示在图像中
				this->calibrator->calc_distance(img, pts2d_nextpose);
			}
			
			// 发送图像
			img_signal(img);
			mutex1.unlock();
			msleep(wait_time);
		}
	};

	// 拍照并保存图片（需要+锁，避免拍到一半没写入）
	string captureSaveMutex(const string &folder="./out", const string &img_end=".bmp") {
		// 新建文件夹
		mkdir(folder);
		mutex2.lock();
		cam->capture(img_save);
		img_count += 1;
		string filename = folder + "/" + to_string(img_count) + img_end;
		cv::imwrite(filename, img_save);
		cout << "save image into:" << filename << endl;
		mutex2.unlock();
		return filename;
	};

	void disconnect() {
		mutex1.lock();
		cam->disconnect();
		mutex1.unlock();
	}

	void setCalibrator(Calibrator &calibrator) {
		this->calibrator = &calibrator;
	}

	void start_detect() {
		mutex2.lock();
		is_detect = true;
		mutex2.unlock();
	}

	void stop_detect() {
		mutex2.lock();
		is_detect = false;
		mutex2.unlock();
	}

	void setImageCountZeros() {
		img_count = 0;
	}

	void mkdir(const string &folder) {
		QString folder_qstr = QString::fromStdString(folder);
		QDir dir;
		// 如果文件夹不存在
		if (!dir.exists(folder_qstr))
		{
			dir.mkpath(folder_qstr);
		}
	}


	void load_nextpose(const string &filename) {
		is_guide = this->calibrator->load_nextpose(filename, pts2d_nextpose);
	}


signals:
	// 信号：用于发送图片
	void img_signal(Mat img);
};

