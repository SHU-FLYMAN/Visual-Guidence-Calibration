/* 灰点相机硬件控制类 */

#pragma once
#include <iostream>
#include <FlyCapture2.h>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <QThread>

using namespace std;
using namespace FlyCapture2;


class Camera_Flycapture
{
public:
	bool  is_connect = false;
	float shutter   = 0.;      // 相机曝光时间
	float gain	    = 0.;      // 增益系数

private:
	Camera cap;                // 相机对象
	unsigned int  cam_num;     // 相机数量
	unsigned int  id;		   // 相机编号
	PGRGuid       guid;        // 硬件编号
	BusManager    busMgr;      // 管理相机
	Image		  imgRAW;	   // 原始
	Image		  imgBGR;	   // 转换 
	cv::Mat		  imgBGRMat;   // 图像
	unsigned int  rowByters;   // 缓存

	CameraInfo    cameraInfo;    // 相机信息
	TriggerMode   triggerMode;   // 触发模式
	TriggerDelay  triggerDelay;  // 触发时间
	FlyCapture2::Error	error;   // 存储错误

public:
	// 自动断开相机连接
	~Camera_Flycapture() {
		disconnect();
	};

	// 连接相机
	bool connect(unsigned int id) {
		this->id = id;
		// 检测到相机数量必须大于0，并且成功连接
		error = busMgr.GetNumOfCameras(&cam_num); is_ok(error);
		if (cam_num < id)
		{
			cerr << "Total camera num:" << cam_num << endl;
		}
		error = busMgr.GetCameraFromIndex(id, &guid); is_ok(error);
		error = cap.Connect(&guid); is_ok(error);
		error = cap.RestoreFromMemoryChannel(1); is_ok(error);
		// 设置软触发
		setTriggeerSoft();
		// 软触发的拍照
		startCaptureSoft();
		is_connect = true;
		// 打印相机信息
		printCameraInfo();
	
		return is_connect;
	};

	// 断开连接
	bool disconnect() {
		if (is_connect)
		{
			error = cap.StopCapture();
			error = cap.Disconnect();
			if (is_ok(error))
			{
				cout << "Successfully to disconnect camera" << endl;
				is_connect = false;
			}
			else
			{
				cerr << "Failed to disconnect camera" << endl;
				is_connect = true;
			}
		}
		return is_connect;
	};

	// 设置曝光
	bool setShutter(float ms = 100.) {
		if (!is_connect)
		{
			cout << "the camera has never been connected!" << endl;
			return false;
		}
		// 曝光仅支持这块范围
		assert(ms > 0 && ms < 1000);

		Property cameraProperty;                       //声明相机参数类的对象
		cameraProperty.type = SHUTTER;            //设置参数的类型，SHUTTER代表曝光
		error = cap.GetProperty(&cameraProperty);//获取当前曝光的参数
		is_ok(error);

		float a = cameraProperty.absValue;

		// 重新设置曝光的参数
		cameraProperty.absValue = ms; //重新设置曝光的大小，单位为ms
		cameraProperty.absControl = true; //采用绝对值输入
		cameraProperty.onOff = true; //使shutter设置有效
		cameraProperty.autoManualMode = false; //关闭自动模式，改成手动设置模式曝光的值才会固定
		error = cap.SetProperty(&cameraProperty);//设置曝光

		float b = cameraProperty.absValue;
		cout << "set exp=" << b << endl;
		cv::waitKey(10); // 等待设置生效

		Property sets(AUTO_EXPOSURE);
		error = cap.GetProperty(&sets);
		sets.absControl = true;
		sets.absValue = 0.;
		sets.onePush = false;
		sets.onOff = true;
		sets.autoManualMode = false;
		error = cap.SetProperty(&sets);
		cv::waitKey(100); // 等待设置生效

		Property frame;
		frame.type = FRAME_RATE;
		frame.onOff = false;
		frame.autoManualMode = false;
		error = cap.SetProperty(&frame);
		cv::waitKey(10); // 等待设置生效

		return is_ok(error);
	};

	bool startCaptureSoft() {
		error = cap.StartCapture(); is_ok(error);
		return is_ok(error);
	};

	// 设置增益
	bool setGain(float X = 0) {
		if (!is_connect)
		{
			cout << "the camera has never been connected!" << endl;
			return false;
		}
		assert(X >= 0); // 增益必须大于0
		gain = X;
		Property sets(GAIN);
		error = cap.GetProperty(&sets);
		sets.absValue = X;
		sets.absControl = true;
		sets.autoManualMode = false;
		sets.onePush = false;
		sets.onOff = false;
		sets.present = true;
		error = cap.SetProperty(&sets);
		cout << "Set gain=" << X << endl;
		cv::waitKey(50);
		return is_ok(error);
	};

	// 拍照并保存到图片中（没有+锁，因此不能直接用）
	bool capture(cv::Mat &img) {
		if (is_connect)
		{
			cap.RetrieveBuffer(&imgRAW);
			// 暂时需要从BGR转换为灰度图
			imgRAW.Convert(PIXEL_FORMAT_BGR, &imgBGR);
			rowByters = (double)imgBGR.GetReceivedDataSize() / imgBGR.GetRows();
			imgBGRMat = cv::Mat(imgBGR.GetRows(), imgBGR.GetCols(), CV_8UC3, imgBGR.GetData(), rowByters);
			if (imgBGRMat.empty())
			{
				cerr << "failed to capture image!" << endl;
				return false;
			}
			else {
				cv::cvtColor(imgBGRMat, img, CV_BGR2GRAY);
				return true;
			}
		}
		else
		{
			return false;
		}
	};

	// 打印相机信息
	void printCameraInfo() {
		if (is_connect)
		{
			error = cap.GetCameraInfo(&cameraInfo);

			cout << "\n ############# Camera Information ############# \n" << endl;
			cout << "Num:" << cam_num << endl;
			cout << "ID:\t" << id << endl;
			cout << "Model: " << cameraInfo.modelName << endl;
			cout << "W X H:" << cameraInfo.sensorResolution << endl;
			cout << "\n ############# ----- ############# \n" << endl;
		}
		else {
			cerr << "the camera has never been connected" << endl;
		}
	};
	
	bool setTriggeerSoft() {
		error = cap.GetTriggerMode(&triggerMode); is_ok(error);
		triggerMode.onOff = false;
		triggerMode.mode = 0;
		triggerMode.parameter = 0;
		triggerMode.source = 0;
		triggerMode.polarity = 1;
		error = cap.SetTriggerMode(&triggerMode); is_ok(error);
		cv::waitKey(100);
		cout << "Trigger-Soft" << endl;
		return is_ok(error);
	};

private:
	// 检查错误
	bool is_ok(FlyCapture2::Error &error) {
		if (error == PGRERROR_OK) {
			return true;
		}
		else {
			error.PrintErrorTrace();
			return false;
		}
	};
};
