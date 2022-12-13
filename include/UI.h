/* UI交互界面（用于触发各类信号）*/ 

#pragma once
#include <fstream>
#include <QtWidgets/QMainWindow>
#include <QPushButton>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <QKeyEvent>
#include <QMutex>
#include <QMessageBox>

#include "ui_UI.h"
#include "CameraEngineer.h"
#include "Calibrator.h"


class UI : public QMainWindow
{
    Q_OBJECT

public:
	Ui::UIClass ui;
	Camera_Flycapture cap;  // 相机类型，可以替换
	vector<string> filenames;   // 用于投影图片
	Calibrator calibrator;
	string save_folder = "./out";
	string img_end     = ".bmp";
	int folder_count = 0;
	QMutex mutex;
	
public:
	UI(QWidget *parent = Q_NULLPTR) : QMainWindow(parent) {
		ui.setupUi(this);
		// 设置固定大小
		setFixedSize(960, 900);

		// 因为并不支持Mat传递，因此要注册一下
		qRegisterMetaType<Mat>("Mat");
		connect(CameraEngineer::Get(this->cap),
			SIGNAL(img_signal(Mat)),  // 发送数据
			ui.imgWindow,
			SLOT(imshow_slot(Mat))    // 接收数据
		);
	};

	// 重写closeEvent: 确认退出对话框
	void closeEvent(QCloseEvent *event)
	{	
		CameraEngineer::Get(this->cap)->disconnect();        // 断开相机连接
		CameraEngineer::Get(this->cap)->stop_flag = true;    // 退出循环
	}

	// 投影功能
	void function_project(const string &filename) {
		const char *file = filename.data();
		SystemParametersInfoA(SPI_SETDESKWALLPAPER, 1, (PVOID)file, SPIF_SENDCHANGE);
	}

// 槽函数
public slots:
	/* 相机连接 */
	void slot_camera_connect() {
		// 先断开之前相机
		CameraEngineer::Get(this->cap)->disconnect();

		// 连接新相机
		this->cap.connect((unsigned int)ui.spinBox_ID->value());

		this->cap.setShutter(ui.spinBox_shutter->value());
		this->cap.setGain(ui.spinBox_gain->value());

		// 设置连续曝光（不动时候不变）

		connect(ui.spinBox_shutter,
			QOverload<int>::of(&QSpinBox::valueChanged),
			[=](int shutter) {
			this->cap.setShutter(shutter);
		});

		// 设置连续增益
		connect(ui.spinBox_gain,
			QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			[=](double X) {
			this->cap.setGain(X);
		});
	}

	/* 相机拍照 */
	void slot_camrea_capture() {
		// 默认保存在out文件夹
		CameraEngineer::Get(this->cap)->captureSaveMutex(save_folder, img_end);
		}

	/* 写入图片 */
	void slot_screen_write() {
		QStringList files = QFileDialog::getOpenFileNames(
			this,
			QString::fromLocal8Bit("Open images"),
			QDir::currentPath() + "/patterns",
			tr("images files(*.bmp *.jpg *.png)"));
		if (files.isEmpty())
		{
			return;
		}
		// 清空文件
		filenames.resize(0);
		cout << "Successfully load images:" << endl;
		for (int i = 0; i < files.size(); i++)
		{
			QString file = files.at(i);
			qDebug() << file << endl;
			string filename = file.toStdString();
			filenames.emplace_back(filename);
		}
		// 投影第一幅图片作为背景
		function_project(filenames[0]);
	}

	/* 投影并拍照 */
	void slot_project_capture() {
		if (filenames.size() < 1)
		{
			cout << "No images written!" << endl;
			return;
		}
		folder_count += 1;
		// 位姿确定，停止检测
		CameraEngineer::Get(this->cap)->stop_detect();
		cout << "Start to project images:" << endl;

		// 将图片编号设置为0
		CameraEngineer::Get(this->cap)->setImageCountZeros();
		for (int i = 0; i < filenames.size(); i++)
		{
			// 投影图片
			function_project(filenames[i]);
			string save_folder_final = save_folder + "/" + to_string(folder_count);
			Sleep(1500);
			// 拍照+保存
			string save_file = CameraEngineer::Get(this->cap)->captureSaveMutex(save_folder_final, img_end);
			Sleep(1500);
		}
		//imgFile << "------" << endl;
		function_project(filenames[0]);
		cout << "Finish!" << endl;
		// 设置重新检测
		CameraEngineer::Get(this->cap)->start_detect();
	}

	void slot_calib() {
		cout << "please use python script to calibrate the camera" << endl;
		/*打开标定的文件
		QString file = QFileDialog::getOpenFileName(
			this,
			QString::fromLocal8Bit("请选择txt图片文件"),
			QDir::currentPath() + "/srcs_calib_camera/data",
			tr("Text files(*.txt)"));
		if (file.isEmpty())
		{
			return;
		}
		 因为要大规模计算，断开相机连接
		CameraEngineer::Get(this->cap)->disconnect();
		mutex.lock();
		 读取并标定
		calibrator.calib(file.toStdString());
		mutex.unlock();
		 重新连接
		slot_camera_connect();*/
	}

	// 设置配置
	void slot_config() {
		QString file = QFileDialog::getOpenFileName(
			this,
			QString::fromLocal8Bit("please to select a config file"),
			QDir::currentPath() + "/srcs_calib_camera/data",
			tr("Text files(*.xml)"));
		if (file.isEmpty()){
			return;
		}
		string filename = file.toStdString();

		// 加载配置文件
		calibrator.load_config(filename);
		
		CameraEngineer::Get(this->cap)->setCalibrator(calibrator);
		CameraEngineer::Get(this->cap)->start_detect();
	}

	void slot_visual() {
		// 01 读取
		QString file = QFileDialog::getOpenFileName(
			this,
			QString::fromLocal8Bit("Select a pose file"),
			QDir::currentPath() + "/srcs_calc_pose/out",
			tr("Text files(*.txt)"));
		if (file.isEmpty())
		{
			return;
		}
		string filename = file.toStdString();
		CameraEngineer::Get(this->cap)->load_nextpose(filename);
	}
};
