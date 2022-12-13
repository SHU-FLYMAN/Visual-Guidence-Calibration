/* Ö÷³ÌÐò */
#include <QtWidgets/QApplication>

#include "include/UI.h"


int main(int argc, char *argv[])
{
	QApplication application(argc, argv);
	UI window;
	window.show();
	return application.exec();
}
