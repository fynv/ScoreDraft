#include <qapplication.h>
#include "QtPCMPlayer.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QtPCMPlayer player;
	player.show();

	return app.exec();
}