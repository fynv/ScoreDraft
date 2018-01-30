#include <QFile>
#include <qapplication.h>
#include <QtNetwork/QLocalSocket>
#include <QtNetwork/QLocalServer>
#include "QtPCMPlayer.h"
#include <string.h>

int main(int argc, char *argv[])
{
	FILE *fp = fopen("rootpath", "r");
	if (fp)
	{
		char root[1024];
		fscanf(fp, "%s", root);
		root[strlen(root) - 1] = 0;

		char QtPluginPath[1024];
		sprintf(QtPluginPath, "%s/QtPlugins", root+1);
		fclose(fp);

		QFile file("rootpath");
		file.remove();

		QApplication::addLibraryPath(QtPluginPath);
	}
	else
	{
		QApplication::addLibraryPath("./QtPlugins");
	}

	QApplication app(argc, argv);

	QLocalSocket socket;
	socket.connectToServer("QtPCMPlayer");

	if (socket.waitForConnected(500))
	{
		socket.close();
		return -1;
	}

	QLocalServer server;
	if (!server.listen("QtPCMPlayer")) return -1;

	QtPCMPlayer player(&server);
	player.show();

	int res= app.exec();

	server.close();

	return res;
}