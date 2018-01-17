#include <qapplication.h>
#include <QtNetwork/QLocalSocket>
#include <QtNetwork/QLocalServer>
#include "QtPCMPlayer.h"

int main(int argc, char *argv[])
{
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