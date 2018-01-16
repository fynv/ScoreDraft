#ifndef _QtPCMPlayer_h
#define _QtPCMPlayer_h

#include <qwidget.h>
#include "ui_QtPCMPlayer.h"

class QtPCMPlayer : public QWidget
{
	Q_OBJECT
public:
	QtPCMPlayer();
	virtual ~QtPCMPlayer();

private:
	Ui_QtPCMPlayer m_ui;
};

#endif

