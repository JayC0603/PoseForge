"""Airy white desktop theme with blue accents."""
DESKTOP_STYLE = """
QWidget { background: #f4f7fb; color: #24334b; font-family: 'Segoe UI'; font-size: 12px; }
QLabel { background: transparent; }
QWidget#appHeader { background: #ffffff; border-bottom: 1px solid #e3e9f2; }
QLabel#brandMark { color: #3975e8; font-size: 28px; }
QLabel#brand { font-size: 21px; font-weight: 700; }
QLabel#headerTag { color: #718096; font-size: 10px; letter-spacing: 2px; padding: 4px 8px; border: 1px solid #dce4f0; border-radius: 4px; }
QWidget#sidebar { background: #ffffff; border-right: 1px solid #e3e9f2; }
QWidget#workspace, QWidget#cameraGrid { background: #f4f7fb; }
QLabel#pageTitle { font-size: 23px; font-weight: 700; }
QLabel#sectionTitle { font-size: 17px; font-weight: 600; }
QLabel#eyebrow { color: #5375a0; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
QLabel#muted { color: #66768d; font-size: 12px; }
QLabel#hint, QLabel#footer { color: #728198; font-size: 11px; }
QLabel#guide { color: #4e6582; background: #edf4ff; border: 1px solid #d5e4fb; border-radius: 8px; padding: 12px; }
QLabel#status { color: #3569b7; background: #eaf2ff; border: 1px solid #cddff9; border-radius: 6px; padding: 5px 10px; font-size: 11px; }
QWidget#previewStage { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #ffffff,stop:1 #edf4ff); border: 1px solid #dce5f0; border-radius: 12px; }
QLabel#viewSymbol { color: #3975e8; font-size: 72px; }
QLabel#emptyTitle { font-size: 24px; font-weight: 600; padding: 8px; }
QLabel#step { color: #526783; background: #ffffff; border: 1px solid #dae5f4; border-radius: 6px; padding: 7px 12px; }
QPushButton { background: #ffffff; border: 1px solid #d7e0ed; border-radius: 7px; padding: 8px 10px; font-weight: 600; }
QPushButton:hover { background: #edf4ff; border-color: #8bafea; }
QPushButton:pressed { background: #e1ecff; }
QPushButton:checked { color: #245fc1; background: #eaf2ff; border-color: #9bbcef; }
QPushButton:disabled { color: #94a1b4; background: #f0f4fa; border-color: #e3e9f2; }
QPushButton#primary { background: #3975e8; border-color: #3975e8; color: #ffffff; min-width: 96px; padding: 10px 16px; }
QPushButton#primary:hover { background: #2864d4; }
QPushButton#primary:disabled { background: #dfe9f8; color: #8c9eb8; border-color: #d4dfef; }
QPushButton#danger { color: #bd4751; background: #fff4f4; border-color: #f0d2d5; padding: 10px 14px; }
QPushButton#danger:disabled { background: #f4f5f8; color: #a6aab4; border-color: #e4e7ed; }
QPushButton#sidebarToggle { font-size: 22px; padding: 0; color: #3975e8; }
QComboBox, QLineEdit, QDoubleSpinBox { background: #ffffff; border: 1px solid #d5dfec; border-radius: 7px; padding: 8px 10px; min-height: 20px; selection-background-color: #d4e5ff; }
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus { border-color: #3975e8; }
QComboBox QAbstractItemView { background: #ffffff; color: #24334b; selection-background-color: #e0ecff; padding: 4px; }
QCheckBox { spacing: 7px; background: transparent; color: #5c6d86; }
QCheckBox::indicator { width: 14px; height: 14px; background: #ffffff; border: 1px solid #a6b5ca; border-radius: 3px; }
QCheckBox::indicator:checked { background: #3975e8; border-color: #3975e8; }
QCheckBox::indicator:disabled { background: #edf1f7; border-color: #d3dce9; }
QWidget#sourcePanel { background: #ffffff; border: 1px solid #dce5f0; border-radius: 10px; }
QWidget#sourcePanel QLabel, QWidget#sourcePanel QCheckBox { background: transparent; }
QLabel#sourceTitle { color: #566780; font-size: 11px; padding: 2px 0; }
QWidget#predictionPanel { background: #f0f4fa; border-radius: 8px; }
QLabel#prediction { color: #3975e8; font-size: 18px; font-weight: 600; padding: 4px 0; }
QLabel#poseInfo { color: #6b7c94; font-size: 11px; }
QLabel#metrics { color: #65768e; font-size: 11px; padding: 7px 0 0 0; }
QLabel#probabilityName { color: #52647f; font-size: 11px; }
QProgressBar { background: #e6edf8; border: 0; border-radius: 3px; min-height: 12px; max-height: 12px; color: #193f77; text-align: center; font-size: 10px; }
QProgressBar::chunk { background: #8db9f5; border-radius: 3px; }
QScrollArea { border: 0; background: transparent; }
QScrollBar:vertical { background: #eef2f8; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #bccade; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #24334b; background: #ffffff; border: 1px solid #c5d5ec; padding: 7px; }
"""
