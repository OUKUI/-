from qfluentwidgets import setTheme, Theme
from PyQt5.QtGui import QFont


def setup_theme(dark=True):
    setTheme(Theme.DARK if dark else Theme.LIGHT)


def get_chinese_font(size=9):
    return QFont("Microsoft YaHei", size)
