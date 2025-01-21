import threading
import wx
from wx.lib.masked import NumCtrl

LABELS = [
    "Ses seviyesi",
    "Sıcaklık",
    "Nem",
    "Mesafe",
    "Yağmur",
    "Işık",
    "Gaz",
    "Yakınlık",
    "Hava Kalitesi",
    "Nabız",
]


class _ValuePanel(wx.Panel):
    values: dict[str, int] = {}

    def __init__(self, parent, labels, *args, **kwargs):
        """Create a panel with labeled text controls.

        Args:
            parent: Parent window or panel.
            labels: List of strings representing the labels for the input fields.
        """
        super().__init__(parent, *args, **kwargs)

        self.controls = {}

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        for label in labels:
            row_sizer = wx.BoxSizer(wx.HORIZONTAL)

            label_widget = wx.StaticText(self, label=label)
            row_sizer.Add(label_widget, proportion=1, flag=wx.ALL | wx.EXPAND, border=5)

            text_ctrl = wx.TextCtrl(self)
            text_ctrl.SetValue("0")
            row_sizer.Add(text_ctrl, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)

            self.controls[label] = text_ctrl

            main_sizer.Add(row_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)

        self.SetSizer(main_sizer)


class _MainFrame(wx.Frame):
    value_panel: _ValuePanel

    def __init__(self):
        super().__init__(None, title="Robot")

        self.value_panel = _ValuePanel(self, LABELS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.value_panel, proportion=1, flag=wx.EXPAND | wx.ALL)

        self.SetSizer(sizer)
        self.Fit()


class ControlPanel:
    _frame: _MainFrame
    _app: wx.App

    def __init__(self) -> None:
        self._app = wx.App(False)
        self._frame = _MainFrame()

    def main_loop(self):
        """Enter the main loop of wxwidgets"""
        self._frame.Show()
        self._app.MainLoop()

    def get_value(self, label: str) -> float:
        """Get the value from a specific text control.
        Args:
            label: The label of the text control.

        Returns:
            str: The value of the text control.
        """

        if label not in self._frame.value_panel.controls:
            return 0

        value = self._frame.value_panel.controls[label].GetValue()

        try:
            return float(value)
        except ValueError:
            return 0


class ControlPanelThread(threading.Thread):
    _control_panel: ControlPanel | None = None

    def __init__(self):
        super().__init__()
        self.daemon = True

    def run(self):
        self._control_panel = ControlPanel()
        self._control_panel._frame.Bind(
            wx.EVT_CLOSE, lambda _: self._control_panel and self._control_panel._app.ExitMainLoop()
        )
        wx.DisableAsserts()
        self._control_panel.main_loop()

        self._control_panel._frame.Destroy()
        self._control_panel._app.Destroy()
        self._control_panel = None

    def get_value(self, label: str):
        if not self._control_panel:
            return 0

        value = self._control_panel.get_value(label)

        if value is None:
            return 0

        return value
