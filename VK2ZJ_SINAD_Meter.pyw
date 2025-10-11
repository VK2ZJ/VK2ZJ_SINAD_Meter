import numpy as np
import scipy.signal
import wx
import wx.lib.newevent
import wx.lib.plot
import pyaudio
import ctypes
import dsp

# This hack sets the Windows taskbar icon to be the same as in SetIcon
# below. Otherwise, Windows would use the python.exe icon. 
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"VK2ZJ_SINAD_meter")


SoundDataEvent, EVT_SOUND_DATA = wx.lib.newevent.NewEvent()


def audio_input_callback(in_data, frame_count, time_info, status):
    evt = SoundDataEvent(sound_data = in_data)
    # global app_window
    wx.PostEvent(app_window, evt)
    return (None, pyaudio.paContinue)


class SINAD_window(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(
            self, 
            None, 
            title = "VK2ZJ SINAD Meter", 
            size = (800, 600))

        self.SetIcon(wx.Icon("VK2ZJ_SINAD_Meter.ico"))

        self.sample_rate = 48000 

        self.n_samples = 2400 # samples per buffer
        self.n_samples_fft = 2048 # must be <= n_samples
        self.n_samples_plot = 1000 # must be <= n_samples
        self.n_samples_sinad_plot = 1000
        self.sinad_plot_max = 30

        self.sinad_filter_smoothing_coeff = 0.95 # 0 to almost 1. Smoothes the SINAD measurements so that they don't jump around as much.
        self.sinad_filter_value = 0
        self.sinad_filter_value_pm = 0

        self.plot_agc_coeff = 0.8
        self.plot_agc_value = None

        self.init_bpf()
        self.init_notch_filter()
        self.init_fft()

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.init_device_selection()
        self.init_sinad_text()
        self.init_sinad_plot()
        self.init_td_plot()
        self.init_fft_plot()
        self.panel.SetSizer(self.sizer)

        self.Bind(EVT_SOUND_DATA, self.process_sound_data)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.Centre()
        self.Show()

        try:
            self.selected_device_index = self.device_indices[0]
            self.start_audio()
        except:
            pass


    def init_bpf(self):
        # Psophometric filter as per ITU-T O.41 (see https://en.wikipedia.org/wiki/Psophometric_weighting)
        self.bpf_coeffs_pm, self.bpf_state_pm = dsp.psophometric_filter(self.sample_rate)
        


    def init_notch_filter(self):
        # Notch filter coefficients for removing the test tone
        self.tone_frequency = 1e3
        self.notch_filter_coeffs, self.notch_filter_state = dsp.notch_filter(self.sample_rate, self.tone_frequency)
        self.notch_filter_coeffs_pm, self.notch_filter_state_pm = dsp.notch_filter(self.sample_rate, self.tone_frequency)


    def init_fft(self):
        self.fft_plot_min = -100
        self.fft_plot_max = 10
        self.fft_plot_offset = -80
        self.fft_plot_freq_limit = 6000
        self.fft_window = np.blackman(self.n_samples_fft)
        self.fft_x_data = np.arange(0, self.fft_plot_freq_limit, self.sample_rate / self.n_samples_fft)
        self.n_samples_fft_plot = len(self.fft_x_data)


    def init_device_selection(self):
        self.audio = pyaudio.PyAudio()

        self.device_names = []
        self.device_indices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                self.device_names.append(info['name'])
                self.device_indices.append(info['index'])

        # Horizontal sizer for device and averaging selection
        #self.control_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Device selection combo box
        self.device_combo = wx.ComboBox(
            self.panel,
            choices=self.device_names,
            style=wx.CB_READONLY)
        self.device_combo.SetSelection(0)
        self.device_combo.Bind(wx.EVT_COMBOBOX, self.on_device_change)
        #self.control_sizer.Add(self.device_combo, 1, wx.EXPAND | wx.ALL, 5)

        self.sizer.Add(self.device_combo, 0, wx.EXPAND)


    def init_sinad_text(self):
        self.spacer_panel = wx.Panel(self.panel)
        self.spacer_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sizer.Add(self.spacer_panel, proportion = 1, border = 2, flag = wx.EXPAND)
        self.sinad_text = wx.StaticText(self.panel, -1, "SINAD = %3.1fdB (psopho: %3.1fdB)" % (0, 0))
        self.sinad_text.SetFont(wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.sinad_text.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sinad_text.SetForegroundColour(wx.Colour(0, 255, 255))
        self.sizer.Add(self.sinad_text, proportion = 1, border = 1, flag = wx.CENTER)


    def init_sinad_plot(self):
        self.sinad_x_data = np.arange(self.n_samples_sinad_plot)
        self.sinad_y_data = np.zeros(self.n_samples_sinad_plot)
        self.sinad_y_data_pm = np.zeros(self.n_samples_sinad_plot)
        xy_data = list(zip(self.sinad_x_data, self.sinad_y_data))
        self.sinad_line = wx.lib.plot.PolySpline(xy_data, colour = wx.Colour(0, 255, 255), width = 1)
        self.sinad_graphics = wx.lib.plot.PlotGraphics([self.sinad_line])
        self.sinad_panel = wx.lib.plot.PlotCanvas(self.panel)
        self.sinad_panel.axesPen = wx.Pen(wx.Colour(100, 100, 100), 1)
        self.sinad_panel.enableGrid = True
        self.sinad_panel.ySpec = (0, self.sinad_plot_max)
        self.sinad_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sinad_panel.SetForegroundColour(wx.Colour(100, 100, 100))
        self.sinad_panel.Draw(self.sinad_graphics)
        self.sizer.Add(self.sinad_panel, proportion = 16, border = 10, flag = wx.EXPAND | wx.BOTTOM)


    def init_td_plot(self):
        self.td_x_data = np.arange(self.n_samples_plot)
        y_data = np.zeros(self.n_samples_plot)
        xy_data = list(zip(self.td_x_data, y_data))
        self.td_line = wx.lib.plot.PolySpline(xy_data, colour = wx.Colour(0, 255, 255), width = 1)
        self.td_graphics = wx.lib.plot.PlotGraphics([self.td_line])
        self.td_panel = wx.lib.plot.PlotCanvas(self.panel)
        self.td_panel.axesPen = wx.Pen(wx.Colour(100, 100, 100), 1)
        self.td_panel.enableGrid = False
        self.td_panel.ySpec = (-2, 2)
        self.td_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.td_panel.Draw(self.td_graphics)
        self.sizer.Add(self.td_panel, proportion = 16, border = 0, flag = wx.EXPAND)


    def init_fft_plot(self):
        y_data = np.zeros(self.n_samples_fft)
        xy_data = list(zip(self.fft_x_data, y_data))
        self.fd_line = wx.lib.plot.PolySpline(xy_data, colour = wx.Colour(0, 255, 255), width = 1)
        self.fd_graphics = wx.lib.plot.PlotGraphics([self.fd_line])
        self.fd_panel = wx.lib.plot.PlotCanvas(self.panel)
        self.fd_panel.axesPen = wx.Pen(wx.Colour(100, 100, 100), 1)
        self.fd_panel.enableGrid = True
        self.fd_panel.ySpec = (self.fft_plot_min, self.fft_plot_max)
        self.fd_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.fd_panel.SetForegroundColour(wx.Colour(100, 100, 100))
        self.fd_panel.Draw(self.fd_graphics)
        self.sizer.Add(self.fd_panel, proportion = 16, border = 10, flag = wx.EXPAND | wx.BOTTOM)


    def process_sound_data(self, event):
        # Convert samples from faw 32-bit buffer to array of floats
        samples = np.frombuffer(event.sound_data, dtype = np.int32).astype(np.float64)

        # Get SINAD for this block of samples, without any bandpass filter
        new_sinad, self.bpf_state_, self.notch_filter_state = dsp.calculate_SINAD(
            self.sample_rate,
            samples,
            None,
            None,
            self.notch_filter_coeffs,
            self.notch_filter_state)

        # Smooth out the SINAD estimate, so that that displayed values changes slowly    
        self.sinad_filter_value = self.sinad_filter_smoothing_coeff * self.sinad_filter_value + (1 - self.sinad_filter_smoothing_coeff) *  new_sinad

        # Repeat. but with the Psophometric filter
        new_sinad_pm, self.bpf_state_pm, self.notch_filter_state_pm = dsp.calculate_SINAD(
            self.sample_rate,
            samples,
            self.bpf_coeffs_pm,
            self.bpf_state_pm,
            self.notch_filter_coeffs_pm,
            self.notch_filter_state_pm)

        self.sinad_filter_value_pm = self.sinad_filter_smoothing_coeff * self.sinad_filter_value_pm + (1 - self.sinad_filter_smoothing_coeff) *  new_sinad_pm

        self.sinad_text.SetLabel("SINAD = %3.1fdB (psopho: %3.1fdB)" % (self.sinad_filter_value, self.sinad_filter_value_pm))

        # Plot the SINAD over time
        self.sinad_y_data[:] = np.roll(self.sinad_y_data, -1)
        self.sinad_y_data[-1] = self.sinad_filter_value
        self.sinad_y_data_pm[:] = np.roll(self.sinad_y_data_pm, -1)
        self.sinad_y_data_pm[-1] = self.sinad_filter_value_pm
        xy_data = list(zip(self.sinad_x_data, self.sinad_y_data))
        self.sinad_line = wx.lib.plot.PolySpline(xy_data, colour = wx.Colour(0, 255, 255), width = 1)
        xy_data_pm = list(zip(self.sinad_x_data, self.sinad_y_data_pm))
        self.sinad_line_pm = wx.lib.plot.PolySpline(xy_data_pm, colour = wx.Colour(200, 0, 200), width = 1)
        self.sinad_graphics = wx.lib.plot.PlotGraphics([self.sinad_line_pm, self.sinad_line])
        self.sinad_panel.Draw(self.sinad_graphics)

        # Apply slow AGC to the samples to keep then in -1..1 range, to simplify the real time plotting
        new_agc = 1.0 / np.max(np.abs(samples))
        if self.plot_agc_value is None:
            self.plot_agc_value = new_agc 
        else:
            self.plot_agc_value = self.plot_agc_coeff * self.plot_agc_value + (1 - self.plot_agc_coeff) *  new_agc

        # Plot the time domain data CRO style
        td_y_samples = samples[:self.n_samples_plot] * self.plot_agc_value
        xy_data = list(zip(self.td_x_data, td_y_samples))
        self.td_line = wx.lib.plot.PolySpline(xy_data, colour = wx.Colour(0, 255, 255), width = 1)
        self.td_graphics = wx.lib.plot.PlotGraphics([self.td_line])
        self.td_panel.Draw(self.td_graphics)

        # Plot the FFT
        samples_for_fft = samples[:self.n_samples_fft] * self.fft_window * self.plot_agc_value
        fft_samples = np.fft.rfft(samples_for_fft)[:self.n_samples_fft_plot]
        fft_pwr = fft_samples.real**2 + fft_samples.imag**2 + 1e-9
        fft_dB = 10 * np.log10(fft_pwr) + self.fft_plot_offset
        xy_data = list(zip(self.fft_x_data, fft_dB))
        self.fd_line = wx.lib.plot.PolySpline(xy_data, colour = wx.Colour(0, 255, 255), width = 1)
        self.fd_graphics = wx.lib.plot.PlotGraphics([self.fd_line])
        self.fd_panel.Draw(self.fd_graphics)

        self.Update()


    def start_audio(self):
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.audio_stream = self.audio.open(
            format = pyaudio.paInt32,  # -> 24 bits per sample
            channels = 1,
            rate = self.sample_rate,
            input = True,
            input_device_index = self.selected_device_index,
            frames_per_buffer = self.n_samples,
            stream_callback = audio_input_callback)
        self.audio_stream.start_stream()


    def on_device_change(self, event):
        selection = self.device_combo.GetSelection()
        new_device_index = self.device_indices[selection]
        if new_device_index != self.selected_device_index:
            self.selected_device_index = new_device_index
            self.start_audio()


    def on_close(self, event):
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.audio.terminate()
        self.Destroy()


app = wx.App()
app_window = SINAD_window()
app.MainLoop()

