import numpy as np
import scipy.signal
import wx
import wx.lib.newevent
import wx.lib.plot
import pyaudio
import ctypes

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
            title = "VK2ZJ_SINAD_Meter", 
            size = (600, 400))

        self.SetIcon(wx.Icon("VK2ZJ_SINAD_Meter.ico"))

        self.sample_rate = 48000 

        self.n_samples = 2400 # samples per buffer
        self.n_samples_fft = 2048 # must be <= n_samples
        self.n_samples_plot = 2000 # must be <= n_samples

        self.sinad_filter_coeff = 0.8
        self.sinad_filter_value = None

        self.plot_agc_coeff = 0.8
        self.plot_agc_value = None

        self.init_bpf()
        self.init_notch_filter()
        self.init_fft()

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.init_sinad_text()
        self.init_td_plot()
        self.init_fft_plot()
        self.panel.SetSizer(self.sizer)

        self.Bind(EVT_SOUND_DATA, self.process_sound_data)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.Centre()
        self.Show()
        self.start_audio()


    def init_bpf(self):
        # Psophometric filter as per ITU-T O.41 (see https://en.wikipedia.org/wiki/Psophometric_weighting)
        freq = (
            0, 16.66, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 
            1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000, 
            self.sample_rate/2)
        gain = (
            -85.0, -85.0, -63.0, -41.0, -21.0, -10.6, -6.3, -3.6, -2.0, -0.9, -0.0,
             0.6, 1.0, -0.0, -0.9, -1.7, -2.4, -3.0, -4.2, -5.6, -8.5, -15.0, -25.0,
             -36.0, -43.0, -80, -80)
        gain_lin = 10**(np.array(gain)/20.0)
        gain_lin[0] = 0
        gain_lin[-1] = 0
        self.bpf_coeffs = scipy.signal.firwin2(1001, freq, gain_lin, fs = self.sample_rate, antisymmetric = True)
        self.bpf_state = scipy.signal.lfilter_zi(self.bpf_coeffs, [1])


    def init_notch_filter(self):
        # Notch filter coefficients for removing the test tone
        self.tone_frequency = 1000.0
        self.notch_filter_coeffs = scipy.signal.iirfilter(
            16,
            [self.tone_frequency - 50, self.tone_frequency + 50],
            fs = self.sample_rate,
            rp = 0.1,
            rs = 150,
            btype = 'bandstop',
            ftype = 'ellip',
            output = 'sos')
        self.notch_filter_state = scipy.signal.sosfilt_zi(self.notch_filter_coeffs)


    def init_fft(self):
        self.fft_plot_min = -100
        self.fft_plot_max = 10
        self.fft_plot_offset = -80
        self.fft_plot_freq_limit = 6000
        self.fft_window = np.blackman(self.n_samples_fft)
        self.fft_x_data = np.arange(0, self.fft_plot_freq_limit, self.sample_rate / self.n_samples_fft)
        self.n_samples_fft_plot = len(self.fft_x_data)


    def init_sinad_text(self):
        self.spacer_panel = wx.Panel(self.panel)
        self.spacer_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sizer.Add(self.spacer_panel, proportion = 1, border = 1, flag = wx.EXPAND)
        self.sinad_text = wx.StaticText(self.panel, -1, "SINAD = --- dB")
        self.sinad_text.SetFont(wx.Font(18, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.sinad_text.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sinad_text.SetForegroundColour(wx.Colour(0, 255, 255))
        self.sizer.Add(self.sinad_text, proportion = 2, border = 1, flag = wx.CENTER)


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
        self.sizer.Add(self.td_panel, proportion = 16, border = 1, flag = wx.EXPAND)


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
        self.sizer.Add(self.fd_panel, proportion = 16, border = 1, flag = wx.EXPAND)
        self.spacer_panel2 = wx.Panel(self.panel)
        self.spacer_panel2.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.sizer.Add(self.spacer_panel2, proportion = 1, border = 1, flag = wx.EXPAND)


    def process_sound_data(self, event):
        # Convert samples from faw 32-bit buffer to array of floats
        samples = np.frombuffer(event.sound_data, dtype = np.int32).astype(np.float64)

        # Band-pass filter the audio with the psophometric filter as per ITU-T O.41 
        bpf_samples, self.bpf_state = scipy.signal.lfilter(self.bpf_coeffs, [1], samples, zi = self.bpf_state)

        # Calculate the signal + noise + distortion power, in dB
        sind_power_dB = 20 * np.log10(np.sqrt(np.mean(bpf_samples**2)))

        # Notch out the tone
        notched_samples, self.notch_filter_state = scipy.signal.sosfilt(self.notch_filter_coeffs, bpf_samples, zi = self.notch_filter_state)

        # Calculate the noise + distortion power, in dB
        nd_power_dB = 20 * np.log10(np.sqrt(np.mean(notched_samples**2)))

        # Smooth out the SINAD estimate, so that that displayed values changes slowly    
        new_sinad = sind_power_dB - nd_power_dB
        if self.sinad_filter_value is None:
            self.sinad_filter_value = new_sinad
        else: 
            self.sinad_filter_value = self.sinad_filter_coeff * self.sinad_filter_value + (1 - self.sinad_filter_coeff) *  new_sinad
        self.sinad_text.SetLabel("SINAD = %3.1fdB" % self.sinad_filter_value)

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
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format = pyaudio.paInt32, # -> 24 bits per sample
            channels = 1,
            rate = self.sample_rate,
            input = True,
            frames_per_buffer = self.n_samples,
            stream_callback = audio_input_callback)
        self.audio_stream.start_stream()


    def OnClose(self, event):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()
        self.Destroy()


app = wx.App()
app_window = SINAD_window()
app.MainLoop()

