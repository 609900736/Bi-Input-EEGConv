# coding:utf-8
# TODO: visualization method

import pywt
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(pywt.wavelist())
    print(pywt.ContinuousWavelet('gaus8'))
    x = np.linspace(0, 4, 1000)
    y = (1 + 0.5 * np.cos(2 * np.pi * 5 * x)
         ) * np.cos(2 * np.pi * 50 * x)
    plt.plot(x, y)  # doctest: +SKIP
    coef, freqs = pywt.cwt(y,
                           np.arange(1, 129),
                           'gaus4',
                           sampling_period=1.0 / 250)
    plt.matshow(abs(coef))  # doctest: +SKIP
    print(freqs)
    plt.figure()
    plt.contourf(x, freqs, abs(coef))
    f, t, Zxx = signal.stft(y, fs=250, window='hann')
    plt.figure()
    print(f)
    plt.pcolormesh(t,
                   f,
                   np.abs(Zxx),
                   vmin=np.min(np.abs(Zxx)),
                   vmax=np.max(np.abs(Zxx)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # t = np.linspace(-1, 1, 200, endpoint=False)
    # sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    # widths = np.arange(1, 31)
    # cwtmatr, freqs = pywt.cwt(sig, widths, 'gaus8')
    # plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
    # plt.show() # doctest: +SKIP
    #t = np.linspace(0, 1, 2000, False)  # 1 second
    #sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)  # 构造10hz和20hz的两个信号
    #sig = np.array([[sig,sig],[sig,sig]])
    #print(sig.shape)
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #ax1.plot(t, sig[0,0,:])
    #ax1.set_title('10 Hz and 20 Hz sinusoids')
    #ax1.axis([0, 1, -2, 2])

    #b, a = signal.butter(4, [14,30], 'bandpass', fs=2000, output='ba') #采样率为1000hz，带宽为15hz，输出ba
    #z, p, k = signal.tf2zpk(b, a)
    #eps = 1e-9
    #r = np.max(np.abs(p))
    #approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    #print(approx_impulse_len)
    #filtered = signal.filtfilt(b, a, sig, method='gust', irlen=approx_impulse_len) #将信号和通过滤波器作用，得到滤波以后的结果。在这里sos有点像冲击响应，这个函数有点像卷积的作用。
    #ax2.plot(t, filtered[0,0,:])
    #ax2.set_title('After 15 Hz high-pass filter')
    #ax2.axis([0, 1, -2, 2])
    #ax2.set_xlabel('Time [seconds]')
    #plt.tight_layout()
    #plt.show()
    #print('BiInputConv.core.utils')