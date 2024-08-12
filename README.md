<p align="center">
  <img src="https://raw.githubusercontent.com/s0ma0000/Blink-Detection-System/main/image/figure1.png" width="400" height="250">
  <img src="https://raw.githubusercontent.com/s0ma0000/Blink-Detection-System/main/image/figure2.png" width="400" height="250">
  <img src="https://raw.githubusercontent.com/s0ma0000/Blink-Detection-System/main/gif1.gif" width="400" height="250">
</p>

# Blink-Detection-System
Real-time detection of facial landmarks from camera images, especially tracking eye movements to detect and count blinks.

## Overview

This project uses a combination of OpenCV and MediaPipe to analyze real-time video from a camera, detect facial landmarks, and monitor blink frequency using eye landmarks in particular. This system can be used to measure eye fatigue and concentration.

## Requirement

- **Operating System**: macOS
- **OpenCV**: Used to acquire video from the camera and for basic image processing.
- **MediaPipe**: Used for facial landmark detection.
- **Matplotlib**: Used for data visualization.
- **NumPy**: Used for numerical calculations.

## Features

- Real-time detection of facial landmarks using the MediaPipe Face Mesh model.
- Blink detection using eye landmarks.
- Real-time display of blink frequency on the frame.
- **Fourier Transform Analysis**: The system performs a Fourier transform on the iris position data to analyze frequency components. This allows for the detection of periodic movements in the iris position, which can provide insights into eye movement patterns.

## Fourier Transform Analysis

### Overview of Fourier Transform

The Fourier transform is a mathematical technique used to transform a signal from its original time domain into the frequency domain. In this project, the Fourier transform is used to analyze the frequency components of the iris position over time, which can reveal periodic patterns in eye movements.

## Mathematical Expression

Given a discrete time series \( x[n] \), the Discrete Fourier Transform (DFT) is defined as follows:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}
$$

Where:

- \( X[k] \) is the frequency domain representation at frequency bin \( k \),
- \( x[n] \) is the time-domain signal at sample \( n \),
- \( N \) is the total number of samples,
- \( j \) is the imaginary unit.

The inverse Fourier transform, which transforms the frequency domain signal back into the time domain, is given by:

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j \frac{2\pi}{N} kn}
$$


### Application in the Project

In this project, the iris position data \( x[n] \) is collected over a series of frames. The Fourier transform is then applied to this data to convert it into the frequency domain \( X[k] \). This process is performed using the `numpy.fft.fft` function, which efficiently computes the DFT.

### Frequency Analysis

After performing the Fourier transform, the resulting frequency components \( X[k] \) are analyzed to determine the amplitude of various frequencies present in the iris movement data. The amplitude \( |X[k]| \) indicates the strength of a particular frequency component.

### Visualization

The frequency components are visualized in a plot, where the x-axis represents the frequency and the y-axis represents the amplitude. This visualization helps in identifying dominant frequencies, which can provide insights into periodic eye movements, such as saccades or fixations.

### Code Implementation

Here is a snippet of the code where the Fourier transform is applied:

```python
right_fft = np.fft.fft(list(right_iris_relative_pos))
left_fft = np.fft.fft(list(left_iris_relative_pos))
freq = np.fft.fftfreq(N, d=1)

# FFT の結果は N の半分までが有用（ナイキスト周波数まで）
freq_half = freq[:N//2]
right_fft_half = np.abs(right_fft)[:N//2]
left_fft_half = np.abs(left_fft)[:N//2]

ax_iris_fft.plot(freq_half, right_fft_half, label='Right Iris FFT')
ax_iris_fft.plot(freq_half, left_fft_half, label='Left Iris FFT')

## Author

[twitter](https://twitter.com/kakedasiseinen)

## 🐶 Contribution

Contributions are welcome! Please open an issue or submit a pull request.
