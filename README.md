# A2_316

Use Python 3.9

Check the files to see which libraries we used.

```
usage: fft.py [-h] [-m MODE] [-i IMAGE]

optional arguments:
  -h, --help  show this help message and exit
  -m MODE
    - [1] (Default) for fast mode where the image is converted into its FFT form and displayed
    - [2] for denoising where the image is denoised by applying an FFT, truncating high
    frequencies and then displayed
    - [3] for compressing and saving the image
    - [4] for plotting the runtime graphs for the report
    - [5] (Extra mode) for testing if our output is the same as numpy's fft methods

  -i IMAGE    filename of the image we wish to take the DFT of (default: moonlanding.png)
```

Google doc of our report: https://docs.google.com/document/d/1E9MfSqbCLaHIBlYxmepxhwBoUhXF76diD7jGC6suzQc/edit?usp=sharing