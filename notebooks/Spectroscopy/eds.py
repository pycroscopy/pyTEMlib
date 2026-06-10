import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    <font size = "5"> **EDS_Tools:  [Spectroscopy](../4_EELS_Tools.ipynb)** </font>

    <hr style="height:1px;border-top:4px solid #FF8200" />

    # Analysis of EDS Spectrum Images
    <br>

    [<img src=https://www.coeuscreativegroup.com/wp-content/uploads/2020/04/download-button.png, width=125>](https://raw.githubusercontent.com/pycroscopy/pyTEMlib/main/notebooks/Spectroscopy/EDS-SI.ipynb)

    [![OpenInColab](https://colab.research.google.com/assets/colab-badge.svg)](
        https://colab.research.google.com/github/pycroscopy/pyTEMlib/blob/main/notebooks/Spectroscopy/EDS-SI.ipynb)

    part of

    <font size = "5"> **[pyTEMlib](https://pycroscopy.github.io/pyTEMlib/about.html)**</font>

    a [pycroscopy](https://pycroscopy.github.io/pycroscopy/about.html) ecosystem package



    Notebook by Gerd Duscher, 2025

    Microscopy Facilities<br>
    Institute of Advanced Materials & Manufacturing<br>
    The University of Tennessee, Knoxville

    Model based analysis and quantification of data acquired with transmission electron microscopes

    ## Content
    An Introduction into displaying and analyzing EDS spectrum images and spectra
    This works also on Google Colab.

    Unlike in an EELS spectrum, a single pixel in an EDS spectrum image does not contain enough information to quantify the chemical composition or even detect trace elements.

    Several different stategies will be explored in this notebook of how to analyze these kind of data.


    ## Prerequesites

    ### Install pyTEMlib

    If you have not done so in the [Introduction Notebook](_.ipynb), please test and install [pyTEMlib](https://github.com/gduscher/pyTEMlib) and other important packages with the code cell below.
    """)
    return


@app.cell
def _():
    import sys
    import marimo
    import numpy as np
    import matplotlib.pylab as plt
    sys.path.insert(0, '../../')
    import pyTEMlib

    print('pyTEM version: ',pyTEMlib.__version__)
    __notebook__ = 'EDS_Spectrum_Analysis'
    __notebook_version__ = '2026_1_19'
    return marimo, pyTEMlib


@app.cell
def _():
    #file_button = marimo.ui.file(kind="button")
    #file_area = marimo.ui.file(kind="area")
    #marimo.vstack([file_button, file_area])
    return


@app.cell
def _(marimo):
    f = marimo.ui.file(kind="area")
    f
    return


@app.cell
def _(marimo):
    file_browser = marimo.ui.file_browser( multiple=True)
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, pyTEMlib):
    # C:\Users\gduscher\OneDrive - University of Tennessee\google_drive\2022 Experiments\Spectra\20221214\AlCe-200kV
    datasets = pyTEMlib.file_tools.open_file((str(file_browser.path(index=-1))))
    datasets['Channel_000'].plot()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
