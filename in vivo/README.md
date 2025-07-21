The Z-stack TIFF files obtained through microscopy were first processed using thresholdingframe.py to apply intensity thresholding and select a representative Alx647 channel frame, along with its corresponding GFP channel frame.

Subsequently, the selected Alx647 frame was subjected to morphological erosion using erosionmask.py to refine the donor cell regions.

To quantify how far extracellular vesicles (EVs) travel from the donor cells, spatial distance measurements were performed using distTraAnalysis.py on the erosionmask alx647 files. Background signal analysis was conducted using backgroundanalysis.py, which was applied to the GFP frames representing recipient cells.
