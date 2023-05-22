# BID-Tucker
How can we efficiently store and mine dynamically generated dense tensor data? 
Much of the multidimensional dynamic data in the real world is generated in the form of time-growing tensors. 
For example, air quality measurements are tensor data consisting of multiple sensory data over large regions and time. 
Such data, accumulated over time, is redundant and consumes a lot of memory in its raw form. 
We need a way to efficiently store dynamically generated block-wise tensor data, allowing us to analyze them between arbitrary time blocks on demand.
To this end, we propose a Block Incremental Dense Tucker Decomposition (BID-Tucker) method for efficient storage and on-demand analysis of multidimensional spatiotemporal data.  Assuming that tensors come in unit blocks, our proposed BID-Tucker slices the blocks into matrices and decomposes them via singular value decomposition (SVD). The SVDs of the $time \times space$ sliced matrices are stored instead of the raw tensor blocks to save space. 
When data analysis is required at particular time blocks, the SVDs of corresponding time blocks are retrieved and incremented to be used for Tucker decomposition. 
The factor matrices and core tensor of the decomposed results can then be used for further data analysis. 
We compare our proposed method with D-Tucker, which our method extends, and vanilla Tucker decomposition, and show that our method is faster than both D-Tucker and vanilla Tucker decomposition and uses less memory for storage with a comparable reconstruction error.   We applied our proposed method to analyze the spatial and temporal trends of air quality data collected in South Korea from 2018 to 2022. We were able to verify unusual events, such as chronic ozone alerts and large fire events. We were also able to verify spatial trends with similar air quality measurements.


The study was conducted on a 12-core Intel(R) Core(TM) i7-6850K CPU @ 3.60 GHz. 
The following libraries were used with Python version 3.10: NumPy version 1.23.5, scikit-learn version 1.2.1, Pytorch 2.0.0, and Tensorly 0.8.1. 
    
You can run the test of small tensor data by runing test_BID.py. Please set the hyperparameters (Global variable) correctly. 
