# MDE-SpikingCamera
 Codes and Datasets of "Spike Transformer: Monocular Depth Estimation for Spiking Camera"

ECCV 2022


## Code

* Inference the pre-trained model:
    
    To inference the network using our pre-trained model, firstly you need to download the pre-trained weights file using the link [https://drive.google.com/file/d/1_YMVTe-egeOKk16GMVBwjX8Sk3wdrd-E/view?usp=sharing](https://drive.google.com/file/d/1_YMVTe-egeOKk16GMVBwjX8Sk3wdrd-E/view?usp=sharing), unzip it and put this folder under the `s2d_weights/` folder.

    Then, you need to run the `test_DENSE.py` script, `'--path_to_model'` is the model path that you want to evaluate, `'--output_path'` is the folder that saves all the testing results including files with .npy and .png formats ( these files are used in the `evaluation_DENSE.py` script), and you alse need to set the `'--data_folder'` to set the root folder of the testing dataset.

    For example:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_DENSE.py --path_to_model .../model_best.pth.tar --output_path ... --data_folder .../DENSE/test/
    ```

* Environments:

    To help successfully run the code, the esstential environment on Linux is included in the `requirements.txt`.

* More Details for Training and Evaluating
    
    Please refer to *README_CODE.md* for more details of **training** and **evaluating**.

## Dataset

In this work, we propose two datasets. The description are as followed: 

* Synthetic Dataset: *DENSE-Spike*

    The Synthetic spike dataset named "DENSE-Spike" is generated based on the [DENSE](https://rpg.ifi.uzh.ch/E2DEPTH.html) dataset using the method described in our paper. The spike data is available and you can download them at [https://pan.baidu.com/s/1Lg2spMW4OYlsYy0iT4u_7g](https://pan.baidu.com/s/1Lg2spMW4OYlsYy0iT4u_7g) with the password **1008**.

* Real Dataset: *Outdoor-Spike*

    The Real Dataset named "Ourdoor-Spike" contains 33 sequences of outdoor scenes are captured in a driving car from the first perspective. Each sequence contains 20000 spike frames. The spike data is available and you can download them at [https://pan.baidu.com/s/1hji5GnFH5Ke_nDt-1Q76rg](https://pan.baidu.com/s/1hji5GnFH5Ke_nDt-1Q76rg) with the password **1997**. Every sequence is stored with a single file with *".dat"* format.