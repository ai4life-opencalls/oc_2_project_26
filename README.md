  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true" width="70%">
  </a>
</p>


# Project #26: Ultrastructural protein mapping through Correlation of Light and Electron Microscopy


<!-- [![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/) -->


---
This repository contains code for the automatic correlation of light microscopy (LM) and electron microscopy (EM) images. Developed as part of the [AI4Life project](https://ai4life.eurobioimaging.eu), it uses data provided by Jan van der Beek from the Center for Molecular Medicine, Utrecht.
All images used in this tutorial are licensed under **CC-BY**. If any of the instructions are not working, please [open an issue](https://github.com/ai4life-opencalls/oc_2_project_26/issues) or contact us at [ai4life@fht.org](ai4life@fht.org)!

## Introduction
The project focuses on accurately aligning LM and EM images to combine molecular and ultrastructural information for high-resolution analysis of endo-lysosomal alterations. LM provides multi-channel protein labeling through fluorescence microscopy, while EM offers nanoscale morphological details. Aligning these images enables comprehensive insights into cellular structures.

_Figure 1: Example of an electron microscopy (EM) image (left) and the fiducial marker channel from a light microscopy (LM) image (right), both showing an enlarged cutout of the same corresponding region._

![EM_LM_image](https://github.com/user-attachments/assets/83107107-5ded-4100-8d6e-2a032c11db68)

Before the project start, the image alignment relied on manually selecting points of interest and using ec-CLEM software to detect the correlation between them. This was a time-consuming and non-reproducible approach. This project instead aims to automate the process by detecting the reference features in both image modalities automatically.

For choosing the reference features, nuclei would be ideal candidates. They are relatively easy to detect in DAPI channel of LM images using [CellPose](https://www.cellpose.org/), however, they were impossible to detect in EM images. Methods like [SAM](https://segment-anything.com/), [Micro-SAM](https://github.com/computational-cell-analytics/micro-sam), [ConvPaint](https://github.com/guiwitz/napari-convpaint), or traditional segmentation techniques did not yield good results for EM images. As a solution, we decided to use fiducial particles as reference markers.

_Figure 2: Light microscopy image on the right, with individual channels shown on the left. The blue channel represents nuclei stained with DAPI, and the red channel corresponds to fiducial markers._

![LM_image](https://github.com/user-attachments/assets/0ec79835-3a9c-4063-89e5-6ce2a720e111)

The fiducial particles in the LM image were detected using the Big-FISH Python package for spot detection. Corresponding spots in EM images were located using template matching. Then the Probreg package was used for point cloud registration and finally the LM image was warped based on the resulting displacement field. This workflow enables robust, fully automatic registration of LM and EM images for integrated biological analysis.


## Installation
Install the [conda](https://conda.io) package, dependency and environment manager.

You can download this repository from the green `Code` button → download ZIP, or clone through the command line with

    cd <path to any folder of choice>
    git clone https://github.com/BIIFSweden/AI4Life_OC2_2024_26.git

Then create the `AI4Life_OC2_2024_26` conda environment:

    cd <path to your 'AI4Life_OC2_2024_26' directory>
    conda env create -f environment.yml

This will install all necessary project dependencies.

## Usage
Copy all project data to the [data](data) directory (or use symbolic links).

Then run [Jupyter Lab](https://jupyter.org) from within the `AI4Life_OC2_2024_26` conda environment:

    cd <path to your 'AI4Life_OC2_2024_26' directory>
    conda activate AI4Life_OC2_2024_26
    jupyter-lab

Inside the `notebooks` folder you will find Jupyter notebooks for:

### Step 1 : [Detecting the fiducial particles in EM images](notebooks/Detect_fiducial_particles_in_EM.ipynb)

  This notebook detects fiducial particles in electron microscopy (EM) images for use in CLEM workflows. The fiducial particles in EM image are characterized by bright rings with a dark center and can be detected using template matching.
The pipeline consists of:

1. **Fiducial particle detection**: Detection of fiducial particles is done using the template matching algorithm with an artificial template (dark-centered spot).
2. **Cluster detection**: Filtering the set of individual fiducial particles by recognizing clusters of overlapping or closely located detections (≥3) and saving their centroids.
3. **Results saving**: Saving the positions of all detected fiducial particles and the positions of fiducial clusters into files in multiple formats (Pandas DataFrame, XML, PLY) for downstream analysis. 

_Figure 3: Electron microscopy (EM) image with an enlarged cutout (left), the fiducial particle detection (result after step 1, middle), and the fiducial clusters detection (result after step 2, right)._

![EM-FP_detection](https://github.com/user-attachments/assets/c674ea56-de9f-4846-929e-cf53ed7f637d)

### Step 2 : [Detecting the fiducial particles in LM images](notebooks/Detect_fiducial_particles_in_LM.ipynb)

  This notebook detects fiducial particles in light microscopy (LM) images for use in CLEM workflows. The fiducial particles in LM image are characterized by bright spots and can be detected using the Big-FISH Python package originally developed for smFISH image analysis.
The pipeline consists of:

1. **Spot detection** - Detection of fiducial particles using spot detection methods from Big-FISH package. It identifies individual regions of fiducial particles as local maxima in the LM image using filtering, thresholding, and detection techniques.
2. **Dense region decomposition** - Recognizing larger and brighter (dense) spots as clusters and decomposing them into individual fiducial particles using Gaussian modeling and simulation.
3. **Cluster detection** - Groups closely located spots into clusters based on spatial proximity and connectivity.
4. **Results saving** - Saving the positions of all detected fiducial particles and the positions of fiducial clusters into files in multiple formats (Pandas DataFrame, XML, PLY) after upscaling the LM image to match EM image dimensions.

_Figure 4: Light microscopy (LM) image (left), the fiducial particle regions detection (result after step 1, middle), and the fiducial particles [red circles] and clusters [green circles] detection (result after step 3 and 4, right)._

![LM-FP_detection1](https://github.com/user-attachments/assets/735a0788-e19d-415b-9739-7ecd30676f4e)

### Step 3 : [Finding correlation between EM and LM images](notebooks/Finding_correlation_between_EM_and_LM_images.ipynb)
  
  This notebook performs automated registration of fiducial particles detected in electron microscopy (EM) and light microscopy (LM) images. The goal is to align LM images to EM images using matched fiducial landmarks.
The registration workflow consist of:

1. **Loading fiducial locations** - Fiducial coordinates are imported from an .xml file for both EM (target) and LM (source) images.
2. **Rescaling LM coordinates** - The LM fiducial coordinates are rescaled to match the coordinate system of the EM image.
3. **Multilevel registration** - The algorithm performs a two-stage Coherent Point Drift (CPD) registration: first rigid, then non-rigid. It automatically identifies corresponding fiducial particles between the two modalities and assigns them matching IDs.
4. **Warping the LM image** - The LM image is warped to correlate with the EM reference image using a displacement field computed from matched fiducial points.

NOTE: An updated .xml file with matched point IDs can also be used directly in ec-CLEM software for image overlay without manual alignment.

_Figure 5: Results of the registration algorithm. Warped LM image (result after step 4) with matched fiducial particles from EM image as red circles on top (result after step 3, left), the displacement field (result after step 4) with matched fiducial particles from EM image (middle), and EM image overlaid by a semitransparent warped LM image (DAPI channel and fiducial marker channel as two separate images, right)._

![EM_overlaid_by_LM](https://github.com/user-attachments/assets/9b6c647f-ce74-4dcd-b017-d612ad2d5bc9)


## Acknowledgements
AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.


Developed by [Kristína Lidayová](mailto:kristina.lidayova@scilifelab.se)
