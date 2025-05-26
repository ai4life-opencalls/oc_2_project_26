<p align="center">
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

Initially, image alignment relied on manually selecting points of interest and using ec-CLEM software to detect the correlation between them. This was a time-consuming and non-reproducible approach. This project instead aims to automate the process by detecting the reference features in both image modalities automatically.

Nuclei would be ideal reference features. They were relatively easy to detect in LM images using [CellPose](https://www.cellpose.org/), however, they were impossible to detect in EM images. Methods like [SAM](https://segment-anything.com/), [Micro-SAM](https://github.com/computational-cell-analytics/micro-sam), [ConvPaint](https://github.com/guiwitz/napari-convpaint), or traditional segmentation techniques did not yield good results for EM. As a solution, the team had to use fiducial particles as reference markers.

The fiducial particles in the LM image were detected using the Big-FISH Python package for spot detection. Cooresponding spots in EM images were located using template matching. Then the Probreg package was used for point cloud registration and finally the LM image was warped based on the resulting displacement. This workflow enables robust, fully automatic registration of LM and EM images for integrated biological analysis.


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

Inside the `notebooks` folder you will find notebooks for:

- [Detecting the fiducial particles in EM images](notebooks/Detect_fiducial_particles_in_EM.ipynb)
- [Detecting the fiducial particles in LM images](notebooks/Detect_fiducial_particles_in_LM.ipynb)
  

## Acknowledgements
AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.


## Contact

[SciLifeLab BioImage Informatics Facility (BIIF)](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Kristína Lidayová](mailto:kristina.lidayova@scilifelab.se)
