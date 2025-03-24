<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true" width="70%">
  </a>
</p>


# Project #26: Ultrastructural protein mapping through Correlation of Light and Electron Microscopy


<!-- [![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/) -->


---
This repo was created by the [AI4Life project](https://ai4life.eurobioimaging.eu) using data provided by Jan van der Beek at Center for Molecular Medicine, Utrecht.
All the images demonstrated in this tutorial are provided under **CC-BY** licence.

If any of the instructions are not working, please [open an issue](https://github.com/ai4life-opencalls/oc_2_project_26/issues) or contact us at [ai4life@fht.org](ai4life@fht.org)!


# Introduction
The project aims to achieve an accurate automatic correlation of light and electron microscopy images. The combination of multi-channel protein labeling through fluorescence microscopy with nm-scale ultrastructural morphology from electron microscopy can provide both molecular and ultrastructural information of endo-lysosomal alternations at high resolution.

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

- [Detection of fiducial particles in EM](notebooks/Detect_fiducial_particles_in_EM.ipynb)
- [Detection of fiducial particles in LM](notebooks/Detect_fiducial_particles_in_LM.ipynb)
  

## Acknowledgements
AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.


## Contact

[SciLifeLab BioImage Informatics Facility (BIIF)](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Kristína Lidayová](mailto:kristina.lidayova@scilifelab.se)
