"""Script to plot lesion overlap
---------------------------------
This script allows to calculate the lesion overlap between subjects.
More specifically, the following functions are executed:
1) make_lesionOverlapMask: create a lesion-overlap mask (unthresholded for a minimum amount of subjects with overlap).
This function is then used to plot the lesion overlap (a threshold is applied in the plotting functions)
2) threshold_lesionOverlapMask: Threshold a lesion overlap mask to include only voxels with overlap >= threshold.
This threshold function can be used elsewhere, when you want to use a nifti-file of a thresholded lesion overlap 'cluster'
(e.g., to calculate the lesion overlap distribution: use this file in the calculate_cluster_distribution.py script).
In this script, it is not used to plot the lesion overlap, as the thresholding is already included in the plotting functions.
3) plot_lesionOverlap_Zcoord: Plot the lesion overlap among subjects in an axial plot (using Z-coordinates)
4) plot_lesionOverlap_surf: Plot the lesion overlap among subjects in a surface map plot
To run the script, change variable names and paths where prompted (via a TODO statement)

Note: if you get errors, it is often 1) either because of packages, install earlier versions,
or 2) because you have problems with your dimensions of your images not matching across subjects do not match.
Read the manual of the VLSM at the bottom, there is more info included.
"""

import nibabel as nib
import numpy as np
# from nilearn.datasets.data.convert_templates import nifti_image
from scipy import ndimage
from nilearn import plotting, datasets
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import surface



def make_lesionOverlapMask(lesion_files_path,out_path
):
    """
    Create a lesion-overlap mask (unthresholded) and store it in a path

    :param lesion_folder_path: path to folder with lesion files (= subj-XX.nii files)
    :param out_path: output path for the lesion overlap mask file (= .nii file)
    :return: the path to a lesion overlap mask file (= .nii file)
    """
    allFiles = [f for f in os.listdir(lesion_files_path) if f.endswith('.nii')]  # list with all files (= all subjects)
    i = 0 # will be the maximal lesion overlap (sf)
    # Ex: in PDC, the highest sum = 7, so maximal lesion overlap = 7 persons (i = 7); in IANSA i = 49.

    for subject in allFiles:
        path_subject = lesion_files_path + subject # concatenate the path
        img = nib.load(path_subject)
        if i == 0:
            lesionOverlap = np.round(np.array(img.get_fdata()))
        else:
            lesionOverlap = lesionOverlap + np.round(np.array(img.get_fdata()))
        i += 1
        # Note: all individual data are binary files, 3D matrices. To make lesionOverlap, sum all binary files

    print("Shape van individuele data: {0}".format(lesionOverlap.shape))
    print("Unieke waarden: {0}".format(np.unique(lesionOverlap)))
    print("Max lesion overlap: {0}".format(np.max(lesionOverlap)))

    lesionOverlapMask = nib.Nifti1Image(lesionOverlap,
                                        img.affine)
    # affine argument needed by nilearn.
    # that img was loaded in previous for-loop above (so is from 1 participant)but since all images
    # have the same dimensions, that affine is also the same for everyone.
    nib.save(lesionOverlapMask, out_path)

    print(f"Unthresholded lesion overlap mask saved to: {out_path}")
    return out_path


def threshold_lesionOverlapMask(overlap_mask_path, out_path, threshold):
    """
    Threshold a lesion overlap mask to include only voxels with overlap >= threshold.

    :param overlap_mask_path: Path to existing lesion overlap mask (.nii file)
    :param out_path: Output path for thresholded lesion overlap mask (.nii file)
    :param threshold: Minimum number of overlapping lesions to retain a voxel (e.g., 5)
    :return: the path to the thresholded lesion overlap mask (Nifti1Image object)

    Note: this lesion overlap mask does not have the same axes as a typical lesion cluster mask
    How to resample to the atlas axes
    # Load images
    lesion_img = nib.load(thresh_lesionOverlapMask_path)  # your lesion overlap image
    atlas_img = nib.load(
        "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/helper files/harvard_new.nii")

    # Resample lesion image to atlas resolution and space
    lesion_resampled = ni.resample_to_img(lesion_img, atlas_img, interpolation='nearest')
    resampled_thresh_lesionOverlapMask_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/resampled_thresh_lesionOverlap_Mask.nii"
    nib.save(lesion_resampled, resampled_thresh_lesionOverlapMask_path)

    """
    # Load the lesion overlap mask
    img = nib.load(overlap_mask_path)
    data = img.get_fdata()

    # Apply threshold: keep only voxels where at least `threshold` subjects overlap
    # Note: the value of each voxel == the amount of overlapping persons that have a lesion in this voxel.
    thresholded_data = np.where(data >= threshold, data, 0)

    # Save as new NIfTI file
    thresholded_img = nib.Nifti1Image(thresholded_data, img.affine)
    nib.save(thresholded_img, out_path)

    print(f"Thresholded lesion overlap mask saved to: {out_path}")
    print(f"Voxels kept (≥{threshold}): {np.count_nonzero(thresholded_data)}")

    return out_path


def plot_lesionOverlap_Zcoord(unthresholded_overlap_mask_path,threshold, out_path
):
    """
    Plot the lesion overlap among subjects in an axial plot (using Z-coordinates)
    :param unthresholded_overlap_mask_path: the original unthresholded lesion overlap mask (nifti-file) PATH, based on the function make_lesionOverlapMask
    :param threshold: the minimum amount of subjects (absolute number) that should have overlapping lesions
    to be shown as overlap in the plot
        # Ex threshold = 1: from lesion overlap of 1 person (in essence lesion of each individual), it will be shown on the plot
        # Ex threshold = 2: from lesion overlap between 2 persons, it will be shown on the plot
        (in essence, lesions that only occur in 1 person will NOT be shown)
    :param out_path: the output path for the lesion overlap PLOT
    :return: a lesion overlap plot, in the form of Zcoord layer figures (side-by-side) (this can also easily be made
    using MRIcroGL manually, without this code)

    Source: https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_roi.html
    """

    # Get the leion Overlap Mask (nifti)
    img = nib.load(unthresholded_overlap_mask_path)

    # Plot the lesionOverlapMask
    plotting.plot_roi(img,
                      cut_coords=(-20, -4, 12, 20, 28, 36),
                      display_mode='z',
                      colorbar=True,
                      cmap='cubehelix',
                      threshold = int(threshold))

    # Save the figure
    plt.savefig(out_path)
    print(f"lesion overlap Zcoord plot saved to: {out_path}")
    print(f"Only voxels (≥{threshold}) are shown")
    plotting.show()

    return out_path


def plot_lesionOverlap_surfMap(unthresholded_overlap_mask_path, threshold, out_path
):
    """
    Plot the lesion overlap among subjects in a surface map plot
    :param unthresholded_overlap_mask_path: the original, unthresholded lesion overlap mask (nifti-file) PATH,
    based on the function make_lesionOverlapMask
    :param threshold: the minimum amount of subjects (absolute number) that should have overlapping lesions
    to be shown as overlap in the plot
        # Ex threshold = 1: from lesion overlap of 1 person (in essence lesion of each individual), it will be shown on the plot
        # Ex threshold = 2: from lesion overlap between 2 persons, it will be shown on the plot
        (in essence, lesions that only occur in 1 person will NOT be shown)
    :param out_path: the output path for the lesion overlap PLOT
    :return: a lesion overlap plot, in the form of a Surface Map

    Source: suggested by PDC, cannot be done (easily) in MRIcroGL
    """
    # Get the leion Overlap Mask (nifti)
    img = nib.load(unthresholded_overlap_mask_path)

    # texture background is needed for a surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    texture = surface.vol_to_surf(img, fsaverage.pial_left)

    # Plot the surface map
    figure = plotting.plot_surf_roi(fsaverage.infl_left,
                                    texture,
                                    hemi='left',
                                    title='Surface left hemisphere',
                                    colorbar=True,
                                    threshold=int(threshold),
                                    bg_map=fsaverage.sulc_left,
                                    cmap='cubehelix',
                                    vmax=20)

    plt.savefig(out_path)
    print(f"lesion overlap Zcoord plot saved to: {out_path}")
    print(f"Only voxels (≥{threshold}) are shown")

    plotting.show()

    return out_path



if __name__ == "__main__":
    ### Preparations
    # ----------------
    # TODO: change paths
    path_to_lesion_files = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/maps/"
    # "D:/PhD PDC/voorbeeldscripts_info/voorbeeldData_lesion_overlap_map/" # hierin staat lijstje van letsel-files (lesion extracted files per subject eg sub-060.nii)
    # Don't forget : / instead of \ ; and / in the end of the path name
    allFiles = [f for f in os.listdir(path_to_lesion_files) if f.endswith('.nii')]  # lijst alle files (=alle proefpersonen)
    out_path_unthresholded_lesionOverlapMask = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_Mask.nii"
    out_path_lesionOverlapPlot_Z_thresh0 = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_noT.svg"
    out_path_lesionOverlapPlot_Z_thresh2 = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_Tresh2.svg"
    out_path_lesionOverlapPlot_Z_thresh5 = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_Tresh5.svg"

    out_path_lesionOverlapPlot_surf_thresh0 = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_surf_noT.svg"
    out_path_lesionOverlapPlot_surf_thresh2 = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_surf_Tresh2.svg"
    out_path_lesionOverlapPlot_surf_thresh5 = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_surf_Tresh5.svg"


    out_path_thresh_lesionOverlapMask = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/thresh_lesionOverlap_Mask.nii"


    ### Run the functions
    # ------------
    unthresh_lesion_overlap_mask_path = make_lesionOverlapMask(lesion_files_path = path_to_lesion_files,
                                                               out_path=out_path_unthresholded_lesionOverlapMask,
                                                               )
    atlas_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/helper files/harvard_new.nii"
    thresh_lesion_overlap_mask_path = threshold_lesionOverlapMask(unthresh_lesion_overlap_mask_path,
                                                                  out_path=out_path_thresh_lesionOverlapMask,
                                                                  threshold = 5, # TODO: change threshold
                                                                  )
    # Note: this latter function will only be used ELSEWHERE, nl only in lesion_distribution_calculation.py
    # to calculate which regions lie in the overlap region.

    # TODO: change threshold en out_path (see above)

    ## Z-coord plots
    # Plot without thresholds (= plot of overlap between ALL subjects, without limitation of 'minimal overlap')
    plot_lesionOverlap_Zcoord(unthresh_lesion_overlap_mask_path,
                              threshold = 0,
                              out_path = out_path_lesionOverlapPlot_Z_thresh0
                              )
    
    # Plot threshold = 2
    plot_lesionOverlap_Zcoord(unthresh_lesion_overlap_mask_path,
                              threshold=2,
                              out_path=out_path_lesionOverlapPlot_Z_thresh2
                              )
    # Plot threshold = 5
    # IANSA: VLSM analysis minimal_lesion_overlap = 10%
    # => absolute amount = 5 (van n tot = 50): choose threshold = 5: from 5 persons overlap, this is shown in the plot
    plot_lesionOverlap_Zcoord(unthresh_lesion_overlap_mask_path,
                              threshold=5,
                              out_path=out_path_lesionOverlapPlot_Z_thresh5
                              )

    ## Surface map plots
    # Plot without thresholds
    plot_lesionOverlap_surfMap(unthresh_lesion_overlap_mask_path,
                               threshold = 0,
                               out_path=out_path_lesionOverlapPlot_surf_thresh0
                               )
    
    # Plot with threshold = 2
    plot_lesionOverlap_surfMap(unthresh_lesion_overlap_mask_path,
                               threshold=2,
                               out_path=out_path_lesionOverlapPlot_surf_thresh2
                               )
    # Plot with threshold = 5
    plot_lesionOverlap_surfMap(unthresh_lesion_overlap_mask_path,
                               threshold=5,
                               out_path=out_path_lesionOverlapPlot_surf_thresh5
                               )

