"""Script to plot VLSM-results"""
## Description
# This script allows for plotting the results of a VLSM-analysis.


## Imports
import nibabel as nib
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from nilearn.image import load_img
from scipy import ndimage
from nilearn import plotting, datasets, surface
import os
from bisect import bisect_left



## VARIABLES
# dit zijn nog enkele variabelen die we nodig hebben om te plotten (based on PDC):
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
# = a coordinate system for the cortical surface, based on intersubject averaging
# nb 5: low-resolution fsaverage5 mesh (10242 nodes)
# source: https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_surf_fsaverage.html
# mesh: In the context of brain surface data, a mesh refers to a 3D representation of the brain’s surface geometry.
# It is a collection of vertices, edges, and faces that define the shape and structure of the brain’s outer surface.
# Each vertex represents a point in 3D space, and edges connect these vertices to form a network.
# Faces are then created by connecting three or more vertices to form triangles.
# source: https://nilearn.github.io/dev/glossary.html#term-mesh
curv_left = surface.load_surf_data(fsaverage.curv_left)
# = Gifti file, left hemisphere curvature data
curv_left_sign = np.sign(curv_left)
# Returns an element-wise indication of the sign of a number.
# The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
# https://numpy.org/doc/stable/reference/generated/numpy.sign.html


## FUNCTIONS
def check_VLSM_output_by_threshold(cluster_img_path, threshold):
    # Load the NIfTI image using nibabel
    img = nib.load(cluster_img_path)

    # Get the image data as a numpy array
    img_data = img.get_fdata()

    # Get the values of voxels below the threshold
    values_below_threshold = img_data[img_data < threshold]
    values_above_threshold = img_data[img_data > threshold]

    # Count the number of voxels above the threshold
    voxels_above_threshold = np.sum(img_data > threshold)

    # Count the number of voxels below the threshold
    voxels_below_threshold = np.sum(img_data < threshold)

    # Get the minimum and maximum values of the voxels below the threshold
    min_value_below_threshold = np.min(values_below_threshold) if len(values_below_threshold) > 0 else None
    max_value_below_threshold = np.max(values_below_threshold) if len(values_below_threshold) > 0 else None

    # Get the minimum and maximum values of the voxels below the threshold
    min_value_above_threshold = np.min(values_above_threshold) if len(values_above_threshold) > 0 else None
    max_value_above_threshold = np.max(values_above_threshold) if len(values_above_threshold) > 0 else None

    # Get the number of voxels above 0
    voxels_above_zero = np.sum(img_data > 0)

    # output results
    print('voxels_above_threshold', voxels_above_threshold)
    print('voxels_below_threshold', voxels_below_threshold)

    print('min_value_above_threshold', min_value_above_threshold)
    print('max_value_above_threshold', max_value_above_threshold)

    print('min_value_below_threshold', min_value_below_threshold)
    print('max_value_below_threshold', max_value_below_threshold)

    print('voxels_above_zero', voxels_above_zero)

    return (voxels_above_threshold, voxels_below_threshold, min_value_above_threshold, max_value_above_threshold,
            min_value_below_threshold, max_value_below_threshold, voxels_above_zero)


def plot_VLSM_cluster_new(cluster_img_path, zthreshold=1.645

                          ):
    # Load the data
    cluster_img = nib.load(cluster_img_path)
    # cluster_data = img.get_fdata()  # negeer oranje stippellijn

    # Load fsaverage surface data (can be customized depending on your study)
    fsaverage = datasets.fetch_surf_fsaverage()

    # Convert the NIfTI image to a surface map (we will use the pial surface for this example)
    texture = surface.vol_to_surf(img=cluster_img, surf_mesh=fsaverage.pial_left)

    # Ensure colorbar shows the real z-values by determining the minimum and maximum z-values in your cluster_img
    z_values = cluster_img.get_fdata()
    z_min = np.min(z_values)
    z_max = np.max(z_values)

    # Plot the surface map
    figure = plotting.plot_surf_stat_map(surf_mesh=fsaverage.infl_left,
                                         # Use the inflated mesh for better visualization
                                         stat_map=texture,  # The data mapped on the surface
                                         hemi='left',  # Left hemisphere
                                         title='Cluster Image on Left Hemisphere',  # Title of the plot
                                         symmetric_cbar=False,  # Colorbar is not symmetric around 0
                                         colorbar=True,  # Display colorbar
                                         threshold= zthreshold,  # Apply threshold for better visibility (z-threshold)
                                         cmap = 'Greens',
                                         # cmap='twilight_shifted',  # A perceptually uniform colormap
                                         bg_map=fsaverage.sulc_left,  # Use sulcal depth as background map
                                         vmin = zthreshold - 0.001,
                                         # vmin=z_min,  # Set minimum value for the color scale (real z-values)
                                         vmax=z_max)  # Set maximum value for the color scale (real z-values)

    # Optionally save the figure
    figure.savefig(
        f"L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/VLSM_factored_permTest_5000_Factor_3_option2.svg")
    # from plt.savefig(
    #         os.path.join(output_dir, "figures", f"feature_importances_{label}_{interview_part}.png"), dpi = 300)
    plotting.show()


def plot_VLSM_cluster_axial(cluster_img_path,zthreshold=1.645
                            ):

    # Load the data
    cluster_img = nib.load(cluster_img_path)
    # cluster_data = img.get_fdata()  # negeer oranje stippellijn

    ## Plot met tresholds
    plotting.plot_roi(cluster_img, cut_coords=(-20, -4, 12, 20, 28, 36),
                      display_mode='z', colorbar=True, cmap='Greens',threshold= zthreshold-0.001,vmax= 5) # threshold=zthreshold)

    ## Save figure
    plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/VLSM_factored_permTest_5000_Factor_3_axial.png")
    plotting.show()


def plot_VLSM_cluster(cluster_img_path,

):
    """ Plot the surviving/largest cluster from a univariate VLSM-analysis for a certain behavioural variable.
    The plot will show the cluster (calculated based on the cluster threshold), and if correct is not None, it will
    show the cluster for all z-values larger than the z threshold.
    """
    ## Initialiseer variabelen
    img = nib.load(cluster_img_path)
    cluster_data = img.get_fdata()  # negeer oranje stippellijn
    plot_threshold = 0.001

    # Make a copy of the data (if needed, for manipulating negative values)
    cluster_data = cluster_data * 1  # -1 indien interesse in negatieve z-waarden
    cluster_img = nib.Nifti1Image(cluster_data, img.affine)  # maak er opnieuw Nifti-image van

    # Convert volume data to surface texture (Extract surface data from a Nifti image.)
    texture = surface.vol_to_surf(
        img = cluster_img,
        surf_mesh = fsaverage.pial_left,
        # file containing surface mesh geometry: fsaverage.pial_left = Gifti file, left hemisphere pial surface mesh
    )

    # Ensure that texture values below the zthreshold are also set to zero
    # if correct is not None:
        # texture[texture < zthreshold] = 0
    # why: ChatGPT:
    # It's also possible that the thresholding operation (cluster_data[cluster_data < zthreshold] = 0) works correctly in the volume data,
    # but when mapping to the surface (texture = surface.vol_to_surf(cluster_img, fsaverage.pial_left)), some values are interpolated or rounded,
    # and values slightly below the threshold might appear.
    #
    # Solution: Consider applying a threshold directly to the surface data (after it's been mapped), in case any residual values remain due to interpolation.

    # Plot the surface map
    figure = plotting.plot_surf_stat_map(surf_mesh = fsaverage.infl_left,
                                         # =  refers to the left hemisphere of the brain's inflated surface mesh (a common anatomical reference surface in neuroimaging) from the fsaverage subject in FreeSurfer.
                                         # likely contains the coordinates of the left hemisphere's inflated surface, which is used to plot the statistical map on the brain's cortical surface.
                                         stat_map = texture,
                                         # The texture here represents the statistical data (such as t-values, p-values, z-scores, or any other map) that is going to be visualized on the brain surface.
                                         # This data is mapped to each vertex (point) of the surface, determining the "color" or intensity of the statistical map at each point on the brain's surface.
                                         hemi='left',
                                         title='Surface left hemisphere',
                                         symmetric_cbar=False,
                                         colorbar=True,
                                         threshold=plot_threshold,
                                         cmap='twilight_shifted',
                                         bg_map=fsaverage.sulc_left,
                                         )
    figure.savefig(
        f"L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/VLSM_factored_permTest_5000_Factor_4_option2.svg")
    # from plt.savefig(
    #         os.path.join(output_dir, "figures", f"feature_importances_{label}_{interview_part}.png"), dpi = 300)
    plotting.show()

    return


if __name__ == "__main__":
    # Define the file paths for the lesion mask and atlas image
    ## Initialize some variables
    # TODO: Vul dit zelf aan
    cluster_img_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/output/VLSM_factored_withMonthsPO_perm_5000_lesionregr_MCcorrected/surviving_clusters_Factor_3.nii"
        # "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/maps/sub-01.nii"
    # Path to cluster img (nifti-file), make sure to use / instead of \; and add .nii extension
    threshold = 1.645

    # Get the voxel counts
    check_VLSM_output_by_threshold(cluster_img_path, threshold)

    # Output the result
    """print(f"Number of voxels above the threshold: {above}")
    print(f"Number of voxels below the threshold: {below}")
    print(f"Minimum value of voxels below the threshold: {min_below}")
    print(f"Maximum value of voxels below the threshold: {max_below}")
    print(f"Number of voxels above zero: {above_zero}")"""

    # plot_VLSM_cluster_new(cluster_img_path)
    # plot_VLSM_cluster_axial(cluster_img_path)
    # plot_VLSM_cluster(cluster_img_path)


