"""Script to plot VLSM-results
------------------------------
This script allows for plotting the results of a VLSM-analysis.
1) check_VLSM_output_by_threshold: Check the VLSM-analysis output by threshold (returns text file and tuple)
2) plot_VLSM_cluster_surfMap: Plot the VLSM output cluster on a surface map
3) plot_VLSM_cluster_axial: Plot the VLSM output cluster on an axial plot (using Z-coordinates)

To run the script, change variable names and paths where prompted (via a TODO statement)

# NOTE: all VLSM output is with ABSOLUTE z-values !!! so no negative z-values !!!
"""
from contextlib import redirect_stdout

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
## ---------
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
## ---------
def check_VLSM_output_by_threshold(cluster_img_path, tables_dir, variable, table_type, threshold = 1.645
                                   ):
    """
    Check the VLSM-analysis output by threshold (returns text file and tuple)
    :param cluster_img_path: the path to the cluster image (aka the output of the VLSM-analysis, corrected for Multiple
    Comparisons if necessary)
    :param tables_dir: the path to the directory containing the tables in which the VLSM output is stored
    :param variable: which variable is analysed
    :param table_type: whether the results are significant or non-significant
    :param threshold: the corrected z-threshold (corresponding to the corrected p-threshold), defaults to 1.645
    :return: a tuple containing the following parameters:
    (voxels_above_threshold, voxels_below_threshold, min_value_above_threshold, max_value_above_threshold,
            min_value_below_threshold, max_value_below_threshold, voxels_above_zero, voxels_below_zero)
            and a text file stored in the tables_dir --> log folder
    """
    # Load the NIfTI image using nibabel
    img = nib.load(cluster_img_path)

    # Get the image data as a numpy array
    img_data = img.get_fdata()

    img = nib.Nifti1Image(img_data, img.affine)
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
    voxels_below_zero = np.sum(img_data < 0)

    # Create path to text_log_dir
    output_folder = os.path.join(tables_dir, "logs")
    os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist
    log_path = os.path.join(output_folder, f"log_VLSM_output_{variable}_{table_type}.txt")  # define log file path

    # Open the log file and redirect print output
    with open(log_path, 'w') as log_file:
        with redirect_stdout(log_file):

            # output results
            print('voxels_above_threshold', voxels_above_threshold)
            print('voxels_below_threshold', voxels_below_threshold)

            print('min_value_above_threshold', min_value_above_threshold)
            print('max_value_above_threshold', max_value_above_threshold)

            print('min_value_below_threshold', min_value_below_threshold)
            print('max_value_below_threshold', max_value_below_threshold)

            print('voxels_above_zero', voxels_above_zero)
            print('voxels_below_zero', voxels_below_zero)

    return (voxels_above_threshold, voxels_below_threshold, min_value_above_threshold, max_value_above_threshold,
            min_value_below_threshold, max_value_below_threshold, voxels_above_zero, voxels_below_zero)


def plot_VLSM_cluster_surfMap(cluster_img_path, colour, variable, plot_type, zthreshold=1.645

                          ):
    """
    Plot the VLSM output cluster on a surface map
    :param cluster_img_path: the path to the VLSM cluster (nii-file)
    :param colour: the colour of the surface map
    :param variable: which variable is analysed
    :param plot_type: whether the plot is significant or not
    :param zthreshold: the threshold used to perform the VLSM analysis (corrected threshold, z-value)
    :return: the plot of the VLSM output cluster
    """
    # Load the data
    cluster_img = nib.load(cluster_img_path)

    # Load fsaverage surface data (can be customized depending on your study)
    fsaverage = datasets.fetch_surf_fsaverage()

    # Convert the NIfTI image to a surface map (we will use the pial surface for this example)
    texture = surface.vol_to_surf(img=cluster_img, surf_mesh=fsaverage.pial_left)

    # Ensure colorbar shows the real z-values by determining the minimum and maximum z-values in your cluster_img
    z_values = cluster_img.get_fdata()
    z_min = np.min(z_values)
    z_max = np.max(z_values)


    plot_threshold = zthreshold
    plot_max = z_max
    plot_min = zthreshold - 0.001

    # Plot the surface map
    figure = plotting.plot_surf_stat_map(surf_mesh=fsaverage.infl_left,
                                         # Use the inflated mesh for better visualization
                                         stat_map=texture,  # The data mapped on the surface
                                         hemi='left',  # Left hemisphere
                                         title='Cluster Image on Left Hemisphere',  # Title of the plot
                                         symmetric_cbar=False,  # Colorbar is not symmetric around 0
                                         colorbar=True,  # Display colorbar
                                         threshold= plot_threshold,  # Apply threshold for better visibility (z-threshold)
                                         cmap = colour,
                                         # cmap='twilight_shifted',  # A perceptually uniform colormap
                                         bg_map=fsaverage.sulc_left,  # Use sulcal depth as background map
                                         vmin = plot_min,
                                         # vmin=z_min,  # Set minimum value for the color scale (real z-values)
                                         vmax=plot_max)  # Set maximum value for the color scale (real z-values)

    # Optionally save the figure
    figure.savefig(os.path.join(save_plot_path, f"{variable}_{plot_type}_surf_threshview.png"))
    plotting.show()


def plot_VLSM_cluster_axial(cluster_img_path, colour, variable, plot_type, zthreshold=1.645
                            ):
    """
    Plot the VLSM output cluster on an axial plot (using Z-coordinates)
    :param cluster_img_path: the path to the VLSM cluster (nii-file)
    :param colour: the colour of the surface map
    :param variable: which variable is analysed
    :param plot_type: whether the plot is significant or not
    :param zthreshold: the threshold used to perform the VLSM analysis (corrected threshold, z-value)
    :return: the plot of the VLSM output cluster
    """
    # Load the data
    cluster_img = nib.load(cluster_img_path)
    # cluster_data = img.get_fdata()  # negeer oranje stippellijn

    # Ensure colorbar shows the real z-values by determining the minimum and maximum z-values in your cluster_img
    z_values = cluster_img.get_fdata()
    # switch datatype if necessary
    zmax = round(np.max(z_values))

    ## Plot with tresholds
    plotting.plot_roi(cluster_img, cut_coords=(-20, -4, 12, 20, 28, 36),
                      display_mode='z', colorbar=True, cmap= colour,threshold= zthreshold-0.001,vmax= zmax) # threshold=zthreshold)

    ## Save figure
    plt.savefig(os.path.join(save_plot_path, f"{variable}_{plot_type}_axial_threshview.png"))
    plotting.show()




if __name__ == "__main__":
    ## Initialize some variables
    ## -------------------------
    # TODO: Change this yourself
    variable = "variable_of_interest"
    variable_type = "Lexical"  # choices: see below
    cluster_is_significant = True  # switch to false if not sign cluster
    path_to_VLSM_folder = "path_to_your_VLSM_folder"
    tables_dir = os.path.join(path_to_VLSM_folder, 'tables')
    corrected_VLSM_output_folder_name = "path_to_NiiStat_VLSM_output_+_MCcorrected"
    threshold_abs = 1.645

    # DO NOT CHANGE THIS
    if cluster_is_significant:
        cluster_type = "surviving_clusters"
        plot_type = 'sign'
        table_type = 'sign'
    else:
        cluster_type = "nonsign_cluster"
        plot_type = 'nonsign'
        table_type = 'nonsign'

    if variable_type == "Semantics":
        colour = 'Oranges'
    elif variable_type == "Phonology":
        colour = 'Blues'
    elif variable_type == "Lexical":
        colour = "Greens"
    elif variable_type == "Grammatical":
        colour = "Reds"
    elif variable_type == "Macrostructure":
        colour = "Purples"
    elif variable_type == "Fluency":
        colour = "YlOrBr"  # yellows doesn't exist
    # source: https://matplotlib.org/stable/users/explain/colors/colormaps.html

    cluster_img_path = os.path.join(path_to_VLSM_folder, 'output', corrected_VLSM_output_folder_name, f"Z{cluster_type}_{variable}.nii")
    # Path to cluster img (nifti-file), make sure to use / instead of \; and add .nii extension
    save_plot_path = os.path.join(path_to_VLSM_folder, 'figures')

    ## Functions
    ## ---------

    # Get the voxel counts
    check_VLSM_output_by_threshold(cluster_img_path, tables_dir, variable, table_type, threshold_abs)

    # Plot the clusters
    plot_VLSM_cluster_surfMap(cluster_img_path, colour, variable, plot_type, threshold_abs)
    plot_VLSM_cluster_axial(cluster_img_path, colour, variable, plot_type, threshold_abs)

