"""Script to plot VLSM-results"""
## Description
# This script allows for plotting the results of a VLSM-analysis.


## Imports
import nibabel as nib
import numpy as np
import pandas as pd
import os
from scipy import ndimage
from nilearn import plotting, datasets, surface
import os
from bisect import bisect_left



## VARIABLES
# dit zijn nog enkele variabelen die we nodig hebben om te plotten (based on PDC):
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
curv_left = surface.load_surf_data(fsaverage.curv_left)
curv_left_sign = np.sign(curv_left)


## FUNCTIONS
def plot_VLSM_cluster(cluster_img_path, zthreshold = None

):
    """ Plot the surviving/largest cluster from a univariate VLSM-analysis for a certain behavioural variable.
    The plot will show the cluster (calculated based on the cluster threshold), and if zthreshold is not None, it will
    show the cluster for all z-values larger than the z threshold.
    """
    ## Initialiseer variabelen
    img = nib.load(cluster_img_path)
    cluster_data = img.get_fdata()  # negeer oranje stippellijn
    cluster_data = cluster_data * 1 # -1 indien interesse in negatieve z-waarden
    name_add = "unthresholded_z"
    if zthreshold is not None:
        cluster_data[cluster_data < zthreshold] = 0
        name_add = "thresholded_z"
    cluster_data = nib.Nifti1Image(cluster_data, img.affine)  # maak er opnieuw Nifti-image van
    texture = surface.vol_to_surf(cluster_data, fsaverage.pial_left)

    figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                         texture, hemi='left',
                                         title='Surface left hemisphere', symmetric_cbar=False,
                                         colorbar=True, threshold=0.001, cmap='twilight_shifted',
                                         bg_map=fsaverage.sulc_left)
    figure.savefig(f"L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/VLSM_factored_permTest_5000_Factor_2_nonsign_{name_add}.svg")
    # from plt.savefig(
    #         os.path.join(output_dir, "figures", f"feature_importances_{label}_{interview_part}.png"), dpi = 300)
    plotting.show()
    return


if __name__ == "__main__":
    # Define the file paths for the lesion mask and atlas image
    ## Initialize some variables
    # TODO: Vul dit zelf aan
    cluster_img_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/output/VLSM_factored_withMonthsPO_perm_5000_lesionregr_MCcorrected/non_sign_largest_cluster_Factor_2.nii"
        # "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/maps/sub-01.nii"
    # Path to cluster img (nifti-file), make sure to use / instead of \; and add .nii extension

    # plot first unthresholded results
    plot_VLSM_cluster(cluster_img_path)

    # plot thresholded results
    plot_VLSM_cluster(cluster_img_path, zthreshold=1.645)