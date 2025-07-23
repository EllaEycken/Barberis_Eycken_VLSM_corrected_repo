"""Script to calculate and plot CORRECTED VLSM-statistics (using permutation-based cluster thresholds)
-------------------------------------------------------------------------------------------------------
This script performs and plots univariate voxel-wise lesion-symptom mapping corrected for multiple comparisons
using permutation testing, based on NiiStat outcomes. (for 1 behavioral variable)

1) What do you need before running this script:
- permutation tests per variable: 5000 (or modify this yourself) permutation tests,
that will be stored in a 'permTest' folder in the VLSM_output directory.
- (uncorrected) z-statistic maps (per variable) of VLSM-analysis: nii-file containing z-statistics (matrix),
that will be stored in VLSM output folder.
The permutation tests and (uncorrected) z-statistics are calculated in Matlab with NiiStat software.

2) Workflow to correct for multiple comparisons, using the data above (assuming you focus on POSITIVE z-values*):
- PREPARATIONS: you set the uncorrected and corrected p-values:
    1) The uncorrected p-value. For example p=.01, which is Z=2.33.
    You will keep the Z-values above 2.33, everything below will be set to 0.
    You will do this for effective Z-map as well as for the 1000 permutation tests.
    2) The corrected p-value: For example, p= 0.01, which is Z=2.33.
    You will use this p-value to corrected for multiple comparisons.
    This value will determine a 'threshold cluster' size (using the permutation test results), that will be held
    against all effective Z-map voxels that survived the uncorrected z-threshold:
    if the voxel belongs to a cluster smaller than the threshold cluster, it will NOT be kept.
- CORRECTION: you are going to do a kind of “correction for multiple comparisons” by means of
cluster-based permutation testing (Winkler et al., 2014) using the method implemented in Stark et al. (2019).
    STEP 1) Determine the cluster threshold (using the permutation tests)
        i) As explained above, in all permutation tests, keep the z-values > uncorrected z-value,
        and set the z-values < uncorrected z-value to 0
        ii) Determine the cluster size in all 5000 permutation tests. A cluster is a group of voxels
        with Z-value > uncorrected threshold 2.33. For each permutation test, keep only the largest cluster size.
        iii) then you rank those largest cluster sizes of the permutation tests
        iv) then you choose a particular p-value on which to correct. If you choose p<.05 threshold,
        then your threshold = 250th largest cluster size of all 5000 permutation tests.
    STEP 2) use the threshold cluster size to correct for multiple comparisons in your effective Z-map (the VLSM output of NiiStat),
        i) Uncorrected part: you set all voxels = 0 where those voxels have a Z-value < the uncorrected z-value
        (as explained in the first lines above).
        ii) Corrected part: you set all voxels = 0 where those voxels do not belong to a
        cluster with a size >= the threshold cluster size you just calculated above.
        all voxels that survive this are significant.
- PLOTTING the results: plot the corrected VLSM output
    i) If there are no clusters surviving cluster threshold, then plot the largest cluster that was found.
    ii) If there are clusters surviving the cluster threshold, plot them

- SAVING THE CORRECTED Z-MAPS

Ref: Stark, B. C. (2019). A comparison of three discourse elicitation methods in aphasia and age-matched adults:
Implications for language assessment and outcome. American Journal of Speech-Language Pathology, 28(3),
1067–1083. https://doi.org/10.1044/2019_AJSLP-18-0265


*IMPORTANT: think carefully whether you are interested in negative or positive Z-values.
For example: in PDC, brain lesions should correlate to lower brain response (lower brain response = worse).
So I was interested in negative Z-values.
Imagine you are researching semantic errors: brain lesions should correlate with more semantic errors
(“semantic errors” = Worse), then you are interested in positive Z-values.
# in IANSA': brain lesions and their correlation to factors and degree to which someone matches that factor
=> interested in POSITIVE z-values

NOTE: if you get errors, it is often 1) either because of packages, install earlier versions, or 2) because you
have problems with your dimensions of your images not matching across subjects.
Read the manual of the VLSM at the bottom, more info there.

NOTE: final results on Z-file and plot show permutation-based corrected z-clusters
(that survived cluster threshold or that were large enough) => z-values below the corrected z threshold are set to zero.

NOTE: This script is not built as a function and should be run by defining and changing variables in-code (see TODO statements).

If you want to perform further analyses on the VLSM output:
- calculate_cluster_distribution.py: to examine in which regions the VLSM output cluster lies
- plot_VLSM_output.py: to plot VLSM results -> legenda of plot must be taken into account!

"""

## IMPORT
import nibabel as nib
from nilearn import image, regions
import numpy as np
from scipy import ndimage
from nilearn import plotting, datasets, surface
import os
from bisect import bisect_left


""" -- PREPARATIONS: SETTING THE THRESHOLDS  --"""
"""----------------------"""

## Setting variables
# TODO: Specify these variables YOURSELF at each run
variable = 'linguistic_variable_to_analyze'
path_to_VLSM_folder = "path/to/your/VLSM/folder"
uncorrected_VLSM_output_folder_name = "NiiStat_VLSM_zmaps_day_hour"
    # this is the output file of NiiStat (with no corrections for multiple comparisons)
uncorrected_z_map = '_'.join(uncorrected_VLSM_output_folder_name.split("_")[:6])
    # this will be the name of the uncorrected_VLSM_output_folder, without the day and hour
corrected_VLSM_output_folder_name = '_'.join([uncorrected_z_map, 'MCcorrected'])
    # this WILL be the output file of the Multiple Comparisons correction performed in this script
uncorrected_pthreshold = 0.05
    # the uncorrected p-threshold (before correcting for multiple comparisons)
corrected_pthreshold = 0.05
    # the corrected p-threshold (after correcting for multiple comparisons)
nb_of_permutations = 5000
    # How many permutation tests you want to run
focus_on_positive_zvalues = True
    # is False if you want to focus on NEGATIVE z-values


## Other variables that are initialized (DO NOT CHANGE THIS CODE)
# statistics variables:
if uncorrected_pthreshold != corrected_pthreshold:
    if uncorrected_pthreshold == 0.05:
        uncorrected_zthreshold = 1.645
    elif uncorrected_pthreshold == 0.01:
        uncorrected_zthreshold = 2.33
    else:
        uncorrected_zthreshold = None
    if corrected_pthreshold == 0.05:
        corrected_zthreshold = 1.645
    if corrected_pthreshold == 0.01:
        corrected_zthreshold = 2.33
    else:
        corrected_zthreshold = None
else:
    if uncorrected_pthreshold == 0.05:
        uncorrected_zthreshold = 1.645
    elif uncorrected_pthreshold == 0.01:
        uncorrected_zthreshold = 2.33
    else:
        uncorrected_zthreshold = None
    corrected_zthreshold = uncorrected_zthreshold
#  Plotting variables:
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
curv_left = surface.load_surf_data(fsaverage.curv_left)
curv_left_sign = np.sign(curv_left)


"""-- HELPER FUNCTION --"""
"""---------------------"""
# define a function to check where a value is closest to in a list (will be used later)
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value (AND its index) to myNumber.

    If two numbers are equally close, return  both (and their indices).

    Source: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return [myList[0], pos]
    if pos == len(myList):
        return [myList[-1], pos]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return [after, pos]
    else: # the numbers are equally close
        return [before, after, pos-1, pos]


""" -- BEFORE RUNNING SCRIPT, PART 1: uncorrected VLSM analysis in NiiStat--"""
""" ----------------------------------------------------------------------"""
# returns uncorrected z-values (must be done in NiiStat: see manual PDC)



""" -- BEFORE RUNNING SCRIPT, PART 2: generate Permutation tests in NiiStat --"""
"""-------------------------------------------------------------------------"""
# returns 5000 permutation tests per variable (each perm test = Z-map with z-values) (is done simultaneously with
# step 1 in NiiStat: see manual PDC)



"""-- MULTIPLE COMPARISONS CORRECTION--"""
"""------------------------------------"""
""" STEP 1: Determine the cluster threshold (using the permutation tests)"""
""" ---------------------------------------------------------------------"""
# TODO: edit PAD yourself (only if really necessary, normally this is done automatically according to the following pattern: /path VLSM + output/permTest + variable/
variable_perm_path = os.path.join(path_to_VLSM_folder, 'output/permTest',f"{variable}/")
    # Load the path to the PERMUTATION TESTS for the variable of interest
    # beware: after variable name also put ‘/’!

allPerms = [f for f in os.listdir(variable_perm_path) if f.endswith('.nii')]  # create list of all permutation tests for the variable of interest

size = list()  # in this list, the largest cluster size for each permTest will be stored

for file in allPerms: # for each permutation test: append the largest cluster size within this perm test to a list
    # read the nifti file (z-map of the perm test)
    img = nib.load((variable_perm_path + file))
    img_data = img.get_fdata()

    """ i) EXCLUDE non-active z-values in the permutation tests (< uncorrected p-value) """
    if focus_on_positive_zvalues:
        # Keep only the perm test z-values > uncorrected threshold (or for negative focus: z < -treshold),
        # set the other values to 0.
        thresholded_img_data = img_data > uncorrected_zthreshold
    else:
        thresholded_img_data = img_data < -uncorrected_zthreshold


    """ ii) For each permutation test, determine clusters of active voxels in permutation tests (> uncorrected p-value)
        and keep only the largest cluster."""
    labeled_clusters, num_clusters = ndimage.label(thresholded_img_data)  # look for clusters in the perm test z-map
    cluster_sizes = ndimage.sum(thresholded_img_data, labeled_clusters,
                                range(num_clusters + 1))  # calculate the sizes of the clusters
    size.append(np.max(cluster_sizes))  # save only the largest cluster size for that perm test z-map in the list

""" iii) rank those largest cluster sizes across permutation tests, from largest to smallest """
ranked_values = np.sort(size)[::-1]  # '-1' statement means descending order (larg to small)

""" iv) Based on the corrected p-threshold (you defined above), define the cluster threshold """
# If you chose p<.05 corrected threshold, the cluster threshold is the cluster that represents
# the 5% (~0.05 or 5/100) largest cluster size of all permutation tests. This adds an extra criterium that a cluster
# found in the effective z-map (see further) should be AT LEAST larger than the 5% largest cluster in a permuted sample,
# to be counted as a significant cluster.
# => If you generated 5000 permutation tests, then your threshold = 0.05*5000 = 250th largest cluster size of all 5000 permutation tests.
# => If you generated 1000 permutation tests, then your threshold = 0.05*1000 = 50th largest cluster size of all 1000 permutation tests.

index_perm_cluster_threshold= int((corrected_pthreshold * nb_of_permutations)-1)
    # decide the index of the pth largest cluster size
    # (starts counting from 0, so -1: e.g., if 50th largest cluster, take index 49)
cluster_threshold = ranked_values[index_perm_cluster_threshold]
print("cluster threshold: N = {0}".format(cluster_threshold))



""" STEP 2: Correct the uncorrected z-values (of the effective VLSM output) using the cluster threshold """
""" ----------------------------------------------------------------------------------------------------"""
# TODO: # TODO: edit PAD yourself (only if really necessary, normally this is done automatically according to the following pattern: /path VLSM + output/uncorrected_VLSM_output_folder + Zmap_variable/
img = nib.load(os.path.join(path_to_VLSM_folder, 'output', uncorrected_VLSM_output_folder_name, f"Z{uncorrected_z_map}{variable}.nii"))
    # Load the path to the EFFECTIVE VLSM OUTPUT Z-MAP for the variable of interest
img_data = img.get_fdata()

""" i) Uncorrected part: you set all voxels = 0 where those voxels have a Z-value < the uncorrected z-value """
if focus_on_positive_zvalues:
    # Keep only the z-values > uncorrected threshold (or for negative focus: z < -treshold),
    # set the other values to 0.
    thresholded_img_data = img_data > uncorrected_zthreshold
else:
    thresholded_img_data = img_data < -uncorrected_zthreshold

""" ii) Corrected part: you set all voxels = 0 where those voxels do not belong to a cluster with a size >= the threshold cluster size you just calculated above.
        all voxels that survive this are significant."""
labeled_clusters, num_clusters = ndimage.label(thresholded_img_data)  # look for clusters in the effective z-map
cluster_sizes = ndimage.sum(thresholded_img_data, labeled_clusters, range(num_clusters + 1))  # determine the cluster sizes of those clusters

print("Identified cluster sizes: {0}".format(
    cluster_sizes))  # print all cluster sizes in the effective z-map
print("Largest cluster size= {0} voxels".format(np.max(cluster_sizes)))  # print the largest cluster size in the effective z-map

surviving_clusters = np.where(cluster_sizes >= cluster_threshold)[0]  # only keep those clusters that are larger than the cluster threshold
# np.where returns tuple [(indices of elements in array where condition is true), empty] so np.where(..)[0] returns just the indices
print("Number of clusters that survived cluster threshold: {0}".format(len(surviving_clusters)))  # print the amount of surviving clusters



"""-- PLOTTING THE CORRECTED VLSM OUTPUT RESULTS: significant clusters or largest clusters -- """
""" -------------------- """
""" i) If there are no clusters surviving cluster threshold, then plot the largest cluster that was found. """
if len(surviving_clusters) == 0:
    # calculate largest cluster size and check its corresponding corrected p-value
    largest_cluster_size = np.max(cluster_sizes)
    closest_perm_cluster = take_closest(myList= ranked_values[::-1], myNumber= largest_cluster_size)
    if len(closest_perm_cluster) > 2:  # if there are 4 elements in list aka if you have 2 cluster sizes (and their positions) closest to your largest cluster
        corresponding_perm_cluster_size = closest_perm_cluster[:1]
        corresponding_p_value = [1-(closest_perm_cluster[3]/nb_of_permutations), 1-(closest_perm_cluster[2]/nb_of_permutations)]
        # 1 - X because X = xe ‘smallest’ cluster, and we want y'the largest cluster because that is p
        print("permutation cluster sizes around your largest cluster : {0}".format(corresponding_perm_cluster_size))
        print("p-values around your largest cluster : {0}".format(corresponding_p_value))
    else:  # 1 cluster size that is closest to your largest cluster
        corresponding_perm_cluster_size = closest_perm_cluster[0]
        corresponding_p_value = 1-(closest_perm_cluster[1]/nb_of_permutations)
        print("permutation cluster size closest to your largest cluster : {0}".format(corresponding_perm_cluster_size))
        print("p-value closest to your largest cluster : {0}".format(corresponding_p_value))


    # decide on largest cluster img data
    largest_cluster = np.where(cluster_sizes == largest_cluster_size)[0] # should return index of the largest cluster
    largest_cluster_mask = np.isin(labeled_clusters, largest_cluster)
    largest_cluster_data = img_data * largest_cluster_mask

    # To plot, you need to convert negative z-values to positive (if necessary)
    if focus_on_positive_zvalues:
        largest_cluster_data = largest_cluster_data * 1
    else:
        largest_cluster_data = largest_cluster_data * -1

    # Plot the resulting largest cluster(s)
    largest_cluster_img = nib.Nifti1Image(largest_cluster_data, img.affine)
    texture = surface.vol_to_surf(largest_cluster_img, fsaverage.pial_left)  # surface map
    figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                         texture, hemi='left',
                                         title='Surface plot left hemisphere of largest cluster'.format(largest_cluster),
                                         colorbar=True, threshold=0.001, cmap='twilight',
                                         bg_map=fsaverage.sulc_left)

    # Save the plot
    # TODO: change the path (normally not necessary, see format) (specify file type (.svg) or (.png))
    figure.savefig(os.path.join(path_to_VLSM_folder, 'figures', f"{variable}_nonsign.png"))

    plotting.show();
    print("Nothing survived threshold, nothing significant to plot, largest cluster (not significant) is shown")


    """ ii) If there are clusters surviving the cluster threshold, plot them """
else:
    ## Plot each surviving cluster separately first
    for this_cluster in surviving_clusters:  # go over each surviving cluster
        # Get the img_data of the surviving cluster
        surviving_clusters_mask = np.isin(labeled_clusters, this_cluster)
        surviving_clusters_data = img_data * surviving_clusters_mask

        # To plot, you need to convert negative z-values to positive (if necessary)
        if focus_on_positive_zvalues:
            surviving_clusters_data = surviving_clusters_data * 1
        else:
            surviving_clusters_data = surviving_clusters_data * -1

        # Plot the surviving cluster(s)
        surviving_clusters_img = nib.Nifti1Image(surviving_clusters_data, img.affine)
        texture = surface.vol_to_surf(surviving_clusters_img, fsaverage.pial_left)  # surface map
        figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                             texture, hemi='left',
                                             title='Surface plot left hemisphere of cluster {0}'.format(this_cluster),
                                             colorbar=True, threshold=0.001, cmap='twilight',
                                             bg_map=fsaverage.sulc_left)
        plotting.show();

    ## Now plot all surviving clusters in 1 big plot
    surviving_clusters_mask = np.isin(labeled_clusters, surviving_clusters)
    surviving_clusters_data = img_data * surviving_clusters_mask
    if focus_on_positive_zvalues:
        surviving_clusters_data = surviving_clusters_data * 1
    else:
        surviving_clusters_data = surviving_clusters_data * -1

    surviving_clusters_img = nib.Nifti1Image(surviving_clusters_data, img.affine)
    texture = surface.vol_to_surf(surviving_clusters_img, fsaverage.pial_left)
    figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                         texture, hemi='left',
                                         title='Surface plot left hemisphere of all clusters combined',
                                         colorbar=True, threshold=0.001, cmap='twilight',
                                         bg_map=fsaverage.sulc_left)

    # Save the plot
    # TODO: change the path (normally not necessary, see format) (specify file type (.svg) or (.png))
    figure.savefig(os.path.join(path_to_VLSM_folder, 'figures', f"{variable}_sign.png"))

    plotting.show();



""" -- EXTRA: save corrected z-files --"""
""" ----------------------------------- """
## Should you ever want to save a corrected Z-file folder (e.g., the Z folder but with cluster threshold) (e.g., to load into MRIcroGL/mricron)
# then use the following code and modify it:
if len(surviving_clusters) == 0:
    # TODO: Customize PATH yourself (normally runs automatically)
    #  (choose appropriate name, specifying Path to output file “VLSM/Permutation_analysis_MCcorrected/Znonsign_clusters_VARIABLE that you specified above.nii”)
    nib.save(largest_cluster_img,
             filename = os.path.join(path_to_VLSM_folder, 'output', corrected_VLSM_output_folder_name, f"Znonsign_cluster_{variable}.nii"))

else:
    # TODO: Customize PATH yourself (normally runs automatically)
    #  (choose appropriate name, specifying Path to output file “VLSM/Permutation_analysis_MCcorrected/Zsurviving_clusters_VARIABLE that you specified above.nii”)
    # Note: if VLSM analysis does not find any cluster, this line will give an error (because surviving_clusters_img is then not defined), ignore that Error (is not a big deal)
    nib.save(surviving_clusters_img,
             filename = os.path.join(path_to_VLSM_folder, 'output', corrected_VLSM_output_folder_name, f"Zsurviving_clusters_{variable}.nii"))

