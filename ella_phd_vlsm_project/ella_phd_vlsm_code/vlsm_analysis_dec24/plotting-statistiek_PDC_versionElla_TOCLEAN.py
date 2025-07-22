"""Script to calculate and plot CORRECTED VLSM-statistics (using permuation-based cluster thresholds)
-------------------------------------------------------------------------------------------------------
This script performs and plots univariate voxel-wise lesion-symptom mapping corrected for multiple comparisons
using permutation testing, based on NiiStat outcomes.

1) What do you need before running this script:
- permutation tests per variable: 5000 (or modify this yourself) permutation tests,
that will be stored in a 'permTest' folder in the VLSM_output directory.
- (uncorrected) z-statistic maps (per variable) of VLSM-analysis: nii-file containing z-statistics (matrix),
that will be stored in VLSM output folder.
The permutation tests and (uncorrected) z-statistics are calculated in Matlab with NiiStat software.

2) Workflow to correct for multiple comparisons, using the data above:
- First you select uncorrected p-value. For example p=.01, which is Z=2.33. These Z-values above 2.33 you keep,
everything below = 0. Do this for effective Z-map as well as for the 1000 permutation tests.
- Next, you are going to do a kind of “correction for multiple comparisons” by means of
cluster-based permutation testing (Winkler et al., 2014) using the method implemented in Stark et al. (2019).
    i) Determine the cluster size in all 5000 permutation tests. A cluster is a group of voxels with Z-value > 2.33
    ii) then you rank the cluster sizes per permutation test
    iii) then you choose a particular p-value on which to correct. If you choose p<.05 threshold,
    then your threshold = 250th largest cluster size of all 5000 permutation tests.
    iv) then you use that cluster size to correct for multiple comparisons.
    v) then you set in your effective Z-map all voxels = 0 where those voxels
    do not belong to a cluster with a size >= your cluster size you just calculated above.
    all voxels that survive this are significant.
Ref: Stark, B. C. (2019). A comparison of three discourse elicitation methods in aphasia and age-matched adults:
Implications for language assessment and outcome. American Journal of Speech-Language Pathology, 28(3),
1067–1083. https://doi.org/10.1044/2019_AJSLP-18-0265


IMPORTANT: think carefully whether you are interested in negative or positive Z-values.
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

NOTE: This script should be run

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


## -- PREPARATIONS --
# ----------------------

## Specifieer volgende variabelen telkens (bij elke run) ZELF
# TODO: specifieer dit allemaal ZELF (tem 'focus_on_positive_zvalues'
variable = 'Factor_1'
    # = "ANTAT_speechrate"
path_to_VLSM_folder = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA"
    # = "C:/Users/u0146803/Documents/VLSM_masterthesis"
uncorrected_VLSM_output_folder_name = "VLSM_factored_withMonthsPO_perm_5000_lesionregr_10Dec2024_094405"
    # = "VLSM_ANTAT_perm_1000_lesionregr_23Dec2024_224000"  # this is the output file of NiiStat (with no corrections for multiple comparisons)
uncorrected_z_map = '_'.join(uncorrected_VLSM_output_folder_name.split("_")[:6])
    # '_'.join(uncorrected_VLSM_output_folder_name.split("_")[:5])
# should be: VLSM_factored_withMonthsPO_perm_5000_lesionregr
# should be: VLSM_ANTAT_perm_1000_lesionregr
corrected_VLSM_output_folder_name = "VLSM_factored_withMonthsPO_perm_5000_lesionregr_MCcorrected"
    # "VLSM_ANTAT_perm_1000_lesionregr_MCcorrected"  # this WILL be the output file of the Multiple Comparisons correction performed in this script
uncorrected_pthreshold = 0.05
corrected_pthreshold = 0.05
nb_of_permutations = 5000
focus_on_positive_zvalues = True  # noteer False indien focus op NEGATIEVE z-waarden


## Andere variabelen die geïnitialiseerd worden (NIET aan te passen):
# dit zijn variabelen die we nodig hebben voor de statistiek:
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

#  dit zijn nog enkele variabelen die we nodig hebben om te plotten:
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
curv_left = surface.load_surf_data(fsaverage.curv_left)
curv_left_sign = np.sign(curv_left)


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


## -- STEP 1: VLSM analysis in NiiStat
# ----------------------------------------
# returns uncorrected z-values (must be done in NiiStat: see manual Pieter)


## -- STEP 2: Permutation testing generation in NiiStat
#--------------------------------------------------------
# returns 1000 permutation tests per variable (each perm test = Z-map with z-values)


## -- STEP 3: niet-actieve z-waarden (onder uncorrected p-value) EXCLUDEREN + STEP 4: Clusters van actieve voxels (z > 2.33*) bepalen en ranken
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Als eerste begin je met al je permutation tests in te lezen. Hier in for-loop
# in die for-loop zet ik alle voxels met Z-waarde < 1.65 naar 0.
# vervolgens bereken ik cluster size per permutation test (zie paper)

# dit doe ik nu bij data van mijn VLSM paper. Die data staan onder "paper4_VLSM_aphasia", bij output --> permTest
# toevoeging door Ella: door dit voor IEDERE variable of interest!
# TODO: PAD zelf aanpassen (alleen indien echt nodig, normaal wordt dit automatisch gedaan volgens volgend stramien: /pad VLSM + output/permTest + variable/
variable_perm_path = os.path.join(path_to_VLSM_folder, 'output/permTest',f"{variable}/")
    # ("C:/Users/u0146803/Documents/VLSM_masterthesis/output/permTest/ANTAT_afgebrokenwoord/")
    #"E:/vlsm_scratch/output/permTest/broad40_all/"  # lokaal laten lopen, pas het pad zelf aan
    # pas op: na naam variable ook '/' zetten!

allPerms = [f for f in os.listdir(variable_perm_path) if f.endswith('.nii')]  # lijst alle permutation tests

size = list()  # hierin store je de grootste cluster size

for file in allPerms:
    # inlezen van nii file hieronder (twee lijntjes)
    img = nib.load((variable_perm_path + file))
    img_data = img.get_fdata()

    if focus_on_positive_zvalues: # zet zthreshold, behoud alleen die img_data > treshold (of bij neg z-waarden: die x <- treshold)
        thresholded_img_data = img_data > uncorrected_zthreshold
    else:
        thresholded_img_data = img_data < -uncorrected_zthreshold

    labeled_clusters, num_clusters = ndimage.label(thresholded_img_data)  # hier zoek je naar clusters
    cluster_sizes = ndimage.sum(thresholded_img_data, labeled_clusters,
                                range(num_clusters + 1))  # hier bereken je sizes van clusters
    size.append(np.max(cluster_sizes))  # sla per permutation test de cluster size op in de empty list


## -- STEP 5: Corrected cluster-treshold bepalen
# -----------------------------------------------
# nu heb je alle cluster sizes berekend. Afgaand op je corrected threshold voor cluster size die je kiest (=/= uncorrected treshold van hierboven!), neem je nu de N-grootste size.
# bijvoorbeeld, bij corrected cluster treshold p=.05 en 1000 permutaties, is dat N=50
# Aanvulling Ella: Stark: bij p=.01 als cluster treshold en 5000 permutaties, is dat N=50 (note: zij gebruikt N=100, terwijl wel p .01 en 5000 perm?)
# Aanvulling Ella VLSM IANSA: p 0.01 te strikt, toch p 0.05 als cluster threshold gekozen, is dan N=250
# Aanvulling Ella masterthesis: 1000 perm, p 0.05 => 50e cluster
# dat is je correctie voor multiple comparisons via cluster sizes
ranked_values = np.sort(size)[::-1]  # rank ze (-1 statement staat voor descending order (van groot nr klein))
#  (Pieter: 1000 permutaties => corrected p treshold p=0.05 komt overeen met N=50 dus 50e cluster size kiezen als cluster_treshold)
index_perm_cluster_threshold= int((corrected_pthreshold * nb_of_permutations)-1) # neemt de 50ste (start te tellen vanaf 0 dus daarom 49)
cluster_threshold = ranked_values[index_perm_cluster_threshold]
# cluster_threshold = ranked_values[49]  # neem de 50ste (start te tellen vanaf 0 dus daarom 49)

print("cluster threshold: N = {0}".format(cluster_threshold))


## -- STEP 6: Ongecorrigeerde z-waarden CORRIGEREN adhv gecorrigeerde cluster-treshold
# --------------------------------------------------------------------------------------
# nu gaan we kijken naar de effectieve Z-map (NIET permTest map).
# TODO: PAD zelf aanpassen (opnieuw PER VARIABELE, doe dit dus voor zelfde variabele als die je specifieerde hierboven)
img = nib.load(os.path.join(path_to_VLSM_folder, 'output', uncorrected_VLSM_output_folder_name, f"Z{uncorrected_z_map}{variable}.nii"))
# "C:/Users/u0146803/Documents/VLSM_masterthesis/output/VLSM_ANTAT_perm_1000_lesionregr_23Dec2024_224000/ZVLSM_ANTAT_perm_1000_lesionregrANTAT_afgebrokenwoord.nii")
    #'D:/PhD Pieter De Clercq/paper4_VLSM_aphasia/output/final___31Jan2024_103046/Zfinal__broad40_all.nii')  # laad je data. Staat in mapje paper4_VLSM_aphasia, pas aan (heb dit lokaal laten lopen)
img_data = img.get_fdata()

# eerst: zet alle waarden met Z<2.33 = 0
if focus_on_positive_zvalues:  # zet threshold, behoud alleen die img_data > treshold (of bij neg z-waarden: die x <- treshold)
    thresholded_img_data = img_data > uncorrected_zthreshold
else:
    thresholded_img_data = img_data < -uncorrected_zthreshold

# vervolgens: selecteer je clusters die cluster thresholding surviven; dwz clusters met een size die groter is dan je cluster threshold van de permutation tests hierboven, dan weet je dat die cluster significant is!
labeled_clusters, num_clusters = ndimage.label(thresholded_img_data)
cluster_sizes = ndimage.sum(thresholded_img_data, labeled_clusters, range(num_clusters + 1))

print("Identified cluster sizes: {0}".format(
    cluster_sizes))  # hiermee print ik de groottes van alle clusters die het vindt; Alle clusters met een size groter dan de threshold van de permutation test blijven behouden!
print("Largest cluster size= {0} voxels".format(np.max(cluster_sizes)))  # dit is de grootste cluster die het vindt.

# nu kijken welke clusters de thresholding overleven:
surviving_clusters = np.where(cluster_sizes >= cluster_threshold)[0]
# np.where returns tuple [(indices of elements in array where condition is true), empty] so np.where(..)[0] returns just the indices
print("Number of clusters that survived cluster threshold: {0}".format(len(surviving_clusters)))


## -- STEP 7: PLOTTEN
# --------------------
# plotten. Als er niks overleeft, dan plot ik de grootste cluster (= aanpassing dr Ella)
if len(surviving_clusters) == 0:
    # calculate largest cluster size and check its corresponding corrected p-value
    largest_cluster_size = np.max(cluster_sizes)
    closest_perm_cluster = take_closest(myList= ranked_values[::-1], myNumber= largest_cluster_size)
    if len(closest_perm_cluster) > 2:  # als er 4 elementen in lijst zitten aka als je 2 sizes (en hun posities) hebt die het dichtst bij jouw grootste cluster liggen
        corresponding_perm_cluster_size = closest_perm_cluster[:1]
        corresponding_p_value = [1-(closest_perm_cluster[3]/nb_of_permutations), 1-(closest_perm_cluster[2]/nb_of_permutations)]
        # 1 - X omdat X = xe 'kleinste' cluster, en wij willen y'de grootste cluster want dat is p
        print("permutation cluster sizes around your largest cluster : {0}".format(corresponding_perm_cluster_size))
        print("p-values around your largest cluster : {0}".format(corresponding_p_value))
    else:  # 1 cluster die dichtst bij jouw grootste cluster ligt
        corresponding_perm_cluster_size = closest_perm_cluster[0]
        corresponding_p_value = 1-(closest_perm_cluster[1]/nb_of_permutations)
        print("permutation cluster size closest to your largest cluster : {0}".format(corresponding_perm_cluster_size))
        print("p-value closest to your largest cluster : {0}".format(corresponding_p_value))


    # decide on largest cluster img data
    largest_cluster = np.where(cluster_sizes == largest_cluster_size)[0]
    # largest_cluster = np.where(np.max(cluster_sizes))[0]  # should return index of the largest cluster
    largest_cluster_mask = np.isin(labeled_clusters, largest_cluster)
    largest_cluster_data = img_data * largest_cluster_mask

    if focus_on_positive_zvalues:
        largest_cluster_data = largest_cluster_data * 1 # die 1 (positieve) of -1 (negatieve) hangt af of je geïnteresseerd bent in negatieve of positieve Z-waarden. Speel hiermee tot je zelf hebt wat je wil
    else:
        largest_cluster_data = largest_cluster_data * -1

    # terug naar nii format voor plotting
    largest_cluster_img = nib.Nifti1Image(largest_cluster_data, img.affine)
    texture = surface.vol_to_surf(largest_cluster_img, fsaverage.pial_left)  # surface map
    figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                         texture, hemi='left',
                                         title='Surface plot left hemisphere of largest cluster'.format(largest_cluster),
                                         colorbar=True, threshold=0.001, cmap='twilight',
                                         bg_map=fsaverage.sulc_left)
    # voorbeeld om op te slaan
    # TODO: PAD zelf aanpassen (opnieuw PER VARIABELE, doe dit dus voor zelfde variabele als die je specifieerde hierboven); specifieer type (.svg) of (.png)
    figure.savefig(os.path.join(path_to_VLSM_folder, 'figures', f"{variable}_nonsign.png"))
        # "C:/Users/u0146803/Documents/VLSM_masterthesis/figures/ANTAT_afgebrokenwoord_nonsign.png")
    # figure.savefig('/media/pieter/7111-5376/vlsm_scratch/plots/nonsign_largest_cluster165_broad.svg')
    plotting.show();
    print("Nothing survived threshold, nothing significant to plot, largest cluster (not significant) is shown")

else:  # indien er wel iets overleeft, loop ik over alle clusters die cluster threshold overleven:
    for this_cluster in surviving_clusters:
        # clusters die het niet overleven, zet ik hieronder op 0. Dan ga ik er per cluster door. 3 lijntjes code hieronder
        surviving_clusters_mask = np.isin(labeled_clusters, this_cluster)
        surviving_clusters_data = img_data * surviving_clusters_mask
        # TODO: Pas dit zelf aan afhankelijk van interesse in POSITIEVE (* 1; of volgende lijn outcommenten want geen effect) of NEGATIEVE (* -1) z-waarden
        if focus_on_positive_zvalues:
            surviving_clusters_data = surviving_clusters_data * 1  # die 1 (positieve) of -1 (negatieve) hangt af of je geïnteresseerd bent in negatieve of positieve Z-waarden. Speel hiermee tot je zelf hebt wat je wil
        else:
            surviving_clusters_data = surviving_clusters_data * -1

        # terug naar nii format voor plotting
        surviving_clusters_img = nib.Nifti1Image(surviving_clusters_data, img.affine)
        texture = surface.vol_to_surf(surviving_clusters_img, fsaverage.pial_left)  # surface map
        figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                             texture, hemi='left',
                                             title='Surface plot left hemisphere of cluster {0}'.format(this_cluster),
                                             colorbar=True, threshold=0.001, cmap='twilight',
                                             bg_map=fsaverage.sulc_left)
        plotting.show();

    ## Hier, uit de for-loop, herhaal ik de code, maar plot ik alle clusters samen in 1 plot.
    surviving_clusters_mask = np.isin(labeled_clusters, surviving_clusters)
    surviving_clusters_data = img_data * surviving_clusters_mask
    if focus_on_positive_zvalues:
        surviving_clusters_data = surviving_clusters_data * 1  # die 1 (positieve) of -1 (negatieve) hangt af of je geïnteresseerd bent in negatieve of positieve Z-waarden. Speel hiermee tot je zelf hebt wat je wil
    else:
        surviving_clusters_data = surviving_clusters_data * -1

    surviving_clusters_img = nib.Nifti1Image(surviving_clusters_data, img.affine)
    texture = surface.vol_to_surf(surviving_clusters_img, fsaverage.pial_left)
    figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                         texture, hemi='left',
                                         title='Surface plot left hemisphere of all clusters combined',
                                         colorbar=True, threshold=0.001, cmap='twilight',
                                         bg_map=fsaverage.sulc_left)

    # voorbeeld om op te slaan
    # TODO: PAD zelf aanpassen (opnieuw PER VARIABELE, doe dit dus voor zelfde variabele als die je specifieerde hierboven); specifieer type (.svg)
    figure.savefig(os.path.join(path_to_VLSM_folder, 'figures', f"{variable}_sign.png"))
    # figure.savefig('/media/pieter/7111-5376/vlsm_scratch/plots/cluster165_broad.svg')
    plotting.show();


## -- EXTRA: save corrected z-files
# -----------------------------------
## Mocht je ooit een mapje die ik hier aan maak (bvb, de Z-map maar dan met cluster threshold) willen opslaan (bvb om eens in te laden in MRIcroGL/mricron)
# gebruik dan die code en pas aan:
if len(surviving_clusters) == 0:
    # TODO: PAD zelf aanpassen (kies passende naam, met specificatie van Pad naar output file "VLSM/Permutatie_analyse_MCcorrected/Znonsign_clusters_VARIABELE die je specifieerde hierboven.nii")
    nib.save(largest_cluster_img,
             filename = os.path.join(path_to_VLSM_folder, 'output', corrected_VLSM_output_folder_name, f"Znonsign_cluster_{variable}.nii"))

             #"C:/Users/u0146803/Documents/VLSM_masterthesis/output/VLSM_correctedMC/Znonsign_clusters_ANTAT_afgebrokenwoord.nii")
    # nib.save(surviving_clusters_img, 'path/to/save/nonsign_cluster.nii') #pas pad aan, doe comment weg
else:
    # TODO: PAD zelf aanpassen (kies passende naam, met specificatie van Pad naar output file "VLSM/Permutatie_analyse_MCcorrected/Zsurviving_clusters_VARIABELE die je specifieerde hierboven.nii")
    # Note: als VLSM analyse geen enkele cluster vindt, zal deze lijn een error geven (omdat surviving_clusters_img dan niet gedefinieerd wordt), negeer die Error (is niet erg)
    nib.save(surviving_clusters_img,
             filename = os.path.join(path_to_VLSM_folder, 'output', corrected_VLSM_output_folder_name, f"Zsurviving_clusters_{variable}.nii"))
             # "C:/Users/u0146803/Documents/VLSM_masterthesis/output/VLSM_correctedMC/Zsurviving_clusters_ANTAT_afgebrokenwoord.nii")
    # nib.save(surviving_clusters_img, 'path/to/save/surviving_clusters.nii') #pas pad aan, doe comment weg

