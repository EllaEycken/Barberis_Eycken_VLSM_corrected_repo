import nibabel as nib
from nilearn import image, regions
import numpy as np
from scipy import ndimage
from nilearn import plotting, datasets, surface
import os

### ------ Script voor het plotten en bepalen van thresholds bij cluster-based permutation tests -----

## -- WHAT ---

# De permutation tests zijn berekend in Matlab met die niistat software. Als output krijg je 1000 (of pas dit zelf aan) permutation tests

# Workflow om te corrigeren voor multiple comparisons:
# eerst selecteer je uncorrected p-waarde. Bijvoorbeeld p=.01, dat is Z=2.33. Deze Z-waardes boven 2.33 behoud je, alles eronder zet je = 0. Doe dit zowel bij effectieve Z-map als bij de 1000 permutation tests
# daarna ga je een soort van "correctie voor multiple comparisons" doen door cluster size te bepalen in alle 1000 permutation tests. Een cluster is een groep van voxels waarvan die een Z-waarde hebben > 2.33
# daarna rank je de cluster sizes per permutation test
# dan kies je een bepaalde p-waarde waarop je corrigeert. Als je p<.05 threshold kiest, dan is je threshold = 50ste grootste cluster size van alle 1000 permutation tests
# die cluster size gebruik je om te corrigeren voor multiple comparisons.
# daarna zet je in je effectieve Z-map alle voxels = 0 waar die voxels niet behoren tot een cluster met een size >= je cluster size die je net berekent hebt hierboven
# alle voxels die dit overleven, zijn significant. Lees paper op het gemak


# BELANGRIJK: denk goed na of je geïnteresseerd bent in negatieve of positieve Z-waarden.
# Bijvoorbeeld: in mijn paper is lagere hersenrespons = Slechter. Dus was ik geïnteresseerd in negatieve Z-waarden.
# Stel je voor dat je onderzoekt meer "semantische fouten" = slechter, dan ben je geïnteresseerd in positieve Z-waarden.

# als je errors krijgt, dan is dit vaak 1) ofwel door packages, installeer dan eerdere versies, of 2) doordat je problemen hebt met je dimensies van je beelden die over
# proefpersonen heen niet overeenkomen. Lees de manual van de VLSM onderaan, daar heb ik meer info geplaatst.


## -- PREPARATIONS --

# dit zijn nog enkele variabelen die we nodig hebben om te plotten:
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
curv_left = surface.load_surf_data(fsaverage.curv_left)
curv_left_sign = np.sign(curv_left)


## -- STEP 1: VLSM analysis in NiiStat
# returns uncorrected z-values (must be done in NiiStat: see manual Pieter)


## -- STEP 2: Permutation testing generation in NiiStat
# returns 1000 permutation tests per variable (each perm test = Z-map with z-values)


## -- STEP 3: niet-actieve z-waarden (onder uncorrected p-value) EXCLUDEREN + STEP 4: Clusters van actieve voxels (z > 2.33*) bepalen en ranken
# Als eerste begin je met al je permutation tests in te lezen. Hier in for-loop
# in die for-loop zet ik alle voxels met Z-waarde < 1.65 naar 0.
# vervolgens bereken ik cluster size per permutation test (zie paper)

# dit doe ik nu bij data van mijn VLSM paper. Die data staan onder "paper4_VLSM_aphasia", bij output --> permTest
path = "E:/vlsm_scratch/output/permTest/broad40_all/"  # lokaal laten lopen, pas het pad zelf aan

allPerms = [f for f in os.listdir(path) if f.endswith('.nii')]  # lijst alle permutation tests

threshold = 2.33  # Z-threshold voor p=.01 zoals in mijn paper. Pas zelf aan (1.65 is .05 bvb)

size = list()  # hierin store je de grootste cluster size

for file in allPerms:
    # inlezen van nii file hieronder (twee lijntjes)
    img = nib.load((path + file))
    img_data = img.get_fdata()

    thresholded_img_data = img_data > threshold  # zet threshold

    labeled_clusters, num_clusters = ndimage.label(thresholded_img_data)  # hier zoek je naar clusters
    cluster_sizes = ndimage.sum(thresholded_img_data, labeled_clusters,
                                range(num_clusters + 1))  # hier bereken je sizes van clusters
    size.append(np.max(cluster_sizes))  # sla per permutation test de cluster size op in de empty list


## -- STEP 5: Corrected cluster-treshold bepalen
# nu heb je alle cluster sizes berekend. Afgaand op je threshold voor cluster size die je kiest, neem je nu de N-grootste size.
# bijvoorbeeld, bij p=.05 is dat N=50
# dat is je correctie voor multiple comparisons via cluster sizes
ranked_values = np.sort(size)[::-1]  # rank ze
cluster_threshold = ranked_values[49]  # neem de 50st

print("cluster threshold: N = {0}".format(cluster_threshold))


## -- STEP 6: Ongecorrigeerde z-waarden CORRIGEREN advh gecorrigeerde cluster-treshold
# nu gaan we kijken naar de effectieve Z-map.
img = nib.load(
    'D:/PhD Pieter De Clercq/paper4_VLSM_aphasia/output/final___31Jan2024_103046/Zfinal__broad40_all.nii')  # laad je data. Staat in mapje paper4_VLSM_aphasia, pas aan (heb dit lokaal laten lopen)
img_data = img.get_fdata()

# eerst: zet alle waarden met Z<2.33 = 0
thresholded_img_data = img_data > threshold

# vervolgens: selecteer je clusters die cluster thresholding surviven; dwz clusters met een size die groter is dan je cluster threshold van de permutation tests hierboven, dan weet je dat die cluster significant is!
labeled_clusters, num_clusters = ndimage.label(thresholded_img_data)
cluster_sizes = ndimage.sum(thresholded_img_data, labeled_clusters, range(num_clusters + 1))

print("Identified cluster sizes: {0}".format(
    cluster_sizes))  # hiermee print ik de groottes van alle clusters die het vindt; Alle clusters met een size groter dan de threshold van de permutation test blijven behouden!
print("Largest cluster size= {0} voxels".format(np.max(cluster_sizes)))  # dit is de grootste cluster die het vindt.

# nu kijken welke clusters de thresholding overleven:
surviving_clusters = np.where(cluster_sizes >= cluster_threshold)[0]
print("Number of clusters that survived threshold: {0}".format(len(surviving_clusters)))


## -- STEP 7: PLOTTEN
# plotten. Als er niks overleeft, dan plot ik niks.
if len(surviving_clusters) == 0:
    print("Nothing survived threshold, nothing to plot")
else:  # indien er wel iets overleeft, loop ik over alle clusters die cluster threshold overleven:
    for this_cluster in surviving_clusters:
        # clusters die het niet overleven, zet ik hieronder op 0. Dan ga ik er per cluster door. 3 lijntjes code hieronder
        surviving_clusters_mask = np.isin(labeled_clusters, this_cluster)
        surviving_clusters_data = img_data * surviving_clusters_mask
        surviving_clusters_data = surviving_clusters_data * -1  # die -1 hangt af of je geïnteresseerd bent in negatieve of positieve Z-waarden. Speel hiermee tot je zelf hebt wat je wil

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
    surviving_clusters_data = surviving_clusters_data * -1  # die -1 hangt af of je geïnteresseerd bent in negatieve of positieve Z-waarden. Speel hiermee tot je zelf hebt wat je wil

    surviving_clusters_img = nib.Nifti1Image(surviving_clusters_data, img.affine)
    texture = surface.vol_to_surf(surviving_clusters_img, fsaverage.pial_left)
    figure = plotting.plot_surf_stat_map(fsaverage.infl_left,
                                         texture, hemi='left',
                                         title='Surface plot left hemisphere of all clusters combined',
                                         colorbar=True, threshold=0.001, cmap='twilight',
                                         bg_map=fsaverage.sulc_left)

    # voorbeeld om op te slaan
    # figure.savefig('/media/pieter/7111-5376/vlsm_scratch/plots/cluster165_broad.svg')
    plotting.show();

## Mocht je ooit een mapje die ik hier aan maak (bvb, de Z-map maar dan met cluster threshold) willen opslaan (bvb om eens in te laden in MRIcroGL/mricron)
# gebruik dan die code en pas aan:

# nib.save(surviving_clusters_img, 'path/to/save/surviving_clusters.nii') #pas pad aan, doe comment weg