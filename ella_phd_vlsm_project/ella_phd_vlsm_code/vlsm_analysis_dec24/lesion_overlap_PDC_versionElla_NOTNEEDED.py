import nibabel as nib
import numpy as np
from scipy import ndimage
from nilearn import plotting, datasets
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import surface

### Voorbeeldscript om een lesion overlap map te maken over proefpersonen heen.

# als je errors krijgt, dan is dit vaak 1) ofwel door packages, installeer dan eerdere versies, of 2) doordat je problemen hebt met je dimensies van je beelden die over
# proefpersonen heen niet overeenkomen. Lees de manual van de VLSM onderaan, daar heb ik meer info geplaatst.


### Voorbereiding
# ----------------
# TODO: pas Path aan naar folder met lesion files (= subj-XX.nii files)
path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/maps/"
    #"D:/PhD Pieter De Clercq/voorbeeldscripts_info/voorbeeldData_lesion_overlap_map/" # hierin staat lijstje van letsel-files (lesion extracted files per subject eg sub-060.nii)
    # vergeet niet: / ipv \ ; en / op einde path-naam
allFiles = [f for f in os.listdir(path) if f.endswith('.nii')] # lijst alle files (=alle proefpersonen)


### alle data verzamelen
# -------------------------
# alle individuele data zijn binary files, 3D
# om lesionOverlap te maken, maak ik een som van alle binary files

i = 0 # zal de maximale lesion overlap zijn (zv)
# bvb Pieter  is de hoogste som = 7, dus een maximum lesionOverlap van 7 proefpersonen zal hier (zie output) dus i = 7 worden
# aanvulling Ella: i zal 49 worden (49 participanten)
for subject in allFiles:
    path_subject = path + subject # concateneert het pad
    img = nib.load(path_subject)
    if i == 0:
        lesionOverlap = np.round(np.array(img.get_fdata()))
    else:
        lesionOverlap = lesionOverlap + np.round(np.array(img.get_fdata()))
    i += 1

print("Shape van individuele data: {0}".format(lesionOverlap.shape))
print("Unieke waarden: {0}".format(np.unique(lesionOverlap)))
print("Max lesion overlap: {0}".format(np.max(lesionOverlap)))


### Eerste voorbeeld plotjes: via Z-coordinaat.
# ---------------------------------------------
# Dit soort plotje kun je makkelijk ook in MRIcroGL maken (manueel, geen code)
# hier enkele voorbeelden van coordinaten.

# TODO: speel met tresholds (zie lijnen hieronder) en andere parameters
# meer info te vinden op: https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_roi.html

## Plot zonder tresholds (= plot van overlap tussen alle participanten, zonder beperking op 'minimale overlap')
plotting.plot_roi(nib.Nifti1Image(lesionOverlap, img.affine), cut_coords=(-20,-4,12,20, 28, 36), display_mode='z',colorbar=True, cmap = 'cubehelix')
# img.affine = affine argument dat nilearn nodig heeft. die img werd geladen in vorige for-loop hierboven (is dus van 1 participant).
# maar aangezien alle beelden zelfde dimensies hebben, is die affine ook voor iedereeen hetzelfde.
# speel met andere argumenten voor colorbar etc... ook met threshold spelen. Hier geen threshold, hieronder wel thresholds om verschil aan te tonen.

# voorbeeld hoe je opslaat:
# TODO: pas pad aan om te saven (vergeet niet .svg extensie en argumenten te vermelden in naamgeving!)
plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_noT.svg")
    #'/media/pieter/7111-5376/2022-fMRI/secondLevel/plots/lesionOverlap_noT.svg')

## Plot met tresholds (= plot van overlap, waarbij treshold = 'minimum deze hoeveelheid mensen (absoluut aantal) moeten overlap hebben om als overlap getoond te worden in de plot')
# vb treshold = 1: vanaf lesionoverlap van 1 persoon (dus gewoon letsel van ieder persoon) wordt dit weergegeven op de plot
# vb treshold = 2: vanaf lesionoverlap tussen 2 personen wordt dit weergegeven op de plot (dus letsels die slechts bij 1 persoon voorkomen, worden niet getoond)
plotting.plot_roi(nib.Nifti1Image(lesionOverlap, img.affine), cut_coords=(-20,-4,12,20, 28, 36), display_mode='z',colorbar=True, cmap = 'cubehelix', threshold=2)
plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_Tresh2.svg")
# voor Ella's studie: VLSM analyse minimal_lesion_overlap = 10% => absolute aantal = 5 (van n tot = 50): kies treshold = 5: vanaf 5 personen overlap, wordt dit getoond in plot
plotting.plot_roi(nib.Nifti1Image(lesionOverlap, img.affine), cut_coords=(-20,-4,12,20, 28, 36), display_mode='z',colorbar=True, cmap = 'cubehelix', threshold=5)
plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_Tresh5.svg")


### Tweede voorbeeld: surface maps
# ----------------------------------
# heeft mijn voorkeur en kan niet (goed) in MRIcroGL

# texture achtergrond nodig voor surface map
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
texture = surface.vol_to_surf(nib.Nifti1Image(lesionOverlap, img.affine), fsaverage.pial_left)

## Plot zonder tresholds (= plot van overlap tussen alle participanten, zonder beperking op 'minimale overlap')
# TODO: speel opnieuw met treshold en andere parameters
figure = plotting.plot_surf_roi(fsaverage.infl_left,
                                     texture, hemi='left',
                                     title='Surface left hemisphere',
                                     colorbar=True, #threshold=2,
                                     bg_map=fsaverage.sulc_left, cmap = 'cubehelix', vmax = 20) # speel met argumenten.

# TODO: pas pad aan om te saven (vergeet niet .svg extensie en argumenten te vermelden in naamgeving!)
plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_surf_noT.svg")
    #'/media/pieter/7111-5376/2022-fMRI/secondLevel/plots/lesionOverlap_surf_noT.svg')
plotting.show();

## Plot met tresholds (= plot van overlap, waarbij treshold = 'minimum deze hoeveelheid mensen (absoluut aantal) moeten overlap hebben om als overlap getoond te worden in de plot')
# TODO: speel opnieuw met treshold en andere parameters
figure = plotting.plot_surf_roi(fsaverage.infl_left,
                                     texture, hemi='left',
                                     title='Surface left hemisphere',
                                     colorbar=True, threshold=int(2),
                                     bg_map=fsaverage.sulc_left, cmap = 'cubehelix', vmax = 20) # speel met argumenten.

# TODO: pas pad aan om te saven (vergeet niet .svg extensie en argumenten te vermelden in naamgeving!)
plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_surf_Tresh2.svg")
    #'/media/pieter/7111-5376/2022-fMRI/secondLevel/plots/lesionOverlap_surf_noT.svg')
plotting.show();
# TODO: speel opnieuw met treshold en andere parameters
figure = plotting.plot_surf_roi(fsaverage.infl_left,
                                     texture, hemi='left',
                                     title='Surface left hemisphere',
                                     colorbar=True, threshold=int(5),
                                     bg_map=fsaverage.sulc_left, cmap = 'cubehelix', vmax = 20) # speel met argumenten.

# TODO: pas pad aan om te saven (vergeet niet .svg extensie en argumenten te vermelden in naamgeving!)
plt.savefig("L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_surf_Tresh5.svg")
    #'/media/pieter/7111-5376/2022-fMRI/secondLevel/plots/lesionOverlap_surf_noT.svg')
plotting.show();