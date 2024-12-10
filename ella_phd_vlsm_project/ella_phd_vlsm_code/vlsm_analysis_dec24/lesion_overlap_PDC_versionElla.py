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

path = "D:/PhD Pieter De Clercq/voorbeeldscripts_info/voorbeeldData_lesion_overlap_map/"
allFiles = [f for f in os.listdir(path) if f.endswith('.nii')] # lijst alle files (=alle proefpersonen)

### alle data verzamelen
# alle individuele data zijn binary files, 3D
# om lesionOverlap te maken, maak ik een som van alle binary files
# in dit voorbeeldje is de hoogste som = 7, dus een maximum lesionOverlap van 7 proefpersonen hier (zie output)
i = 0
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
# Dit soort plotje kun je makkelijk ook in MRIcroGL maken (manueel, geen code)
# hier enkele voorbeelden van coordinaten.

plotting.plot_roi(nib.Nifti1Image(lesionOverlap, img.affine), cut_coords=(-20,-4,12,20, 28, 36), display_mode='z',colorbar=True, cmap = 'cubehelix')
# img.affine = affine argument dat nilearn nodig heeft. die img werd geladen in vorige for-loop hierboven (is dus van 1 participant).
# maar aangezien alle beelden zelfde dimensies hebben, is die affine ook voor iedereeen hetzelfde.
# speel met andere argumenten voor colorbar etc... ook met threshold spelen. Hier geen threshold, hieronder wel thresholds om verschil aan te tonen.

# voorbeeld hoe je opslaat:
#plt.savefig('/media/pieter/7111-5376/2022-fMRI/secondLevel/plots/lesionOverlap_noT.svg')

plotting.plot_roi(nib.Nifti1Image(lesionOverlap, img.affine), cut_coords=(-20,-4,12,20, 28, 36), display_mode='z',colorbar=True, cmap = 'cubehelix', threshold=1)

plotting.plot_roi(nib.Nifti1Image(lesionOverlap, img.affine), cut_coords=(-20,-4,12,20, 28, 36), display_mode='z',colorbar=True, cmap = 'cubehelix', threshold=2)


### Tweede voorbeeld: surface maps
# heeft mijn voorkeur en kan niet (goed) in MRIcroGL

# texture achtergrond nodig voor surface map
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
texture = surface.vol_to_surf(nib.Nifti1Image(lesionOverlap, img.affine), fsaverage.pial_left)

figure = plotting.plot_surf_roi(fsaverage.infl_left,
                                     texture, hemi='left',
                                     title='Surface left hemisphere',
                                     colorbar=True, #threshold=2,
                                     bg_map=fsaverage.sulc_left, cmap = 'cubehelix', vmax = 6) # speel met argumenten.
#plt.savefig('/media/pieter/7111-5376/2022-fMRI/secondLevel/plots/lesionOverlap_surf_noT.svg')
plotting.show();