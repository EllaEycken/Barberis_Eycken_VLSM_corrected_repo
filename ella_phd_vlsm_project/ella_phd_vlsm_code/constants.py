"""Constants for the project."""

# Imports
import os

# Load environment variables
# from dotenv import load_dotenv

# load_dotenv()

## PATHS


## VARIABLES
# A list of the brain regions numbered from 0 to... in the Harvard-Oxford Atlas
harvard_brain_area_names = [ # source: https://scalablebrainatlas.incf.org/services/labelmapper.php?template=HOA06
'[background]',
'Frontal Pole',
'Insular Cortex',
'Superior Frontal Gyrus',
'Middle Frontal Gyrus',
'Inferior Frontal Gyrus, pars triangularis',
'Inferior Frontal Gyrus, pars opercularis',
'Precentral Gyrus',
'Temporal Pole',
'Superior Temporal Gyrus, anterior division',
'Superior Temporal Gyrus, posterior division',
'Middle Temporal Gyrus, anterior division',
'Middle Temporal Gyrus, posterior division',
'Middle Temporal Gyrus, temporooccipital part',
'Inferior Temporal Gyrus, anterior division',
'Inferior Temporal Gyrus, posterior division',
'Inferior Temporal Gyrus, temporooccipital part',
'Postcentral Gyrus',
'Superior Parietal Lobule',
'Supramarginal Gyrus, anterior division',
'Supramarginal Gyrus, posterior division',
'Angular Gyrus',
'Lateral Occipital Cortex, superior division',
'Lateral Occipital Cortex, inferior division',
'Intracalcarine Cortex',
'Frontal Medial Cortex',
'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
'Subcallosal Cortex',
'Paracingulate Gyrus',
'Cingulate Gyrus, anterior division',
'Cingulate Gyrus, posterior division',
'Precuneous Cortex',
'Cuneal Cortex',
'Frontal Orbital Cortex',
'Parahippocampal Gyrus, anterior division',
'Parahippocampal Gyrus, posterior division',
'Lingual Gyrus',
'Temporal Fusiform Cortex, anterior division',
'Temporal Fusiform Cortex, posterior division',
'Temporal Occipital Fusiform Cortex',
'Occipital Fusiform Gyrus',
'Frontal Operculum Cortex',
'Central Opercular Cortex',
'Parietal Operculum Cortex',
'Planum Polare',
"Heschl's Gyrus (includes H1 and H2)",
'Planum Temporale',
'Supracalcarine Cortex',
'Occipital Pole',
'#6A7F00',
'#FFA900',
'#7F5400',
'#FF2A00',
'#7F1500',
'#FF0054',
'#7F002A',
'#FF00D4',
'#7F006A',
'#5500FF',
'#2A007F'
]