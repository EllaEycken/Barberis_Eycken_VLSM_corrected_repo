"""Script to calculate voxel-wise lesion distribution across brain areas, based on an atlas"""
## Description
# This script calculates the voxel-wise lesion distribution across brain areas in 2 ways.
# 1) calculate_lesion_distribution_cluster_based: how much % of the cluster lies in certain brain region?
# input: lesion masks and an atlas (to base brain areas on)
# output: a list of brain areas, the amount of lesioned voxels in those brain areas, and the relative % of the total
# clustervolume that resides in that brain region
# note: sum will add to 100% (total cluster = 100%)
# 2) calculate_lesion_distribution_atlas_based: how much % of a certain brain region (atlas) is occupied by the cluster?
# input: lesion masks and an atlas (to base brain areas on)
# output: a list of brain areas, the amount of lesioned voxels in those brain areas, and the relative percentage of that
# brain area that is lesioned (relative percentage of lesioned voxels in those brain areas).
# note: sum will NOT add to 100% (nl per brain area, X% of that brain area is occupied by the cluster; then for next area etc)
# Source: MRIcroGL, support by ChatGPT


## Imports
import nibabel as nib
import nilearn.image as ni
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

# from ella_phd_vlsm_project.ella_phd_vlsm_code.constants import harvard_brain_area_names


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


### PIETER PART: hoeveel % van cluster ligt in bepaalde hersenregio
# ------------------------------------------------------------------
# Source: L-drive --> Brain and Language --> Datamanagement --> PhD Pieter De Clercq --> voorbeeldscripts_info --> VLSM --> Plotting-statistiek onderste lijnen
# als je ooit zou willen kijken tot welke hersengebieden die cluster behoort, dan kun je die code toepassen. (zie mijn paper tabel met percentages)
# Hier gebruik ik de harvard-oxford atlas en overlap ik die met de cluster size. Je kan die atlas terugvinden in paper4_VLSM_aphasia
# output = percentage van die cluster dat in een bepaald hersengebied ligt.
# zoek dan op in die atlas (bvb in MRIcroGL) tot welke hersengebieden die waardes behoren.

# als je dit per cluster wil weten, dan is dat een leuk projectje voor jezelf om eens uit te proberen, gebruik hiervoor code hierboven als startpunt; maak een for-loop over alle individuele clusters heen.

def calculate_lesion_distribution_cluster_based(lesion_img_path, atlas_img_path, tables_DIR, variable, table_type

):
    """

    :param lesion_img_path: lesion masks
    :param atlas_img_path: atlas (to base brain areas on)
    :param tables_DIR: where to store the tables
    :param variable: for which behavioral variable the lesion distribution should be calculated
    :param table_type: which type of table (significant or non-significant) to calculate lesion distribution for
    :return:  how much % of the cluster lies in certain brain region? Will return a list of brain areas,
    the amount of lesioned voxels in those brain areas, and the relative % of the total clustervolume that resides in
    that brain region

    note: sum will add to 100% (total cluster = 100%) UNLESS some cluster parts end up in the 'background' region. Then,
    check the atlas function to see how many voxels are in the background region.
    """
    ## Initialiseer variabelen
    img = nib.load(lesion_img_path)
    lesion_data = img.get_fdata()  # negeer oranje stippellijn
    atlas = nib.load(
        atlas_img_path).get_fdata()

    ## Prepare the data
    # Create a binary mask where values > 0 are 1, and all others are 0
    data = (lesion_data > 0).astype(
        int)  # Create a binary mask where values >= zthreshold are 1, and all others are 0
    # lesion_data[lesion_data > 0] = 1  # would be of risk to overwrite former parts
    # how is the above done:
    # 1) It creates a boolean array where values greater than zthreshold are True and everything else is False.
    # 2) Then, .astype(int) converts the boolean array to integers, where True becomes 1 and False becomes 0.
    # Why: Z-waarden binary maken (niet meer geïnteresseerd in Z-waarden van de cluster, wel in welk hersengebied de cluster (= alles met z-waarde > 0) ligt
    # dus alleen interesse in OF een voxel behoort tot cluster (dan nl Z-waarde > 0 (behouden: maak van z-waarde een 1)) of niet (niet behouden: z-waarde blijft 0)

    # compute the total amount of voxels (to later calculate percentages)
    total_voxels = np.sum(data)
    print('total voxels', total_voxels)

    ## Bereken percentages
    data[data > 0] = atlas[data > 0]
    # plaats de waarden van atlas (aka voxels die behoren tot bepaalde hersengebieden (gebied 1, gebied 2...): 1 1 1 1 2 2 2 2 ...) in de plaats van de clusterwaarden (aka de data waarden die groter zijn dan 0 aka 1)
    # zodat 'data' nu een representatie is van alleen die hersengebied-waarden (111112222888555) die behoren tot de cluster, rest van data is 0
    uniques = np.unique(data)
    # obv de data (eigenlijk nu die hersengebieden die behoren tot de cluster), maak een lijst van de VERSCHILLENDE hersenregio's die in de cluster liggen
    # door uniques te nemen: (111112222888555) als data, dan is hersengebied 1, 2, 8 en 5 in de cluster vallen
    uniques = uniques[1:]
    # note dat uniques ook de waarde 0 uit de data extraheert terwijl 0 niet overeenstemt met een hersengebied, dus haal 0 uit uniques (pas tellen vanaf 1)
    list_voxels = []
    list_percentages = []
    list_brain_areas = []

    for i in uniques: # ga over de verschillende hersengebieden die in de cluster vallen
        newData = data[data == i]
        # uit de data (die hersengebieden die behoren tot de cluster), sla alle waarden gelijk aan i (aka voxels die tot bepaald hersengebied i behoren) op in newData
        # zodat newData = lijst van alle voxels uit de cluster die tot dat hersengebied behoren
        newData[newData > 0] = 1
        # zet alle waarden in newData (iiiiii, met i = 1...laatste hersengebied) om in de waarde 1 (11111)
        percentage_i = (np.sum(newData) / total_voxels) * 100
        # percentage: som van alle voxels uit dat hersengebied die in de cluster liggen / totaal aantal voxels in de cluster
        list_percentages.append(percentage_i)
        list_voxels.append(np.sum(newData))
        list_brain_areas.append(i)

        # print("Hersengebied: {0}".format(
            # i))  # dit is hersengebied index van die atlas. Laad die atlas eens in in mricrogl en zoek uit welke index tot welk hersengebied behoort.
        # print("Percentage: {0}%".format(percentage_i)  # percentage


    ## Sla resultaten op in dataframe
    # ---- STEP 1: initialize data dictionary of lists ----
    data_distribution_cluster = {
            "Brain Region Number": list_brain_areas,
            "Brain Region Name":[harvard_brain_area_names[int(i)] for i in list_brain_areas],
            "Cluster Voxels": list_voxels,
            "% of Cluster in this region": list_percentages,
        }

    # ---- STEP 2: Create Dataframe ----
    df_distribution_cluster = pd.DataFrame(data_distribution_cluster)
    #  https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
    pd.set_option("display.max.columns", None)
    df_distribution_cluster.style.background_gradient().set_caption("Cluster Distribution across Brain Areas (Harvard)")


    # ---- STEP 3: save Dataframe as excel in table data directory
    # TODO: pas naam van excel file zelf aan (.xlsx niet vergeten)
    file_name = os.path.join(tables_DIR, f"df_distribution_cluster_{variable}_{table_type}.xlsx")
    # from: f"L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/VLSM_factored_permTest_5000_Factor_4_{name_add}_check.svg")
    df_distribution_cluster.to_excel(file_name, index=False)
        # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_distribution_cluster



### OWN PART: hoeveel % van die hersenregio (in atlas) is ingenomen door cluster
# -------------------------------------------------------------------------
def calculate_lesion_distribution_atlas_based(lesion_img_path, atlas_img_path, tables_DIR, variable, table_type
                                              ):
    """

    :param lesion_img_path: lesion masks
    :param atlas_img_path: atlas (to base brain areas on)
    :param tables_DIR: where to store the tables
    :param variable: for which behavioral variable the lesion distribution should be calculated
    :param table_type: which type of table (significant or non-significant) to calculate lesion distribution for
    :return:  how much % of a certain brain region (atlas) is occupied by the cluster? a list of brain areas,
    the amount of lesioned voxels in those brain areas, and the relative percentage of that brain area that is lesioned
    (relative percentage of lesioned voxels in those brain areas)

    note: sum will NOT add to 100% (nl per brain area, X% of that brain area is occupied by the cluster;
    then for next area etc)
    """
    ## Initialize
    # Load the lesion mask and atlas images
    lesion_img = nib.load(lesion_img_path)
    atlas_img = nib.load(atlas_img_path)

    # Get the data arrays for the lesion and atlas
    lesion_data = lesion_img.get_fdata()
    atlas_data = atlas_img.get_fdata()


    ## Prepare the data
    # Create a binary mask where values > 0 are 1, and all others are 0 (based on ChatGPT)
    lesion_data = (lesion_data > 0).astype(
        int)  # Create a binary mask where values >= zthreshold are 1, and all others are 0 (based on ChatGPT)
    # lesion_data[lesion_data > 0] = 1  # would be of risk to overwrite former parts
    # how is the above done:
    # 1) It creates a boolean array where values greater than zthreshold are True and everything else is False.
    # 2) Then, .astype(int) converts the boolean array to integers, where True becomes 1 and False becomes 0.
    # Why: Z-waarden binary maken (niet meer geïnteresseerd in Z-waarden van de cluster, wel in welk hersengebied de cluster (= alles met z-waarde > 0) ligt
    # dus alleen interesse in OF een voxel behoort tot cluster (dan nl Z-waarde > 0 (behouden: maak van z-waarde een 1)) of niet (niet behouden: z-waarde blijft 0)
    print('data')
    # get the atlas data


    ## Calculate percentages
    # Get unique region labels from the atlas (Ella: not (excluding 0, which usually represents background))
    list_brain_areas = np.unique(atlas_data)
    #region_labels = region_labels[region_labels != 0]

    # Initialize a dictionary to store results
    lesion_distribution = {}
    list_total_voxels_per_region = []
    list_lesioned_voxels_per_region = []
    list_percentages = []


    # Iterate over each region in the atlas
    for region in list_brain_areas:
        # Get a mask of the current region in the atlas
        region_mask = atlas_data == region

        # Calculate the total number of voxels in this region
        total_voxels_in_region = np.sum(region_mask)

        # Calculate the number of lesioned voxels in this region
        lesioned_voxels_in_region = np.sum(region_mask * lesion_data)

        # Calculate the percentage of lesioned voxels in this region
        if total_voxels_in_region > 0:
            lesion_percentage = (lesioned_voxels_in_region / total_voxels_in_region) * 100
        else:
            lesion_percentage = 0.0

        # Store the results in the lists
        list_total_voxels_per_region.append(total_voxels_in_region)
        list_lesioned_voxels_per_region.append(lesioned_voxels_in_region)
        list_percentages.append(lesion_percentage)


    ## Save results in dataframe
    # ---- STEP 1: initialize data dictionary of lists ----
    data_distribution_atlas = {
        "Brain Region Number": list_brain_areas,
        "Brain Region Name": [harvard_brain_area_names[int(i)] for i in list_brain_areas],
        "Total Region Voxels": list_total_voxels_per_region,
        "Lesioned Region Voxels": list_lesioned_voxels_per_region,
        "% of regional overlap with cluster": list_percentages,
    }

    # ---- STEP 2: Create Dataframe ----
    df_distribution_atlas = pd.DataFrame(data_distribution_atlas)
    #  https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
    pd.set_option("display.max.columns", None)
    df_distribution_atlas.style.background_gradient().set_caption(
        "Brain Area (parts) (Harvard) overlap with cluster")
    # discard those rows of df that have 0% overlap with the cluster
    df_distribution_atlas = df_distribution_atlas[df_distribution_atlas["% of regional overlap with cluster"] != 0.0]
    # discard first row if that row is the background region:
    # if df_distribution_atlas["Brain Region Number"][0] == 0: # number 0 in Harvard atlas = background region
        # df_distribution_atlas = df_distribution_atlas.iloc[1:]
        # df_distribution_atlas.reset_index(drop = True, inplace = True)

    # ---- STEP 3: save Dataframe as excel in table data directory
    # TODO: pas naam van excel file zelf aan (.xlsx niet vergeten)
    file_name = os.path.join(tables_DIR, f"df_region_overlap_{variable}_{table_type}_with background.xlsx")
    df_distribution_atlas.to_excel(file_name, index=False)
    # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_distribution_atlas



def locate_peak_value(lesion_img_path, atlas_img_path
                        ):
    # Load the lesion image and Harvard Oxford Atlas
    lesion_img = nib.load(lesion_img_path)
    atlas_img = nib.load(atlas_img_path)

    # Get the data arrays from the NIfTI images
    lesion_data = lesion_img.get_fdata()
    atlas_data = atlas_img.get_fdata()

    # Find the index of the voxel with the highest value (eg z-value)
    highest_value_index = np.unravel_index(np.argmax(lesion_data), lesion_data.shape)
    # highest_z_index = np.argmax(z_data) gives you the index of the voxel with the highest z-value.
    # This is a 1D index for the flattened array, which corresponds to a voxel in the 3D grid.
    # then: Since np.argmax will return a flat index (for a 3D array), you need to convert it to
    # multi-dimensional coordinates (x, y, z) with np.unravel_index().
    # Using the coordinates obtained from np.unravel_index, you can access the corresponding voxel in the atlas_data
    # and get the region index for that voxel.
    # source: chatgpt

    # Get the region index from the Harvard Oxford Atlas at that voxel
    region_index = atlas_data[highest_value_index]
    region_name = harvard_brain_area_names[int(region_index)]

    # print results
    print('highest_value',np.max(lesion_data) )  # not argmax: argmax gives coordinates (in original file) of max value
    print('highest_value_index', highest_value_index)
    print('region_index', region_index)
    print('region_name', region_name)

    return highest_value_index, region_index, region_name



def make_histogram(distribution_excel
                         ):
    # Read in excel file
    data = pd.read_excel(distribution_excel)

    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    # df = df.drop(index=0).reset_index(drop=True)  # drop the first row

    # Set the lists of valus, depending on the column values
    x_values = df.iloc[:, 2 ]
    y_values_cluster = df.iloc[:, 3]
    y_values_atlas = df.iloc[:, 4]

    # Set the width of the bars
    bar_width = 0.35

    # Set the x positions for the bars
    x = range(len(x_values))

    # Create the bar graph
    plt.bar(x, y_values_cluster,
            width = bar_width,
            label = '% of cluster overlapping with the region',
            align = 'center')
    plt.bar([i + bar_width for i in x], y_values_atlas,
            width = bar_width,
            label = '% of region overlapping with the cluster',
            align = 'center')

    # Add labels and title
    plt.xlabel('Brain region')
    plt.ylabel('Percentage')
    plt.title('Overlap between the lexical-semantic cluster and brain regions')
    plt.xticks(
        ticks = [i+bar_width/2 for i in x],
        labels = x_values,
        rotation=45)
    plt.legend()

    # Add values above bars
    for index, value in enumerate(y_values_cluster):
        plt.text(index, value + 0.5, str(round(value, 1)), ha='center', va='bottom', size = 'xx-small')
    for index, value in enumerate(y_values_atlas):
        plt.text(index + bar_width, value + 0.5, str(round(value, 1)), ha='center', va='bottom', size = 'xx-small')

    # Position the legend on the right side
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fontsize = 'x-small')
    plt.tight_layout()

    # Optionally save the figure
    # TODO: change the name of the plot (but don't forget .png )
    plt.savefig(
        f"C:/Users/u0146803/Documents/VLSM_regions/VLSM_Factor_3_distribution_histogram_7.png", dpi = 1200)
    # from plt.savefig(
    #         os.path.join(output_dir, "figures", f"feature_importances_{label}_{interview_part}.png"), dpi = 300)
    plt.show()



if __name__ == "__main__":
    """ FOR VLSM OUTPUT CLUSTERS """
    # Define the file paths for the lesion mask and atlas image
    ## Initialize some variables
    # TODO: Vul dit zelf aan
    variable = "ANTAT_TTR"
    cluster_is_significant = True  # switch to false if not sign cluster
    # path_to_VLSM_folder = "C:/Users/u0146803/Documents/VLSM_masterthesis"
    path_to_VLSM_folder = "C:/Users/u0146803/Documents/VLSM_regions"
    corrected_VLSM_output_folder_name = "VLSM_ANTAT_perm_1000_lesionregr_MCcorrected"
    threshold_abs = 1.645

    # Pas dit niet zelf aan
    if cluster_is_significant:
        cluster_type = "surviving_clusters"
        table_type = 'sign'
    else:
        cluster_type = "nonsign_cluster"
        table_type = 'nonsign'

    lesion_img_path = os.path.join(path_to_VLSM_folder, 'output', corrected_VLSM_output_folder_name,
                                    f"Z{cluster_type}_{variable}.nii")
    # "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/output/VLSM_factored_withMonthsPO_perm_5000_lesionregr_MCcorrected/nonsign_largest_cluster_Factor_4.nii"
    # "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/maps/sub-01.nii"
    # Path to cluster img (nifti-file), make sure to use / instead of \; and add .nii extension

    atlas_img_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/helper files/harvard_new.nii"
        # 'D:/PhD Pieter De Clercq/paper4_VLSM_aphasia/harvard_new.nii'  # atlas kan je terugvinden in paper4_VLSM_aphasia
    # Note Pieter: het kan zijn dat je later errors krijgt omdat je dimensies van de atlas niet overeenkomen met je dimensies van je letselmapje.
    # Om dit op te lossen: open een letselmapje in MRIcroGL. Dan Draw --> open VOI. Open als nii. Open die harvard_new.nii zoals op de L-schijf staat
    # Direct erna: draw --> save VOI (opslaan als nifti). harvard_new.nii overwriten. Je doet dus niks, enkel openen en weer opslaan sluiten, maar door te openen bovenop het letselmapje,
    # werk je wel in dezelfde dimensies. Werk dan met die nieuwe harvard_new.nii in dit script. Zie ook de manual van de VLSM op de L-schijf bij punt 6. Troubleshooting.
    tables_DIR = os.path.join(path_to_VLSM_folder, 'tables')
        # "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/tables"
    # Path to tables
    distribution_excel = os.path.join(path_to_VLSM_folder, 'df_distribution_short_Factor_3.xlsx')


    # Calculate the cluster distribution
    """
    calculate_lesion_distribution_cluster_based(
        lesion_img_path,
        atlas_img_path,
        tables_DIR,
    )

    # Calculate the lesion distribution
    calculate_lesion_distribution_atlas_based(
        lesion_img_path,
        atlas_img_path,
        tables_DIR,
    )

    locate_peak_value(
        lesion_img_path,
        atlas_img_path,
    )
    
    make_histogram(distribution_excel)
    """

    """ FOR LESION OVERLAP CLUSTER """
    # unthresh_lesionOverlapMask_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/lesionOverlap_Mask.nii"
    thresh_lesionOverlapMask_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/thresh_lesionOverlap_Mask.nii"

    # Load images
    lesion_img = nib.load(thresh_lesionOverlapMask_path)  # your lesion overlap image
    atlas_img = nib.load(
        "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/helper files/harvard_new.nii")

    # Resample lesion image to atlas resolution and space
    lesion_resampled = ni.resample_to_img(lesion_img, atlas_img, interpolation='nearest')
    resampled_thresh_lesionOverlapMask_path = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/figures/resampled_thresh_lesionOverlap_Mask.nii"
    nib.save(lesion_resampled, resampled_thresh_lesionOverlapMask_path)

    tables_IANSA_DIR = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/VLSM/VLSM_IANSA/tables"
    variable_lesionOverlap = "lesion_overlap"  # name of the variable in the table
    table_type_lesionOverlap = 'threshold_5'
    """
    calculate_lesion_distribution_cluster_based(
        resampled_thresh_lesionOverlapMask_path,
        atlas_img_path,
        tables_IANSA_DIR,
        variable_lesionOverlap,
        table_type_lesionOverlap
    )
    """

    calculate_lesion_distribution_atlas_based(
        resampled_thresh_lesionOverlapMask_path,
        atlas_img_path,
        tables_IANSA_DIR,
        variable_lesionOverlap,
        table_type_lesionOverlap
    )
