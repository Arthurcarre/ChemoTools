"""
#####################################################
##                                                 ##
##       -- STREAMLIT CHEMOTOOLS V1.1 --           ##
##                                                 ##
#####################################################
"""

import os
import copy, random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

###############################################
#                                             #
#            DEFINITIONS SECTION              #
#                                             #                                                      
###############################################

def RMSD_listes(sample) :
    
    """
    -- DESCRIPTION --
    Create a list containing the index lists of molecules that share an RMSD less than the RMSDtarget.
    """
           
    remove_list = list(range(len(sample)))
    output_listes = []

    for individuali in sample :
        subliste = []
        for j in remove_list :
            if rdMolAlign.CalcRMS(st.session_state.mols[individuali],st.session_state.mols[sample[j]]) < RMSD_Target :
                subliste.append(j)
                remove_list.remove(j)
        output_listes.append(subliste[:])

    while [] in output_listes :
        for i, output_liste in enumerate(output_listes) :
            try :
                if output_liste == [] :
                    del output_listes[i]
            except :
                pass
    return output_listes

def gather_groups_RMSD(input_liste) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    liste = []
    for i, groupei in enumerate(input_liste) :
        p = 0
        for j, groupej in enumerate(input_liste) :
            RMSD = []
            n = -1
            if i != j :
                if len(groupej) <= len(groupei):
                    for k in groupej :
                        n += 1
                        RMSD.append(rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[k]],
                                                       st.session_state.mols[st.session_state.samples[input_liste[i][n]]]))
                if len(groupej) > len(groupei):
                    for k in groupei :
                        n += 1
                        RMSD.append(rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[k]],
                                                       st.session_state.mols[st.session_state.samples[input_liste[j][n]]]))
            if i != j :
                #print(f"La moyenne des RMSD entre le groupe {i} et le groupe {j} est de {np.mean(RMSD)}")
                if np.mean(RMSD) < RMSD_Target :
                    p = 1
                    copy1 = copy.deepcopy(groupei)   
                    copy2 = copy.deepcopy(groupej)
                    for l in range(len(copy2)) :
                        copy1.append(copy2.pop())
                    liste.append(copy1)
        if p == 0 : 
            copy1 = copy.deepcopy(groupei)
            liste.append(copy1)
    return liste

def sorted_list_lengroups(filtered_gather_groups_RMSD) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    lengroups_list = [len(i) for i in filtered_gather_groups_RMSD]
    lengroups_list.sort(reverse=True)

    return [x for y in lengroups_list for x in filtered_gather_groups_RMSD if y == len(x)]

def get_unique_numbers(numbers):
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    unique = []

    for number in numbers:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique

def get_filtered_liste(sorted_list_lengroups_var) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    filtered_numbersduplicate_liste = []

    for sorted_list_lengroup in sorted_list_lengroups_var : 
        for j in sorted_list_lengroup :
            filtered_numbersduplicate_liste.append(j)

    filtered_liste = get_unique_numbers(filtered_numbersduplicate_liste)
    return filtered_liste


def get_groups_inside_list(liste) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    liste1 = copy.deepcopy(liste)
    liste2 = copy.deepcopy(liste)
    liste3 = []
    for i, n in enumerate(liste1) :
        try :
            r = rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[n]],
                                   st.session_state.mols[st.session_state.samples[liste1[i+1]]])
        except IndexError :
            pass
        if r > 3 :
        #try :
            #print(f"En position {i}, une séparation de groupe se trouve entre {n} et {liste1[i+1]}", 
            #      f"pour un RMSD de {r} A")
        #except IndexError :
            #pass
            try :
                liste3.append(liste2[:liste2.index(liste1[i+1])])
                del liste2[:liste2.index(liste1[i+1])]
            except IndexError :
                pass
            #print(f"liste 2 --> {liste2}")
    return liste3

def improve_sort(liste, n) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    for i in range(n) :
        liste = get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(get_groups_inside_list(liste))))
    return liste

def get_predominant_poses(input_list) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    k = 0 
    for group in input_list : 
        if len(group)/len(st.session_state.samples)*100 > 1/17*len(st.session_state.samples) :
            k += 1

    predominant_poses = input_list[:k]
    st.info(f"There are {k} predominant poses among all poses.\n")
    st.session_state.numbers_conformation = k
    st.session_state.predominant_poses = predominant_poses

    for i, predominant_pose in enumerate(predominant_poses) :
        st.write(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.samples)*100:.1f}" 
              f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.samples)} poses in total.")
    return predominant_poses

def get_sample_indice_best_score(predominant_poses) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    indice_best_score = []
    m = -1

    for i in predominant_poses :
        m += 1
        n = -1
        suppl_2 = []
        for j in i :
            n += 1
            suppl_2.append(st.session_state.suppl[st.session_state.samples[predominant_poses[m][n]]])
        PLPscore = [x.GetProp(st.session_state.score) for x in suppl_2]
        best_score = 0
        for k in PLPscore :
            if float(k) > float(best_score) :
                best_score = k
                indice = PLPscore.index(k)
        indice_best_score.append(indice)
    return indice_best_score

def get_Sample_Best_PLPScore_Poses() :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    conformations = (st.session_state.suppl[st.session_state.samples[group[indice]]]
                     for group, indice in zip(st.session_state.sample_predominant_poses,
                                              st.session_state.sample_indice_best_score))

    with Chem.SDWriter('Sample_Best_PLPScore_Poses.sdf') as w:
        for m in conformations:
            w.write(m)
    w.close()

    for k, group in enumerate(st.session_state.sample_predominant_poses) :
        with Chem.SDWriter(f'Sample_Conformation{k+1}.sdf') as w : 
            for m in group :
                mol = st.session_state.suppl[st.session_state.samples[m]]
                w.write(mol)
        w.close()

    a_supprimer = []
    for i, indice in enumerate(st.session_state.sample_indice_best_score) :
        subliste = []
        with Chem.SDWriter(f'À_supprimer{i+1}.sdf') as w: #Create a sdf file
            for j, mol in enumerate(st.session_state.suppl) :
                if rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[st.session_state.sample_predominant_poses[i][indice]]], mol) < 2 :
                    subliste.append(st.session_state.suppl[j])
                    w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
            copy1 = copy.deepcopy(subliste)
            a_supprimer.append(copy1)

    indice_best_score = []
    for group in a_supprimer : 
        PLPscore = [x.GetProp(st.session_state.score) for x in group]
        best_score = 0
        for k in PLPscore :
            if float(k) > float(best_score) :
                best_score = k
                indice = PLPscore.index(k)
        indice_best_score.append(indice)

    best_PLPScore_poses = (group[indice] for group, indice in zip(a_supprimer, indice_best_score))

    with Chem.SDWriter(f'Best_PLPScore_Poses.sdf') as w:
        for m in best_PLPScore_poses:
            w.write(m)
    w.close()
    
    for i, indice in enumerate(st.session_state.sample_indice_best_score) :
        os.remove(f'À_supprimer{i+1}.sdf')

def get_data_frame_best_poses(best_PLP_poses) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    
    columns = list(range(len(best_PLP_poses)))
    for i in columns :
        columns[i] = f"Pose n°{i+1}"

    index = list(range(len(best_PLP_poses)))
    for i in index :
        index[i] = f"Pose n°{i+1}"
    
    array = np.ones(shape=(len(best_PLP_poses),len(best_PLP_poses)))

    for i, moli in enumerate(best_PLP_poses) :
        for j, molj in enumerate(best_PLP_poses) :
            array[i, j] = rdMolAlign.CalcRMS(moli, molj)

    data_frame = pd.DataFrame(array, index=index, columns=columns)

    st.write("\nIn order to check that each group is different from each other, a table taking " 
          "the first individual from each group and calculating the RMSD between each was constructed :\n")
    st.dataframe(data_frame)
    st.write("RMSD value between each representative of each conformation.")
    st.session_state.df1 = data_frame
    return data_frame

def get_histogramme_sample_bestPLP(best_PLP_poses) :
   
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
    
    sns.set_style('whitegrid')
    sns.set_context('paper')
    
    if len(best_PLP_poses) == 1 :
        sdf_to_hist = [rdMolAlign.CalcRMS(best_PLP_poses[0], mol) for mol in st.session_state.mols]
        fig, ax = plt.subplots(len(best_PLP_poses), 1, figsize=(15, 0.2*len(best_PLP_poses)*9))
        a, b, c = 0, 0, 0
        for RMSD in sdf_to_hist : 
            if RMSD < 2 :
                a += 1
            if RMSD < 3 :
                b += 1
            if RMSD < 4 :
                c += 1
        ax.hist(sdf_to_hist, bins =100, label = "Conformation n°1") #Create an histogram to see the distribution of the RMSD of the sample
        ax.axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
        ax.annotate(a, (1.5, 0.05*len(st.session_state.mols)), fontsize=15)
        ax.axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
        ax.annotate(b-a, (2.5, 0.05*len(st.session_state.mols)), fontsize=15)
        ax.axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
        ax.annotate(c-b, (3.5, 0.05*len(st.session_state.mols)), fontsize=15)
        ax.legend(loc='best', shadow=True, markerfirst = False)
    else :
        sdf_to_hist = ([rdMolAlign.CalcRMS(best_PLP_pose, mol) for mol in st.session_state.mols] for best_PLP_pose in best_PLP_poses)
        fig, ax = plt.subplots(len(best_PLP_poses), 1, figsize=(15, 0.2*len(best_PLP_poses)*9))
        for z, group in enumerate(sdf_to_hist) :
            a, b, c = 0, 0, 0
            for i in group : 
                if i < 2 :
                    a += 1
                if i < 3 :
                    b += 1
                if i < 4 :
                    c += 1
            ax[z].hist(group, bins =100, label =f"Conformation n°{z+1}") #Create an histogram to see the distribution of the RMSD of the sample
            ax[z].axvline(x=2, ymin=0, ymax=1, color="black", linestyle="--")
            ax[z].annotate(a, (1.5, 0.05*len(st.session_state.mols)), fontsize=15)
            ax[z].axvline(x=3, ymin=0, ymax=1, color="black", linestyle="--")
            ax[z].annotate(b-a, (2.5, 0.05*len(st.session_state.mols)), fontsize=15)
            ax[z].axvline(x=4, ymin=0, ymax=1, color="black", linestyle="--")
            ax[z].annotate(c-b, (3.5, 0.05*len(st.session_state.mols)), fontsize=15)
            ax[z].legend(loc='best', shadow=True, markerfirst = False)

    st.pyplot(fig)
    st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
             " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")
    st.session_state.fig3 = fig
    fig.savefig("Histograms_Best_Score.jpeg", dpi=300)
    print("RMSD distribution between all docking solutions", 
          " and the pose with the highest ChemPLP score of all solutions for a given conformation.")
    
def get_barsplot(k) :
    
    """
    -- DESCRIPTION --
    Commenter cette fonction.
    """
        
    # BAR PLOT
    file_in = [x for x in Chem.SDMolSupplier(f"Conformation{k}.sdf")]
    ensemble = {mol.GetProp(st.session_state.molecule_name) for mol in st.session_state.suppl}
    
    index = list(ensemble)
    index.sort()
    
    columns = "Good Poses", "Total Poses", "Ratio"
    
    array = np.zeros(shape=(len(ensemble),3))

    for i, indice in enumerate(index) :
        for mol in file_in :
            if mol.GetProp(st.session_state.molecule_name) == indice : 
                array[i, 0] += 1
        for mol in st.session_state.suppl :
            if mol.GetProp(st.session_state.molecule_name) == indice : 
                array[i, 1] += 1        
        array[i, 2] = array[i, 0]/array[i, 1]
        
    data_frame = pd.DataFrame(array, index=index, columns=columns)
    table = data_frame.sort_values("Ratio", ascending=False)       
    
    sns.set_context('talk')
    sns.set_style('whitegrid')
    g = sns.catplot(x="Ratio", y=list(table.index), data = table, kind="bar", aspect=st.session_state.aspect_plot,
                    height=st.session_state.height_plot, palette = "rocket")
    g.set_ylabels("")
    xlocs, xlabs = plt.xticks()
    
    for i, v in enumerate(list(table["Ratio"])):
        plt.text(float(v)+0.047, float(i)+0.35, f"{round(v*100, 1)}%", horizontalalignment = "center",
                 fontsize=st.session_state.size_ylabels+3)
    
    plt.yticks(fontsize=18)
    plt.axvline(x=0.5, color="black", linestyle="--")
    plt.axvline(x=0.25, color="black", linestyle=":")
    plt.axvline(x=0.75, color="black", linestyle=":")
    g.set_axis_labels("Ratio", "", fontsize = 25)
    g.set_yticklabels(fontsize = st.session_state.size_ylabels)
    g.set_xticklabels(fontsize = st.session_state.size_xlabels)
    g.fig.savefig(f"Barplot_Conformation{k}.jpeg", dpi=300)
    st.session_state.fig4 = g
    
    # BOX PLOT
    file_synt = (mol for name in list(table.index) for mol in file_in if mol.GetProp(st.session_state.molecule_name) == name)
    
    index2 = []
    
    array2 = np.zeros(shape=(len(file_in),2))
    
    for i, mol in enumerate(file_synt):
                index2.append(mol.GetProp(st.session_state.molecule_name))
                array2[i, 0] = mol.GetProp(st.session_state.score)
    
    data_frame2 = pd.DataFrame(array2, index=index2, columns=[st.session_state.score, "Zero"])
    
    file_synt = (mol for name in list(table.index) for mol in file_in if mol.GetProp(st.session_state.molecule_name) == name)
    for i, mol in enumerate(file_synt):
        array2[i, 1] = np.median(data_frame2.loc[mol.GetProp(st.session_state.molecule_name), st.session_state.score])
    
    data_frame3 = pd.DataFrame(array2, index=index2, columns=[st.session_state.score, "Median"])
    
    data_frame2.reset_index(inplace=True)
    data_frame2.rename(columns={'index': 'Name'}, inplace=True)
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    g = sns.catplot(x=st.session_state.score, y='Name', data=data_frame2, kind = "box", palette="rocket",
                    aspect=st.session_state.aspect_plot, height=st.session_state.height_plot)
    g.set_axis_labels(st.session_state.score, "", fontsize = 20)
    g.set_yticklabels(fontsize = st.session_state.size_ylabels)
    g.set_xticklabels(fontsize = st.session_state.size_xlabels)
    g.fig.savefig(f"Box_Plot{k}.jpeg", dpi=300)
    st.session_state.fig5 = g
    
    # DENSITY PLOT
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_context('talk')

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(len(data_frame2.index), rot=-.25, light=.7)
    g = sns.FacetGrid(data_frame2, row="Name", hue="Name", aspect=st.session_state.aspect_density_plot,
                      height=st.session_state.height_density_plot, palette="rocket")

    # Draw the densities in a few steps
    g.map(sns.kdeplot, st.session_state.score,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, st.session_state.score, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, st.session_state.score)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=st.session_state.gap_density_plot)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.savefig(f"Density_Plot{k}.jpeg", dpi=300)
    st.session_state.fig6 = g
    
    
    
    # SECOND BOX PLOT AND DENSITY PLOT

    data_frame3.reset_index(inplace=True)
    data_frame3.rename(columns={'index': 'Name'}, inplace=True)
    data_frame3.sort_values("Median", ascending=False, inplace=True)
    
    
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    g = sns.catplot(x=st.session_state.score, y='Name', data=data_frame3, kind = "box", palette="rocket",
                    aspect=st.session_state.aspect_plot, height=st.session_state.height_plot)
    g.set_axis_labels(st.session_state.score, "")
    g.set_yticklabels(fontsize = st.session_state.size_ylabels)
    g.set_xticklabels(fontsize = st.session_state.size_xlabels)
    g.fig.savefig(f"Box2_Plot{k}.jpeg", dpi=300)
    st.session_state.fig7 = g
    
    
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_context('talk')
    
    pal = sns.cubehelix_palette(len(data_frame3.index), rot=-.25, light=.7)
    g = sns.FacetGrid(data_frame3, row="Name", hue="Name", aspect=st.session_state.aspect_density_plot,
                      height=st.session_state.height_density_plot, palette="rocket")
    g.map(sns.kdeplot, st.session_state.score,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, st.session_state.score, clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    g.map(label, st.session_state.score)
    g.figure.subplots_adjust(hspace=st.session_state.gap_density_plot)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.savefig(f"Density2_Plot{k}.jpeg", dpi=300)
    st.session_state.fig8 = g





###############################################
#                                             #
#            APPLICATION SECTION              #
#                                             #                                                      
###############################################
    
st.header('ConformationTool !')

st.markdown('Welcome to ConformationTool ! This version is compatible with the docking results files from the GOLD software'
            ' using the scoring functions: ChemPLP Score and GoldScore.\n')

#SDF FILE SECTION#
sdf = st.file_uploader("Upload docked ligand coordinates in SDF format:",
                                    type = ["sdf"])
if sdf:
    st.session_state.sdf_file_stock = sdf
    molecule_name = st.text_input("What is the name of the column in your sdf file that contains the names of the molecules"
                      " (and not the names of each poses resulting from the docking simulations)?", 'Compound Name')
    st.session_state.molecule_name = molecule_name

    score = st.selectbox(
         'What is the scoring function used in your sdf file ?',
         ('Gold.PLP.Fitness', 'Gold.Goldscore.Fitness'))
    st.session_state.score = score
else:
    try : 
        del st.session_state.sdf_file_stock
    except :
        pass

#BENCHMARK MOLECULE SECTION#

if 'molref_stock' not in st.session_state:
    st.session_state.molref_stock = None

molref = st.file_uploader("Input your benchmark molecule (mol file in 2D or 3D) here.")

if molref :
    st.session_state.molref_stock = molref
else:
    try : 
        del st.session_state.molref_stock
    except :
        pass

#INDIVIDUALS SECTION#
    
individuals = st.slider('Select the size of your sample. Default size of sample = 200', 0, 500, 200,
                        help='If you want to change this setting during the program, make sure the box below is unchecked!')


###############################################
#--        CHECKBOX "CHECK YOUR SDF"        --#                                                     
###############################################

first_checkbox = st.checkbox('Check your sdf (Attention ! Before closing this app, please, UNCHECK THIS BOX)')
if first_checkbox :
    try : 
        if 'output_name_prefix' not in st.session_state :
            st.session_state.output_name_prefix = "Chemotools_Work_"
        if 'structures_directory' not in st.session_state :
            st.session_state.structures_directory = st.session_state.output_name_prefix + "_structures"
            with open(st.session_state.output_name_prefix + "_sdf_file.sdf", "wb") as f1:
                f1.write(st.session_state.sdf_file_stock.getbuffer())
            with open(st.session_state.output_name_prefix + "_mol_ref.mol", "wb") as f2:
                f2.write(st.session_state.molref_stock.getbuffer())
        if 'sdf_file' not in st.session_state:
            st.session_state.sdf_file = st.session_state.output_name_prefix + "_sdf_file.sdf"
        if 'suppl_brut' not in st.session_state:
            st.session_state.suppl_brut = [x for x in Chem.SDMolSupplier(st.session_state.output_name_prefix + "_sdf_file.sdf")]
        if 'premols' not in st.session_state:
            st.session_state.premols = [GetScaffoldForMol(x) for x in st.session_state.suppl_brut]
        if 'molref' not in st.session_state:
            st.session_state.molref = Chem.MolFromMolFile(st.session_state.output_name_prefix + "_mol_ref.mol")
    except AttributeError:
        st.error("OOPS ! Did you forget to input the SDF file or the benchmark molecule ?")

    mols = []
    suppl = []
    error_molecules = []

    for i, mol in enumerate(st.session_state.premols) :
        try :
            rdMolAlign.CalcRMS(mol, st.session_state.molref)
            mols.append(mol)
            suppl.append(st.session_state.suppl_brut[i])
        except RuntimeError as e :
            error_molecules.append(mol.GetProp("_Name"))
            
    if 'mols' not in st.session_state:
        st.session_state.mols = mols
    if 'suppl' not in st.session_state:
        st.session_state.suppl = suppl
    if 'error_molecules' not in st.session_state:
        st.session_state.error_molecules = error_molecules
    if 'samples' not in st.session_state:
        st.session_state.samples = random.sample(range(len(st.session_state.mols)), individuals)

    show_molecules = st.checkbox('Show molecules name which will be not included in the algorithm')
    if show_molecules:
        if error_molecules == None :
            st.write('All molecules are good !')
        if error_molecules :
            for i in error_molecules : 
                st.write(i)
            st.warning("All theses molecules have not sub-structure which match between the reference and probe mol.\n"
                      "An RMSD can't be calculated with these molecules, they will therefore not be taken into " 
                      "account by the algorithm.\n")

    st.info(f"There are {len(st.session_state.premols)} molecule poses in the sdf file.\n")
    st.info(f"There are {len(st.session_state.mols)} molecule poses which will be used by the algorithm.\n")
    
    try :
        os.remove(st.session_state.output_name_prefix + "_sdf_file.sdf")
        os.remove(st.session_state.output_name_prefix + "_mol_ref.mol")
    except :
        pass
    
    #SECOND BOX
    second_checkbox = st.checkbox('Get the sample heatmap')
    if second_checkbox :
        try:
            st.pyplot(st.session_state.fig1)
        except:
            pass
        
        if 'fig1' not in st.session_state:
            array = np.ones(shape=(len(st.session_state.samples),len(st.session_state.samples)))
            for i, indivduali in enumerate(st.session_state.samples) :
                for j, indivdualj in enumerate(st.session_state.samples) :
                    array[i, j] = rdMolAlign.CalcRMS(st.session_state.mols[indivduali],st.session_state.mols[indivdualj])

            df = pd.DataFrame(
                array, index=list(range(len(st.session_state.samples))), columns=list(range(len(st.session_state.samples))))

            fig, ax = plt.subplots(figsize=(20, 10))
            sns.set_context('talk')
            sns.heatmap(df, fmt='d', ax= ax)
            st.pyplot(fig)
            st.session_state.fig1 = fig
    
    else:
        if 'fig1' in st.session_state:
            del st.session_state.fig1

    RMSD_Target = st.slider('RMSD threshold: Select the maximum RMSD threshold that should constitute a conformation. Default RMSD threshold = 1',
                         0.0, 15.0, 2.0, help='If you want to change this setting during the program, make sure the box below is unchecked!')
    
    loop = st.slider('Number of Loops', 0, 20, 0, help="In the aim to build the sorted heatmap, the sorting process may requires to"
                     " be repeating many times, this is the number of loops, in order to have more resolution. However, especially"
                     " in the case of there is only one predominant conformation adopted by the large majority of the poses, you may"
                     " be invited to reduce the number of loops as the sorting process may remove a lot of individuals of the sample.")
    
###############################################
#--   CHECKBOX "GET THE SORTED HEATMAP"     --#                                                     
############################################### 
    
    third_checkbox = st.checkbox('Get the sorted heatmap')
    if third_checkbox :
        try:
            st.warning(f"ATTENTION !!! {st.session_state.indviduals_deleted} INDIVIDUALS HAVE BEEN DELETED DURING THE TREATMENT")
            st.write("In order to increase the resolution of your sorted heatmap you can play with 2 setting :\n"
                     "- Decrease the RMSD threshold (Do not hesitate!)\n"
                     "- Increase the number of loops")
            
            st.pyplot(st.session_state.fig2)
            with open("Sorted_Heatmap.jpeg", "rb") as file:
                 btn = st.download_button(
                         label="Download PLOT sorted heatmap",
                         data=file,
                         file_name="Sorted_Heatmap.jpeg",
                         mime="image/jpeg")
            st.info(f"There are {st.session_state.numbers_conformation} predominant poses among all poses.\n")
            
            for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                st.write(f"The predominant pose n°{i+1} represents {len(predominant_pose)/len(st.session_state.samples)*100:.1f}" 
                         f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.samples)} poses in total.")
            
            st.write("\nIn order to check that each group is different from each other, a table taking " 
                      "the first individual from each group and calculating the RMSD between each was constructed :\n")
            
            st.dataframe(st.session_state.df1)
            
            st.write("RMSD value between each representative of each conformation.")
            with open("Sample_Best_PLPScore_Poses.sdf", "rb") as file:
                 btn = st.download_button(
                            label="Download SDF Best Score Poses of each Conformation from the SAMPLE",
                            data=file,
                             file_name="Sample_Best_Score_Poses.sdf")
            with open("Best_PLPScore_Poses.sdf", "rb") as file:
                 btn = st.download_button(
                            label="Download SDF Best Score Poses of each Conformation from the filtered INPUT SDF FILE",
                            data=file,
                             file_name="Best_Score_Poses.sdf")
            for i in range(st.session_state.numbers_conformation):
                with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                     btn = st.download_button(
                                label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                data=file,
                                 file_name=f"Sample_Conformation{i+1}.sdf")
            
            st.pyplot(st.session_state.fig3)
            
            st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
             " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")      
            
            with open("Histograms_Best_Score.jpeg", "rb") as file:
                 btn = st.download_button(
                            label="Download PLOT Histograms",
                            data=file,
                             file_name="Histograms_Best_Score.jpeg",
                             mime="image/jpeg")
            
            ###############################################
            #--    BUTTON "PREPARE YOUR SDF FILE"       --#                                                     
            ###############################################

            if st.session_state.numbers_conformation != 1 :
                temp_options = range(1, st.session_state.numbers_conformation + 1)
                st.session_state.temp = st.select_slider("You want a sdf file or a barplot including molecules in the conformation n°",
                                                         options=temp_options)
                st.write(f"The Conformation selected is {st.session_state.temp}")
                
                st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")
                
                settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots)',
                                                help=st.session_state.help_paragraph)
                if settings_checkbox :
                    st.session_state.aspect_plot = st.slider('Configure the aspect ratio of the barplot and boxplots', 0.0, 10.0, 2.5,
                                                             help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    st.session_state.height_plot = st.slider('Configure the height of the barplot and boxplots', 0, 50, 7,
                                                             help="Height (in inches) of each facet.")
                    st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                    st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 25)
                    st.session_state.aspect_density_plot = st.slider('Configure the aspect ratio of the density plots figure', 0, 100, 25,
                                                                     help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    st.session_state.height_density_plot = st.slider('Configure the height of the density plots figure', 0.0, 2.0, 0.4,
                                                                     help="Height (in inches) of each facet.")
                    st.session_state.gap_density_plot = st.slider('Configure the gap between each density plot'
                                                                  ' (It is preferable to configure the height)', -1.0, 1.0, -0.55)
                else :
                    pass
                
                
                if st.button('Prepare your sdf file and build plots'):             
                    best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                    with Chem.SDWriter(f'Conformation{st.session_state.temp}.sdf') as w: #Create a sdf file
                        for j, mol in enumerate(st.session_state.mols) :
                            if rdMolAlign.CalcRMS(best_PLP_poses[int(st.session_state.temp - 1)], mol) < float(st.session_state.RMSD_Target_conformation) :
                                w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                    w.close()
                        

                    with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                         btn = st.download_button(
                                    label="Download your sdf file",
                                    data=file,
                                     file_name=f"Conformation n°{st.session_state.temp}.sdf")
                         
                    
                    with st.spinner('Please wait, barplot is coming...'):              
                        if "fig4" in st.session_state :
                            del st.session_state.fig4
                        if "fig5" in st.session_state :
                            del st.session_state.fig5
                        if "fig6" in st.session_state :
                            del st.session_state.fig6
                        if "fig7" in st.session_state :
                            del st.session_state.fig7
                        if "fig8" in st.session_state :
                            del st.session_state.fig8 
                        
                        get_barsplot(st.session_state.temp)                    
                        st.pyplot(st.session_state.fig4)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                        
                        with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n°{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n°{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")
                        
                    with st.spinner('Please wait, boxplots and density plots are coming...'):
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write("Boxplot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig5)
                            with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT1 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Boxplot1_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write("Density plot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig6)
                            with open(f"Density2_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT1 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Densityplot1_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                    with st.spinner('Please wait, boxplots and density plots are coming...'):
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig7)
                            with open(f"Box2_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT2 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write(f"Density Plot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig8)
                            with open(f"Density2_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT2 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Densityplot2_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                else :
                    try :
                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                         file_name=f"Conformation n°{st.session_state.temp}.sdf")
                        
                        st.pyplot(st.session_state.fig4)
                        
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                        with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n°{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n°{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")
                        
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write("Boxplot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig5)
                            with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT1 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Boxplot1_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write("Density plot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig6)
                            with open(f"Density2_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT1 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Densityplot1_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig7)
                            with open(f"Box2_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT2 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write(f"Density Plot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig8)
                            with open(f"Density2_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT2 Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Densityplot2_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
        
                    except :
                        pass
                
            else :                              
                st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file or a barplot including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)
                st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")
                
                st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                
                settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots)',
                                                help=st.session_state.help_paragraph)
                if settings_checkbox :
                    st.session_state.aspect_plot = st.slider('Configure the aspect ratio of the barplot and boxplots', 0.0, 10.0, 2.5,
                                                             help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    st.session_state.height_plot = st.slider('Configure the height of the barplot and boxplots', 0, 50, 7,
                                                             help="Height (in inches) of each facet.")
                    st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                    st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 25)
                    st.session_state.aspect_density_plot = st.slider('Configure the aspect ratio of the density plots figure', 0, 100, 25,
                                                                     help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    st.session_state.height_density_plot = st.slider('Configure the height of the density plots figure', 0.0, 2.0, 0.4,
                                                                     help="Height (in inches) of each facet.")
                    st.session_state.gap_density_plot = st.slider('Configure the gap between each density plot'
                                                                  ' (It is preferable to configure the height)', -1.0, 1.0, -0.55)
                
                bouton = st.button('I want to have the sdf file of poses in this conformation AND/OR the analysis.')
                if bouton :
                    best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                    for i, best_PLP_pose in enumerate(best_PLP_poses):
                        with Chem.SDWriter('Conformation1.sdf') as w: #Create a sdf file
                            for j, mol in enumerate(st.session_state.mols) :
                                if rdMolAlign.CalcRMS(best_PLP_pose, mol) < float(st.session_state.RMSD_Target_conformation) :
                                    w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                        w.close()
                    
                    with open('Conformation1.sdf', "rb") as file:
                     btn = st.download_button(
                                label="Download your sdf file",
                                data=file,
                                 file_name=f"Unique Conformation.sdf")
                
                    with st.spinner('Please wait, the barplot is coming...'):
                        if "fig4" in st.session_state :
                            del st.session_state.fig4
                        if "fig5" in st.session_state :
                            del st.session_state.fig5
                        if "fig6" in st.session_state :
                            del st.session_state.fig6
                        if "fig7" in st.session_state :
                            del st.session_state.fig7
                        if "fig8" in st.session_state :
                            del st.session_state.fig8 
                        
                        get_barsplot(1)
                        st.pyplot(st.session_state.fig4)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                        
                        with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n°1 ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n°1.jpeg",
                                     mime="image/jpeg")
                        
                    with st.spinner('Please wait, boxplots and density plots are coming...'):
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write("Boxplot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig5)
                            with open(f"Box_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT1 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Boxplot1_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write("Density plot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig6)
                            with open(f"Density2_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT1 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Densityplot1_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                        
                    with st.spinner('Please wait, boxplots and density plots are coming...'):
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig7)
                            with open(f"Box2_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT2 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write(f"Density Plot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig8)
                            with open(f"Density2_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT2 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Densityplot2_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                else :
                    try :
                        with open(f'Conformation1.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                         file_name=f"Conformation n°1.sdf")
                        
                        st.pyplot(st.session_state.fig4)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                        
                        with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n°1 ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n°1.jpeg",
                                     mime="image/jpeg")
                        
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write("Boxplot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig5)
                            with open(f"Box_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT1 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Boxplot1_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write("Density plot built following the order of the above barplot.")
                            st.pyplot(st.session_state.fig6)
                            with open(f"Density2_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT1 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Densityplot1_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                        
                        col1, col2 = st.columns(2)
                        with col1 :
                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig7)
                            with open(f"Box2_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box PLOT2 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                        with col2 :
                            st.write(f"Density Plot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.fig8)
                            with open(f"Density2_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Density PLOT2 Conformation n°1 ",
                                         data=file,
                                         file_name=f"Densityplot2_Conformation n°1.jpeg",
                                         mime="image/jpeg")
                    except :
                        pass
 
            
        except KeyError :
            st.error("The name of the column in your sdf file that contains the names of the molecules doesn't seem to be "
                       f"'{st.session_state.molecule_name}'. Please uncheck the checkbox 'Check your sdf' before replacing the name.")
        except AttributeError as e :
            #st.exception(e)
            pass
        

###############################################
#--        BEGINNING THIRDCHECKBOX          --#                                                     
###############################################

        if 'fig2' not in st.session_state:
            with st.spinner('Please wait, the sorted heatmap is coming...'):
                output_liste = sorted_list_lengroups(get_groups_inside_list(improve_sort(
                    get_filtered_liste(sorted_list_lengroups(gather_groups_RMSD(RMSD_listes(st.session_state.samples)))), loop)))
                finallyliste = get_filtered_liste(output_liste)
                if len(finallyliste) != individuals :
                    st.warning(f"ATTENTION !!! {individuals-len(finallyliste)} INDIVIDUALS HAVE BEEN DELETED DURING THE TREATMENT")
                st.write("In order to increase the resolution of your sorted heatmap you can play with 2 settings :\n"
                     "- Decrease the RMSD threshold (Do not hesitate!)\n"
                     "- Increase the number of loops")
                st.session_state.indviduals_deleted = individuals-len(finallyliste)
                array = np.ones(shape=(len(finallyliste),len(finallyliste)))

                for i, indivduali in enumerate(finallyliste) :
                    for j, indivdualj in enumerate(finallyliste) :
                        array[i, j] = rdMolAlign.CalcRMS(st.session_state.mols[st.session_state.samples[indivduali]],
                                                         st.session_state.mols[st.session_state.samples[indivdualj]])

                data_frame = pd.DataFrame(array, index=finallyliste,
                                          columns=finallyliste)
                fig, ax = plt.subplots(figsize=(20, 10))
                try : 
                    g = sns.heatmap(data_frame, fmt='d', ax= ax, cmap = "rocket")
                    fig = g.get_figure()
                    fig.savefig("Sorted_Heatmap.jpeg", dpi=300)
                    st.pyplot(fig)
                    with open("Sorted_Heatmap.jpeg", "rb") as file:
                         btn = st.download_button(
                                 label="Download PLOT sorted heatmap",
                                 data=file,
                                 file_name="Sorted_Heatmap.jpeg",
                                 mime="image/jpeg")
                    st.session_state.fig2 = fig
                    if 'output_liste' not in st.session_state:
                        st.session_state.output_liste = output_liste
                    if 'sample_predominant_poses' not in st.session_state:
                        st.session_state.sample_predominant_poses = get_predominant_poses(st.session_state.output_liste)
                    if 'sample_indice_best_score' not in st.session_state:
                        st.session_state.sample_indice_best_score = get_sample_indice_best_score(st.session_state.sample_predominant_poses)
                    get_Sample_Best_PLPScore_Poses()
                    if 'best_PLP_poses' not in st.session_state:
                        st.session_state.best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                    if 'df1' not in st.session_state:
                        get_data_frame_best_poses(st.session_state.best_PLP_poses)
                    with open("Sample_Best_PLPScore_Poses.sdf", "rb") as file:
                         btn = st.download_button(
                                    label="Download SDF Best Score Poses of each Conformation from the SAMPLE",
                                    data=file,
                                     file_name="Sample_Best_Score_Poses.sdf")
                    with open("Best_PLPScore_Poses.sdf", "rb") as file:
                         btn = st.download_button(
                                    label="Download SDF Best Score Poses of each Conformation from the filtered INPUT SDF FILE",
                                    data=file,
                                     file_name="Best_Score_Poses.sdf")
                    for i in range(st.session_state.numbers_conformation):
                        with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                             btn = st.download_button(
                                        label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                        data=file,
                                         file_name=f"Sample_Conformation{i+1}.sdf")
                    if 'fig3' not in st.session_state:
                        get_histogramme_sample_bestPLP(st.session_state.best_PLP_poses)
                    with open("Histograms_Best_Score.jpeg", "rb") as file:
                         btn = st.download_button(
                                    label="Download PLOT Histograms",
                                    data=file,
                                     file_name="Histograms_Best_Score.jpeg",
                                     mime="image/jpeg")
                    
                except ValueError :
                    st.error("OOPS ! The number of loops is too high. Please uncheck the sorted heatmap box and then lower it")
                
            if st.session_state.numbers_conformation != 1 :
                temp_options = range(1, st.session_state.numbers_conformation + 1)
                st.session_state.temp = st.select_slider("You want a sdf file or a barplot including molecules in the conformation n°",
                                                         options=temp_options)
                st.write(f"The Conformation selected is {st.session_state.temp}")
                                    
                st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")
                
                if "aspect_plot" not in st.session_state :
                    st.session_state.aspect_plot = 2.5
                if "height_plot" not in st.session_state :
                    st.session_state.height_plot = 7
                if "size_xlabels" not in st.session_state :
                    st.session_state.size_xlabels = 25
                if "size_ylabels" not in st.session_state :
                    st.session_state.size_ylabels = 25
                if "aspect_density_plot" not in st.session_state :
                    st.session_state.aspect_density_plot = 25
                if "height_density_plot" not in st.session_state :
                    st.session_state.height_density_plot = 0.4
                if "gap_density_plot" not in st.session_state :
                    st.session_state.gap_density_plot = -0.55
                
                st.session_state.help_paragraph = ("To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                                 "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30, Aspect ratio density plot = 25,"
                                 " Height density plot = 0.3, Gap = -0.55\n "
                                 "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                                 " Height = 18, Xlabels Size = 25, Ylabels Size = 15, Aspect ratio density plot = 25,"
                                 " Height density plot = 0.5, Gap = -0.45")
                
                settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots)',
                                                help=st.session_state.help_paragraph)
                if settings_checkbox :
                    st.session_state.aspect_plot = st.slider('Configure the aspect ratio of the barplot and boxplots', 0.0, 10.0, 2.5,
                                                             help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    
                    st.session_state.height_plot = st.slider('Configure the height of the barplot and boxplots', 0, 50, 7,
                                                             help="Height (in inches) of each facet.")
                    
                    st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                    st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 25)
                    
                    st.session_state.aspect_density_plot = st.slider('Configure the aspect ratio of the density plots figure', 0, 100, 25,
                                                                     help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    
                    st.session_state.height_density_plot = st.slider('Configure the height of the density plots figure', 0.0, 2.0, 0.4,
                                                                     help="Height (in inches) of each facet.")
                    
                    st.session_state.gap_density_plot = st.slider('Configure the gap between each density plot'
                                                                  ' (It is preferable to configure the height)', -1.0, 1.0, -0.55)
                
                if st.button('Prepare your sdf file and build plots'):             
                    best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                    with Chem.SDWriter(f'Conformation{st.session_state.temp}.sdf') as w: #Create a sdf file
                        for j, mol in enumerate(st.session_state.mols) :
                            if rdMolAlign.CalcRMS(best_PLP_poses[int(st.session_state.temp - 1)], mol) < float(st.session_state.RMSD_Target_conformation) :
                                w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                    w.close()
            else :
                st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file or a barplot including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)
                
                if "aspect_plot" not in st.session_state :
                    st.session_state.aspect_plot = 2.5
                if "height_plot" not in st.session_state :
                    st.session_state.height_plot = 7
                if "size_xlabels" not in st.session_state :
                    st.session_state.size_xlabels = 25
                if "size_ylabels" not in st.session_state :
                    st.session_state.size_ylabels = 25
                if "aspect_density_plot" not in st.session_state :
                    st.session_state.aspect_density_plot = 25
                if "height_density_plot" not in st.session_state :
                    st.session_state.height_density_plot = 0.4
                if "gap_density_plot" not in st.session_state :
                    st.session_state.gap_density_plot = -0.55
                
                st.session_state.help_paragraph = ("To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                                 "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30, Aspect ratio density plot = 25,"
                                 " Height density plot = 0.3, Gap = -0.55\n "
                                 "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                                 " Height = 18, Xlabels Size = 25, Ylabels Size = 15, Aspect ratio density plot = 25,"
                                 " Height density plot = 0.5, Gap = -0.45"
                                 "\nAfter configuration, settings are saved even if you change the sdf file.")
                
                settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots)',
                                                help=st.session_state.help_paragraph)
                
                if settings_checkbox :
                    st.session_state.aspect_plot = st.slider('Configure the aspect ratio of the barplot and boxplots', 0.0, 10.0, 2.5,
                                                             help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    
                    st.session_state.height_plot = st.slider('Configure the height of the barplot and boxplots', 0, 50, 7,
                                                             help="Height (in inches) of each facet.")
                    
                    st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                    st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 25)
                    
                    st.session_state.aspect_density_plot = st.slider('Configure the aspect ratio of the density plots figure', 0, 100, 25,
                                                                     help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")
                    
                    st.session_state.height_density_plot = st.slider('Configure the height of the density plots figure', 0.0, 2.0, 0.4,
                                                                     help="Height (in inches) of each facet.")
                    
                    st.session_state.gap_density_plot = st.slider('Configure the gap between each density plot'
                                                                  ' (It is preferable to configure the height)', -1.0, 1.0, -0.55)
                
                bouton = st.button('I want the sdf file of poses in this conformation AND/OR the analysis.')
                if bouton :
                    best_PLP_poses = [GetScaffoldForMol(x) for x in Chem.SDMolSupplier("Best_PLPScore_Poses.sdf")]
                    for i, best_PLP_pose in enumerate(best_PLP_poses):
                        with Chem.SDWriter('Unique_Conformation.sdf') as w: #Create a sdf file
                            for j, mol in enumerate(st.session_state.mols) :
                                if rdMolAlign.CalcRMS(best_PLP_pose, mol) < float(st.session_state.RMSD_Target_conformation) :
                                    w.write(st.session_state.suppl[j]) #Then write this molecule pose in the new sdf file.
                        w.close()

                     
                

    else :
        if 'numbers_conformation' in st.session_state:
            del st.session_state.numbers_conformation
            try :
                os.remove('Sorted_Heatmap.jpeg')
                os.remove('Histograms_Best_Score.jpeg')
                os.remove('Sample_Best_PLPScore_Poses.sdf')
                os.remove('Best_PLPScore_Poses.sdf')
                for i in range(st.session_state.numbers_conformation):
                    os.remove(f'Sample_Conformation{i+1}.sdf')
                    try :
                        os.remove(f'Conformation{i+1}.sdf')
                        os.remove(f'Barplot_Conformation{i+1}.jpeg')
                        os.remove(f'Box_Plot{i+1}.jpeg')
                        os.remove(f'Density_Plot{i+1}.jpeg')
                        os.remove(f'Box2_Plot{i+1}.jpeg')
                        os.remove(f'Density2_Plot{i+1}.jpeg')
                    except :
                        pass
            except :
                pass        
            
        if 'indviduals_deleted' in st.session_state:
            del st.session_state.indviduals_deleted
        if 'fig2' in st.session_state:
            del st.session_state.fig2
        if 'predominant_poses' in st.session_state:
            del st.session_state.predominant_poses
        if 'df1' in st.session_state:
            del st.session_state.df1
        if 'fig3' in st.session_state:
            del st.session_state.fig3
        if 'output_liste' in st.session_state:
            del st.session_state.output_liste 
        if 'sample_predominant_poses' in st.session_state:
            del st.session_state.sample_predominant_poses
        if 'sample_indice_best_score' in st.session_state:
            del st.session_state.sample_indice_best_score
        if 'best_PLP_poses' in st.session_state:
            del st.session_state.best_PLP_poses
        if 'fig4' in st.session_state:
            del st.session_state.fig4
        if 'fig5' in st.session_state:
            del st.session_state.fig5
        if 'fig6' in st.session_state:
            del st.session_state.fig6



else :
    try :
        if 'output_name_prefix' in st.session_state:
            del st.session_state.output_name_prefix        
        if 'structures_directory' in st.session_state:
            del st.session_state.structures_directory        
        if 'sdf_file' in st.session_state:
            del st.session_state.sdf_file
        if 'suppl_brut' in st.session_state:
            del st.session_state.suppl_brut 
        if 'premols' in st.session_state:
            del st.session_state.premols
        if 'molref' in st.session_state:
            del st.session_state.molref
        if 'mols' in st.session_state:
            del st.session_state.mols
        if 'suppl' in st.session_state:
            del st.session_state.suppl
        if 'error_molecules' in st.session_state:
            del st.session_state.error_molecules
        if 'samples' in st.session_state:
            del st.session_state.samples
        if 'fig4' in st.session_state:
            del st.session_state.fig4
        if 'numbers_conformation' in st.session_state:
            os.remove('Sorted_Heatmap.jpeg')
            os.remove('Histograms_Best_Score.jpeg')
            os.remove('Sample_Best_PLPScore_Poses.sdf')
            os.remove('Best_PLPScore_Poses.sdf')
            for i in range(st.session_state.numbers_conformation):
                os.remove(f'Sample_Conformation{i+1}.sdf')
                try :
                    os.remove(f'Conformation{i+1}.sdf')
                    os.remove(f'Barplot_Conformation{i+1}.jpeg')
                    os.remove(f'Box_Plot{i+1}.jpeg')
                    os.remove(f'Density_Plot{i+1}.jpeg')
                    os.remove(f'Box2_Plot{i+1}.jpeg')
                    os.remove(f'Density2_Plot{i+1}.jpeg')
                except :
                    pass
                    
            del st.session_state.numbers_conformation
        if 'indviduals_deleted' in st.session_state:
            del st.session_state.indviduals_deleted
        if 'fig2' in st.session_state:
            del st.session_state.fig2
        if 'predominant_poses' in st.session_state:
            del st.session_state.predominant_poses
        if 'df1' in st.session_state:
            del st.session_state.df1
        if 'fig3' in st.session_state:
            del st.session_state.fig3
        if 'output_liste' in st.session_state:
            del st.session_state.output_liste 
        if 'sample_predominant_poses' in st.session_state:
            del st.session_state.sample_predominant_poses
        if 'sample_indice_best_score' in st.session_state:
            del st.session_state.sample_indice_best_score
        if 'best_PLP_poses' in st.session_state:
            del st.session_state.best_PLP_poses
        if 'fig4' in st.session_state:
            del st.session_state.fig4
        if 'fig5' in st.session_state:
            del st.session_state.fig5
        if 'fig6' in st.session_state:
            del st.session_state.fig6
        if 'fig7' in st.session_state:
            del st.session_state.fig7
        if 'fig8' in st.session_state:
            del st.session_state.fig8
    except :
        pass

