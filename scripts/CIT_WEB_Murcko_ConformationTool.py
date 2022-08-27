"""
#####################################################
##                                                 ##
##       -- CIT CONFORMATIONTOOL MURCKO --         ##
##                                                 ##
#####################################################
"""

import os, copy, random, py3Dmol
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw 
from rdkit.Chem.rdMolAlign import CalcRMS
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from ChemoInfoTool import ConformationTool

def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock

def render_mol(xyz):
    xyzview = py3Dmol.view()#(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=1000)

###############################################
#                                             #
#            APPLICATION SECTION              #
#                                             #                                                      
###############################################

def main():
    
    st.header('Murcko ConformationTool !')
    
    st.markdown('Welcome to **Murcko ConformationTool !** Here, the processing and analysis of the docking simulation'
                ' results is done through the **Murcko Scaffold** [DOI: 10.1021/jm9602928].')

    #SDF FILE SECTION#
    sdf = st.file_uploader("Upload the coordinates of the docked ligand in SDF format:",
                                        type = ["sdf"],  key = 'CIT_WEB_Murcko_ConformationTool')
    if sdf:
        molecule_name = st.text_input("What is the name of the column in your sdf file that contains the names of the molecules"
                          " (and not the names of each poses resulting from the docking simulations)?", 'Compound Name',
                                      key = 'CIT_WEB_Murcko_ConformationTool')
        st.session_state.molecule_name_murcko = molecule_name
        score = st.text_input(
                     'What is the scoring function used in your sdf file ?',
                     'Gold.PLP.Fitness', key = 'CIT_WEB_Murcko_ConformationTool')
        st.session_state.score_murcko = score
        if 'sdf_file_stock_murcko' not in st.session_state :
            st.session_state.sdf_file_stock_murcko = sdf
            with open("sdf_file.sdf", "wb") as f:
                f.write(st.session_state.sdf_file_stock_murcko.getbuffer())
            mols = [x for x in Chem.SDMolSupplier("sdf_file.sdf")]
            st.session_state.mol1_murcko = mols[0]
            del mols
            os.remove("sdf_file.sdf")
    else:
        if "sdf_file_stock_murcko" in st.session_state :
            del st.session_state.sdf_file_stock_murcko
        if "mol1_murcko" in st.session_state :
            del st.session_state.mol1_murcko


    #BENCHMARK MOLECULE SECTION#
    col1, col2 = st.columns(2)

    with col1 :
        if 'mol1_murcko' in st.session_state :
            code_smiles = Chem.MolToSmiles(st.session_state.mol1_murcko)
            smiles = st.text_input('Upload the benchmark molecule SMILES code',
                                                      code_smiles, key = 'CIT_WEB_Murcko_ConformationTool')
        else :
            smiles = st.text_input('Upload the benchmark molecule SMILES code',
                                                      'OC(COC1=C2C=CC=CC2=CC=C1)CNC(C)C', key = 'CIT_WEB_Murcko_ConformationTool')

    with col2 :
        molref = st.file_uploader("OR Upload your benchmark molecule (mol file in 2D or 3D) here.", key = 'CIT_WEB_Murcko_ConformationTool')

    if molref :
        with open("mol_ref.mol", "wb") as molref_file:
            molref_file.write(molref.getbuffer())
        mol_file = Chem.MolFromMolFile("mol_ref.mol")
        mol_smiles = Chem.MolToSmiles(mol_file)
        mol_benchmark = Chem.MolFromSmiles(mol_smiles)
        AllChem.EmbedMolecule(mol_benchmark)
        st.session_state.molref_stock_murcko = Chem.MolToSmiles(mol_benchmark)
        os.remove("mol_ref.mol")
    else :
        st.session_state.molref_stock_murcko = smiles

    blk=makeblock(st.session_state.molref_stock_murcko)
    render_mol(blk)
    st.write("Chemical structure of the benchmark molecule")

    #PDB PROTEIN SECTION#

    pdb = st.file_uploader("Upload a pdb file for viewing purposes. (FACULTATIVE)",
                                        type = ["pdb"], key = 'CIT_WEB_Murcko_ConformationTool')
    if pdb :
        st.session_state.pdb_murcko = pdb
        with open("pdb_file.pdb", "wb") as pdb_file:
            pdb_file.write(pdb.getbuffer())


    ###############################################
    #--        CHECKBOX "CHECK YOUR SDF"        --#                                                     
    ###############################################

    first_checkbox = st.checkbox(
        'Check your sdf', key = 'CIT_WEB_Murcko_ConformationTool')
    if first_checkbox :
        st.session_state.murcko_delete_activated = True
        try : 
            if 'ConformationClass' not in st.session_state :
                with open("sdf_file.sdf", "wb") as sdf:
                    sdf.write(st.session_state.sdf_file_stock_murcko.getbuffer())
                st.session_state.sdf_file_murcko = "sdf_file.sdf"        
                st.session_state.ConformationClass = ConformationTool(st.session_state.sdf_file_murcko,
                                                                      st.session_state.score_murcko,
                                                                      conformation_tool = "Murcko",
                                                                      streamlit = True)
            if 'error_mols' not in st.session_state:
                st.session_state.ConformationClass.check_sdf_file(st.session_state.molref_stock_murcko)

        except AttributeError:
            st.error("OOPS ! Did you forget to input the SDF file ?")
            #pass

        mol_benchmark = Chem.MolFromSmiles(st.session_state.molref_stock_murcko)
        scaff_mol_bench = GetScaffoldForMol(mol_benchmark)
        AllChem.Compute2DCoords(mol_benchmark)
        AllChem.Compute2DCoords(scaff_mol_bench)
        col1, col2 = st.columns(2)
        with col1 :
            im = Draw.MolToImage(mol_benchmark)
            st.image(im, caption='Chemical structure of the benchmark molecule')
        with col2 :
            im = Draw.MolToImage(scaff_mol_bench)
            st.image(im, caption='Chemical structure of the scaffold of the benchmark molecule')    

        show_molecules = st.checkbox('Show molecules name which will be not included in the algorithm', key = 'CIT_WEB_Murcko_ConformationTool')
        if show_molecules:
            if st.session_state.error_mols == None :
                st.write('All molecules are good !')
            if st.session_state.error_mols :
                unique_name = []
                unique_mol = []
                for mol in st.session_state.error_mols:
                    if mol.GetProp(st.session_state.molecule_name_murcko) in unique_name:
                        continue
                    else:
                        unique_name.append(mol.GetProp(st.session_state.molecule_name_murcko))
                        unique_mol.append(mol)
                col1, col2, col3 = st.columns(3)
                with col1 :
                    st.write("Molecule Name")
                with col2 :
                    st.write("Chemical structure of the molecule")
                with col3 :
                    st.write("Chemical structure of the scaffold of the molecule")           
                for mol in unique_mol :
                    AllChem.Compute2DCoords(mol)
                    col1, col2, col3 = st.columns(3)
                    with col1 :
                        st.write(mol.GetProp(st.session_state.molecule_name_murcko))
                    with col2 :
                        st.image(Draw.MolToImage(mol))
                    with col3 :
                        st.image(Draw.MolToImage(GetScaffoldForMol(mol)))
                st.warning("All theses molecules have not sub-structure (scaffold) which match between "
                           "the reference and probe mol.\n An RMSD can't be calculated with these molecules,"
                           " they will therefore not be taken into account by the algorithm.\n")

        try :
            st.info(f"There are {st.session_state.mols_brut} molecule poses in the sdf file.\n")
        except AttributeError :
            st.error('OOPS ! Have you submitted your sdf file? ?')
        st.info(f"There are {st.session_state.mols} molecule poses which will be used by the algorithm.\n")

        try :
            os.remove("sdf_file.sdf")
            os.remove("mol_ref.mol")
        except :
            pass

        if st.session_state.mols > 200 :
            size_sample = 200
        else :
            size_sample = st.session_state.mols
        individuals = st.slider('Select the size of your sample. Default size of sample = 200', 0, 500, size_sample,
                                help='If you want to change this setting during the program, make sure the box below is unchecked!')

        #SECOND BOX
        second_checkbox = st.checkbox('Get the sample heatmap')
        if second_checkbox :
            try:
                st.pyplot(st.session_state.heatmap)
            except:
                pass

            if 'heatmap' not in st.session_state :
                st.session_state.ConformationClass.get_heatmap_sample(individuals)

        else:
            if 'heatmap' in st.session_state:
                del st.session_state.heatmap

        RMSD_Target = st.slider('RMSD threshold: Select the maximum RMSD threshold that should constitute a predominant binding mode.'
                                ' Default RMSD threshold = 2 A',
                                 0.0, 15.0, 2.0,
                                help='If you want to change this setting during the program, make sure the box below'
                                ' is unchecked!', key = 'CIT_WEB_Murcko_ConformationTool')

        Proportion = st.slider('Minimum size of the sample defining a predominant binding mode. Default proportion = 0.05',
                             0.0, 1.0, 0.15,
                               help=('This setting define the minimum proportion (value between 0 and 1) of individuals'
                                     ' in a group within the sample to consider that group large enough to be'
                                     ' representative of a full predominant binding mode.'), key = 'CIT_WEB_Murcko_ConformationTool')

    ###############################################
    #--   CHECKBOX "GET THE SORTED HEATMAP"     --#                                                     
    ############################################### 

        tab1, tab2 = st.tabs(["Sorting Process", "Cluster Hierarchy"])
        
        with tab1:
            third_checkbox = st.checkbox('Get the sorted heatmap', key = 'CIT_WEB_Murcko_ConformationTool')
            if third_checkbox :
                if 'sorted_heatmap' not in st.session_state:
                    with st.spinner('Please wait, the sorted heatmap is coming...'):
                        st.session_state.ConformationClass.get_sorted_heatmap(
                            individuals,
                            RMSD_Target,
                            loop = 1,
                            p = Proportion)
                else:
                    if 'indviduals_deleted' in st.session_state:
                        st.warning(
                            f"Attention. The sorting process discarded {st.session_state.indviduals_deleted} individuals")

                    st.pyplot(st.session_state.sorted_heatmap)
                    with open("Sorted_Heatmap.jpeg", "rb") as file:
                         btn = st.download_button(
                                  label="Download PLOT sorted heatmap",
                                  data=file,
                                  file_name="Sorted_Heatmap.jpeg",
                                  mime="image/jpeg",
                                  key = 'CIT_WEB_Murcko_ConformationTool')
                    
                    st.info(f"There is (are) {st.session_state.n_conformations} predominant pose(s) among all poses.\n")

                    if 'pdb' in st.session_state :
                        style = st.selectbox('Style',['cartoon','cross','stick','sphere','line','clicksphere'])
                        #bcolor = st.color_picker('Pick A Color', '#ffffff')
                        pdb_file = Chem.MolFromPDBFile('pdb_file.pdb')
                        best_mols = [x for x in Chem.SDMolSupplier('Best_PLPScore_Poses.sdf')]
                        for i, mol in enumerate(best_mols) :
                            merged = Chem.CombineMols(pdb_file, mol)
                            Chem.MolToPDBFile(merged, f'Predominant Binding Mode n°{i+1}.pdb')
                            xyz_pdb = open(f'Predominant Binding Mode n°{i+1}.pdb', 'r', encoding='utf-8')
                            pdb = xyz_pdb.read().strip()
                            xyz_pdb.close()
                            xyzview = py3Dmol.view(width=700,height=500)
                            xyzview.addModel(pdb, 'pdb')
                            xyzview.setStyle({style:{'color':'spectrum'}})
                            #xyzview.setBackgroundColor(bcolor)#('0xeeeeee')
                            xyzview.setStyle({'resn':'UNL'},{'stick':{}})
                            xyzview.zoomTo({'resn':'UNL'})
                            showmol(xyzview, height = 500,width=1000)
                            os.remove(f'Conformation n°{i+1}.pdb')
                            st.write(f'Predominant Binding Mode n°{i+1}')
                            with open(f"Sample_Predominant_Binding_Mode{i+1}.sdf", "rb") as file:
                                 btn = st.download_button(
                                              label=f"Download all the poses of the predominant binding mode n°{i+1} from the SAMPLE",
                                              data=file,
                                              file_name=f"Sample_Predominant_Binding_Mode{i+1}.sdf",
                                              key = 'CIT_WEB_Murcko_ConformationTool')

                    with open("Best_PLPScore_Poses.sdf", "rb") as file:
                         btn = st.download_button(
                                     label="Download the SDF file including each of the representatives of a predominant binding mode",
                                     data=file,
                                     file_name="Best_Score_Poses.sdf",
                                     key = 'CIT_WEB_Murcko_ConformationTool')

                    for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                        st.write(
                            f"The predominant binding mode n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                            f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

                    st.write("\nIn order to check that each group is different from each other, a table taking " 
                              "the first individual from each group and calculating the RMSD between each was constructed :\n")

                    st.dataframe(st.session_state.df1)

                    st.write("RMSD value between each representative of each predominant binding mode.")

                    for i, fig in enumerate(st.session_state.histplot) :
                        st.pyplot(fig)
                        with open(f"Histograms_Best_Score n°{i+1}.jpeg", "rb") as file:
                             btn = st.download_button(
                                         label=f"Download PLOT Histogram n°{i+1}",
                                         data=file,
                                         file_name=f"Histograms_Best_Score n°{i+1}.jpeg",
                                         mime="image/jpeg",
                                         key = 'CIT_WEB_Murcko_ConformationTool')

                    st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each"
                    " predominant binding mode and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")      

                    ###############################################
                    #--    BUTTON "PREPARE YOUR SDF FILE"       --#                                                     
                    ###############################################

                if st.session_state.n_conformations != 1 :
                    temp_options = range(1, st.session_state.n_conformations + 1)
                    
                    if "temp" not in st.session_state :
                        st.session_state.temp = 1
                    
                    st.session_state.temp = st.select_slider("You want a sdf file or plots including molecules in the predominant binding mode n°",
                                                             options=temp_options, value = st.session_state.temp)
                    st.write(f"The Predominant Binding Mode selected is {st.session_state.temp}")

                    st.session_state.RMSD_threshold_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_threshold_conformation}")

                    st.session_state.help_paragraph = (
                        "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                        "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                        "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                        " Height = 18, Xlabels Size = 25, Ylabels Size = 15")
                    
                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph, key = 'CIT_WEB_Murcko_ConformationTool')
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                             'Configure the aspect ratio of the plots', 0.0, 10.0, 1.75,
                             help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.",
                             key = 'CIT_WEB_Murcko_ConformationTool')

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.", key = 'CIT_WEB_Murcko_ConformationTool')

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25,
                                                                  key = 'CIT_WEB_Murcko_ConformationTool')
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15,
                                                                  key = 'CIT_WEB_Murcko_ConformationTool')


                    if st.button('Prepare your sdf file and build plots'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_threshold_conformation
                            )
                        with open(f'Predominant Binding Mode n°{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                         label="Download your sdf file",
                                         data=file,
                                         file_name=f"Predominant Binding Mode n°{st.session_state.temp}.sdf",
                                         key = 'CIT_WEB_Murcko_ConformationTool')

                        if "barplot" in st.session_state :
                            del st.session_state.barplot
                        if "box_plot" in st.session_state :
                            del st.session_state.box_plot
                        if "scatterplot" in st.session_state :
                            del st.session_state.scatterplot
                        
                        if "aspect_plot" not in st.session_state :
                            st.session_state.aspect_plot = 1.75
                        if "height_plot" not in st.session_state :
                            st.session_state.height_plot = 18
                        if "size_xlabels" not in st.session_state :
                            st.session_state.size_xlabels = 25
                        if "size_ylabels" not in st.session_state :
                            st.session_state.size_ylabels = 15

                        with st.spinner('Plots are coming. Please wait...'):
                            try :
                                st.session_state.ConformationClass.get_plots(
                                    st.session_state.temp,
                                    st.session_state.molecule_name_murcko,
                                    aspect_plot = st.session_state.aspect_plot,
                                    height_plot = st.session_state.height_plot,
                                    size_xlabels = st.session_state.size_xlabels,
                                    size_ylabels = st.session_state.size_ylabels)

                                st.pyplot(st.session_state.barplot)
                                st.write("Ratio of each compounds between the number of poses in the predominant binding mode selected and"
                                         " the number of total poses.")

                                with open(f"Barplot{st.session_state.temp}.jpeg", "rb") as file:
                                     btn = st.download_button(
                                              label=f"Download Bar PLOT Predominant Binding Mode n°{st.session_state.temp} ",
                                              data=file,
                                              file_name=f"Barplot n°{st.session_state.temp}.jpeg",
                                              mime="image/jpeg",
                                              key = 'CIT_WEB_Murcko_ConformationTool')

                                st.write(f"Boxplot built following the descending order of the {st.session_state.score_murcko}.")
                                st.pyplot(st.session_state.box_plot)
                                with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                     btn = st.download_button(
                                              label=f"Download Box PLOT Predominant Binding Mode n°{st.session_state.temp} ",
                                              data=file,
                                              file_name=f"Boxplot n°{st.session_state.temp}.jpeg",
                                              mime="image/jpeg",
                                              key = 'CIT_WEB_Murcko_ConformationTool')
                                
                                st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score_murcko}.")
                                st.pyplot(st.session_state.scatterplot)
                                with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                     btn = st.download_button(
                                              label=f"Download Scatter Plot Predominant Binding Mode n°{st.session_state.temp} ",
                                              data=file,
                                              file_name=f"Scatter_Plot n°{st.session_state.temp}.jpeg",
                                              mime="image/jpeg",
                                              key = 'CIT_WEB_Murcko_ConformationTool')
                            except KeyError :
                                st.error("The name of the column in your sdf file that contains the names of the molecules doesn't seem to be "
                                     f"'{st.session_state.molecule_name_murcko}'. Please correct it.") 
                    else :
                        if os.path.exists(f'Predominant Binding Mode{st.session_state.temp}.sdf') == True :
                            with open(f'Predominant Binding Mode{st.session_state.temp}.sdf', "rb") as file:
                                 btn = st.download_button(
                                             label="Download your sdf file",
                                             data=file,
                                             file_name=f"Predominant Binding Mode n°{st.session_state.temp}.sdf",
                                             key = 'CIT_WEB_Murcko_ConformationTool')

                        if os.path.exists(f'Barplot{st.session_state.temp}.jpeg') == True :
                            st.pyplot(st.session_state.barplot)

                            st.write("Ratio of each compounds between the number of poses in the predominant binding mode selected and"
                                     " the number of total poses.")
                            with open(f"Barplot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Bar PLOT Predominant Binding Mode n°{st.session_state.temp} ",
                                          data=file,
                                          file_name=f"Barplot n°{st.session_state.temp}.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')

                        if os.path.exists(f'Barplot{st.session_state.temp}.jpeg') == True :
                            st.pyplot(st.session_state.box_plot)
                            
                            st.write(f"Boxplot built following the descending order of the {st.session_state.score_murcko}.")
                            with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Box Plot Predominant Binding Mode n°{st.session_state.temp} ",
                                          data=file,
                                          file_name=f"Boxplot n°{st.session_state.temp}.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')
                        
                        if os.path.exists(f'Scatter_Plot{st.session_state.temp}.jpeg') == True :    
                            st.pyplot(st.session_state.scatterplot)
                            
                            st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score_murcko}.")
                            with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Scatter Plot Predominant Binding Mode n°{st.session_state.temp} ",
                                          data=file,
                                          file_name=f"Scatter_Plot n°{st.session_state.temp}.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')

                else :                              
                    st.session_state.RMSD_threshold_conformation = st.slider(
                        'You want a sdf file and/or plot analysis including molecules in the unique predominant predominant binding mode'
                        ' with all poses under a RMSD =', 0.0, 15.0, 2.0)

                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_threshold_conformation}")

                    st.info('There is only one predominant binding mode. Do you want to have the sdf file of poses in this conformation and/or see plot analysis ? ')

                    st.session_state.help_paragraph = (
                        "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                        "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                        "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                        " Height = 18, Xlabels Size = 25, Ylabels Size = 15")
                    
                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plts', 0.0, 10.0, 1.75,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.",
                            key = 'CIT_WEB_Murcko_ConformationTool')

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.",
                            key = 'CIT_WEB_Murcko_ConformationTool')

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25,
                                                                  key = 'CIT_WEB_Murcko_ConformationTool')
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15,
                                                                  key = 'CIT_WEB_Murcko_ConformationTool')


                    bouton = st.button('Prepare your sdf file of poses in this predominant binding mode.')
                    if bouton :
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_threshold_conformation)

                        with open('Predominant Binding Mode n°1.sdf', "rb") as file:
                         btn = st.download_button(
                                     label="Download your sdf file",
                                     data=file,
                                     file_name=f"Unique Predominant Binding Mode.sdf",
                                     key = 'CIT_WEB_Murcko_ConformationTool')

                        if "barplot" in st.session_state :
                            del st.session_state.barplot
                        if "box_plot" in st.session_state :
                            del st.session_state.box_plot
                        if "scatterplot" in st.session_state :
                            del st.session_state.scatterplot 
                        
                        if "aspect_plot" not in st.session_state :
                            st.session_state.aspect_plot = 1.75
                        if "height_plot" not in st.session_state :
                            st.session_state.height_plot = 18
                        if "size_xlabels" not in st.session_state :
                            st.session_state.size_xlabels = 25
                        if "size_ylabels" not in st.session_state :
                            st.session_state.size_ylabels = 15
                        
                        try :
                            st.session_state.ConformationClass.get_plots(
                                1,
                                st.session_state.molecule_name_murcko,
                                aspect_plot = st.session_state.aspect_plot,
                                height_plot = st.session_state.height_plot,
                                size_xlabels = st.session_state.size_xlabels,
                                size_ylabels = st.session_state.size_ylabels)

                            st.pyplot(st.session_state.barplot)
                            st.write("Ratio of each compounds between the number of poses in the predominant binding mode selected and the"
                                     " number of total poses.")

                            with open(f"Barplot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Bar PLOT Predominant Binding Mode n°1 ",
                                          data=file,
                                          file_name=f"Barplot n°1.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')

                            st.write(f"Boxplot built following the descending order of the {st.session_state.score_murcko}.")
                            st.pyplot(st.session_state.box_plot)
                            with open(f"Box_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Box Plot Predominant Binding Mode n°1 ",
                                          data=file,
                                          file_name=f"Boxplot n°1.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')

                            st.write(f"Scatterplot built with the ratio as a function of the {st.session_state.score_murcko}.")
                            st.pyplot(st.session_state.scatterplot)
                            with open(f"Scatter_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Scatter Plot Predominant Binding Mode n°1 ",
                                          data=file,
                                          file_name=f"Scatter_Plot n°1.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')
                        except KeyError :
                            st.error("The name of the column in your sdf file that contains the names of the molecules doesn't seem to be "
                                     f"'{st.session_state.molecule_name_murcko}'. Please correct it.")         
                    else :
                        if os.path.exists(f'Predominant Binding Mode n°1.sdf') == True :
                            with open(f'Predominant Binding Mode n°1.sdf', "rb") as file:
                                 btn = st.download_button(
                                             label="Download your sdf file",
                                             data=file,
                                             file_name=f"Predominant Binding Mode n°1.sdf",
                                             key = 'CIT_WEB_Murcko_ConformationTool')
                        
                        if os.path.exists(f'Barplot.jpeg') == True :
                            st.pyplot(st.session_state.barplot)
                            st.write("Ratio of each compounds between the number of poses in the predominant binding mode selected and"
                                     " the number of total poses.")

                            with open(f"Barplot.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Bar PLOT Predominant Binding Mode n°1 ",
                                          data=file,
                                          file_name=f"Barplot n°1.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')
                        
                        if os.path.exists('Box_Plot1.jpeg') == True :
                            st.write(f"Boxplot built following the descending order of the {st.session_state.score_murcko}.")
                            st.pyplot(st.session_state.box_plot)
                            with open(f"Box_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Box Plot Predominant Binding Mode n°1 ",
                                          data=file,
                                          file_name=f"Boxplot n°1.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')
                        
                        if os.path.exists('Scatter_Plot1.jpeg') == True :
                            st.write(f"Scatterplot built with the ratio as a function of the {st.session_state.score_murcko}.")
                            st.pyplot(st.session_state.scatterplot)
                            with open(f"Scatter_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                          label=f"Download Scatter Plot Predominant Binding Mode n°1 ",
                                          data=file,
                                          file_name=f"Scatter_Plot n°1.jpeg",
                                          mime="image/jpeg",
                                          key = 'CIT_WEB_Murcko_ConformationTool')

            else :
                if 'sorted_heatmap' in st.session_state:
                    del st.session_state.sorted_heatmap
                    if 'temp' in st.session_state:
                        del st.session_state.temp
                    if 'RMSD_threshold_conformation' in st.session_state:
                        del st.session_state.RMSD_threshold_conformation
                    if os.path.exists(f'Sorted_Heatmap.jpeg') == True :
                        os.remove('Sorted_Heatmap.jpeg')
                    if os.path.exists(f'Cluster_Hierarchy_Heatmap.jpeg') == True :
                        os.remove('Cluster_Hierarchy_Heatmap.jpeg')
                    if os.path.exists(f'Cluster_Hierarchy_Heatmap.jpeg') == True :
                        os.remove('Cluster_Hierarchy_Heatmap.jpeg')
                    if os.path.exists(f'Histograms_Best_Score n°{i+1}.jpeg') == True :
                        os.remove(f'Histograms_Best_Score n°{i+1}.jpeg')
                    if os.path.exists(f'Sample_Best_PLPScore_Poses.sdf') == True :
                        os.remove('Sample_Best_PLPScore_Poses.sdf')
                    if os.path.exists(f'Best_PLPScore_Poses.sdf') == True :
                        os.remove('Best_PLPScore_Poses.sdf')

                    if "n_conformations" in st.session_state :
                        for i in range(st.session_state.n_conformations):
                            if os.path.exists(f'Sample_Predominant_Binding_Mode{i+1}.sdf') == True :
                                os.remove(f'Sample_Predominant_Binding_Mode{i+1}.sdf')
                            if os.path.exists(f'Predominant Binding Mode n°{i+1}.sdf') == True :
                                os.remove(f'Predominant Binding Mode n°{i+1}.sdf')
                            if os.path.exists(f'Barplot{i+1}.jpeg') == True :
                                os.remove(f'Barplot{i+1}.jpeg')
                            if os.path.exists(f'Box_Plot{i+1}.jpeg') == True :
                                os.remove(f'Box_Plot{i+1}.jpeg')
                            if os.path.exists(f'Scatter_Plot{i+1}.jpeg') == True :
                                os.remove(f'Scatter_Plot{i+1}.jpeg')

        with tab2:
            col1, col2 = st.columns(2)
            with col1 :                
                metric = st.selectbox('Which metric do you want to use to compute the pairwise distances between observations in n-dimensional space ?',
                                      ('canberra', 'euclidean', 'hamming', 'braycurtis','chebyshev', 'cityblock', 'correlation',
                                       'cosine', 'dice', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                       'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                       'sokalsneath', 'sqeuclidean', 'yule'), key = 'CIT_WEB_Murcko_ConformationTool',
                                       help = "Before changing this setting, make sure the following checkbox is unchecked")
                st.session_state.metric = metric
                
            with col2 :
                method = st.selectbox('Which method do you want to use to compute the distance  between two clusters ?',
                                        ('average', 'single', 'complete', 'weighted', 'centroid', 'median', 'ward'),
                                       key = 'CIT_WEB_Murcko_ConformationTool',
                                       help = "Before changing this setting, make sure the following checkbox is unchecked")
                st.session_state.method = method
            third_checkbox_2 = st.checkbox('Get the cluster hierarchy heatmap', key = 'CIT_WEB_Murcko_ConformationTool')
            if third_checkbox_2 :
                if "cluster_hierarchy_heatmap" not in st.session_state :
                    st.session_state.ConformationClass.get_cluster_heatmap(individuals,
                                                                           method = st.session_state.method,
                                                                           metric = st.session_state.metric)
                    
                    if "n_clusters" not in st.session_state :
                        st.session_state.n_clusters = 2
                    
                    n_clusters = st.slider('In how many clusters do you want to cut the tree (dendrogram) ?', 2, 30, st.session_state.n_clusters,
                                            key = 'CIT_WEB_Murcko_ConformationTool')
                    st.session_state.n_clusters = n_clusters
                    st.session_state.n_clusters_selected = n_clusters
                    
                    if st.button('Get analysis', key = 'CIT_WEB_Murcko_ConformationTool'):
                        st.session_state.ConformationClass.analyze_cluster_heatmap(st.session_state.n_clusters, p = Proportion)
                else :
                    if "cluster_hierarchy_heatmap" in st.session_state :
                        st.pyplot(st.session_state.cluster_hierarchy_heatmap)
                        with open("Cluster_Hierarchy_Heatmap.jpeg", "rb") as file:
                            btn = st.download_button(
                                     label="Download PLOT : Cluster Hierarchy Heatmap",
                                     data=file,
                                     file_name="Cluster_Hierarchy_Heatmap.jpeg",
                                     mime="image/jpeg",
                                     key = 'CIT_WEB_Murcko_ConformationTool')                       
                            
                        n_clusters = st.slider('In how many clusters do you want to cut the tree (dendrogram) ?', 2, 30,
                                               st.session_state.n_clusters, key = 'CIT_WEB_Murcko_ConformationTool')
                        st.session_state.n_clusters = n_clusters
                        
                        if st.button('Get analysis', key = 'CIT_WEB_Murcko_ConformationTool'):
                            st.session_state.n_clusters_selected = st.session_state.n_clusters
                            st.session_state.ConformationClass.analyze_cluster_heatmap(st.session_state.n_clusters, p = Proportion)
                            if st.button('Continue', key = 'CIT_WEB_Murcko_ConformationTool'):
                                "Rerun"
                            
                        else :
                            if "n_conformations" in st.session_state and st.session_state.n_clusters_selected == st.session_state.n_clusters:
                                if 'pdb' in st.session_state :
                                    style = st.selectbox('Style',['cartoon','cross','stick','sphere','line','clicksphere'])
                                    #bcolor = st.color_picker('Pick A Color', '#ffffff')
                                    pdb_file = Chem.MolFromPDBFile('pdb_file.pdb')
                                    best_mols = [x for x in Chem.SDMolSupplier('Best_PLPScore_Poses.sdf')]
                                    for i, mol in enumerate(best_mols) :
                                        merged = Chem.CombineMols(pdb_file, mol)
                                        Chem.MolToPDBFile(merged, f'Conformation n°{i+1}.pdb')
                                        xyz_pdb = open(f'Conformation n°{i+1}.pdb', 'r', encoding='utf-8')
                                        pdb = xyz_pdb.read().strip()
                                        xyz_pdb.close()
                                        xyzview = py3Dmol.view(width=700,height=500)
                                        xyzview.addModel(pdb, 'pdb')
                                        xyzview.setStyle({style:{'color':'spectrum'}})
                                        #xyzview.setBackgroundColor(bcolor)#('0xeeeeee')
                                        xyzview.setStyle({'resn':'UNL'},{'stick':{}})
                                        xyzview.zoomTo({'resn':'UNL'})
                                        showmol(xyzview, height = 500,width=1000)
                                        os.remove(f'Conformation n°{i+1}.pdb')
                                        st.write(f'Predominant Binding Mode n°{i+1}')
                                        with open(f"Sample_Predominant_Binding_Mode{i+1}.sdf", "rb") as file:
                                             btn = st.download_button(
                                                 label=f"Download all the poses of the predominant binding mode n°{i+1} from the SAMPLE",
                                                 data=file,
                                                 file_name=f"Sample_Predominant_Binding_Mode{i+1}.sdf",
                                                 key = 'CIT_WEB_Murcko_ConformationTool')

                                with open("Best_PLPScore_Poses.sdf", "rb") as file:
                                     btn = st.download_button(
                                                label="Download the SDF file including each of the representatives of a predominant binding mode",
                                                data=file,
                                                file_name="Best_Score_Poses.sdf",
                                                key = 'CIT_WEB_Murcko_ConformationTool')

                                st.info(f"There is (are) {st.session_state.n_conformations} predominant pose(s) among all poses.\n")
                                for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                                    st.write(
                                        f"The predominant binding mode n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                                        f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

                                st.write("\nIn order to check that each group is different from each other, a table taking " 
                                          "the **with the best score** from each group and calculating the RMSD between each was constructed :\n")

                                st.dataframe(st.session_state.df1)

                                st.write("RMSD value between each representative of each predominant binding mode.")

                                for i, fig in enumerate(st.session_state.histplot) :
                                    st.pyplot(fig)
                                    with open(f"Histograms_Best_Score n°{i+1}.jpeg", "rb") as file:
                                         btn = st.download_button(
                                                     label=f"Download PLOT Histogram n°{i+1}",
                                                     data=file,
                                                     file_name=f"Histograms_Best_Score n°{i+1}.jpeg",
                                                     mime="image/jpeg",
                                                     key = 'CIT_WEB_Murcko_ConformationTool')

                                st.write("Density of the number of poses as a function of the RMSD calculated between the representative of each conformation"
                                 " and all poses of all molecules in the docking solutions of the filtered incoming sdf file.")      
                                
                                temp_options = range(1, st.session_state.n_conformations + 1)
                                if "temp" not in st.session_state :
                                    st.session_state.temp = 1
                                st.session_state.temp = st.select_slider("You want a sdf file and/or a analysis plots including molecules in the conformation n°",
                                                                         options=temp_options,
                                                                         value = st.session_state.temp,
                                                                         key = 'CIT_WEB_Murcko_ConformationTool')
                                
                                st.write(f"The Predominant Binding Mode selected is {st.session_state.temp}")

                                st.session_state.RMSD_threshold_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0,
                                                                                      key = 'CIT_WEB_Murcko_ConformationTool')
                                st.write(f"The RMSD Target selected is {st.session_state.RMSD_threshold_conformation}")
                           
                                st.session_state.help_paragraph = (
                                    "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                                    "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                                    "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                                    " Height = 18, Xlabels Size = 25, Ylabels Size = 15")
                                
                                settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                                help=st.session_state.help_paragraph)
                                if settings_checkbox :
                                    st.session_state.aspect_plot = st.slider(
                                        'Configure the aspect ratio of the plts', 0.0, 10.0, 1.75,
                                        help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                                    st.session_state.height_plot = st.slider(
                                        'Configure the height of the plots', 0, 50, 18,
                                        help="Height (in inches) of each facet.")

                                    st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                                    st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15)
                                
                                if st.button('Prepare your sdf file', key = 'CIT_WEB_Murcko_ConformationTool'):
                                    st.session_state.ConformationClass.get_sdf_conformations(
                                        st.session_state.temp,
                                        st.session_state.RMSD_threshold_conformation)
                                    
                                    with open(f'Predominant Binding Mode n°{st.session_state.temp}.sdf', "rb") as file:
                                         btn = st.download_button(
                                                    label="Download your sdf file",
                                                    data=file,
                                                     file_name=f"Predominant Binding Mode n°{st.session_state.temp}.sdf")

                                    if "barplot" in st.session_state :
                                        del st.session_state.barplot
                                    if "box_plot" in st.session_state :
                                        del st.session_state.box_plot
                                    if "scatterplot" in st.session_state :
                                        del st.session_state.scatterplot
                                    
                                    if "aspect_plot" not in st.session_state :
                                        st.session_state.aspect_plot = 1.75
                                    if "height_plot" not in st.session_state :
                                        st.session_state.height_plot = 18
                                    if "size_xlabels" not in st.session_state :
                                        st.session_state.size_xlabels = 25
                                    if "size_ylabels" not in st.session_state :
                                        st.session_state.size_ylabels = 15

                                    with st.spinner('Plots are coming. Please wait...'):
                                        try :
                                            st.session_state.ConformationClass.get_plots(
                                                st.session_state.temp,
                                                st.session_state.molecule_name_murcko,
                                                aspect_plot = st.session_state.aspect_plot,
                                                height_plot = st.session_state.height_plot,
                                                size_xlabels = st.session_state.size_xlabels,
                                                size_ylabels = st.session_state.size_ylabels)

                                            st.pyplot(st.session_state.barplot)
                                            st.write("Ratio of each compounds between the number of poses in the predominant binding mode"
                                                     " selected and the number of total poses.")

                                            with open(f"Barplot{st.session_state.temp}.jpeg", "rb") as file:
                                                 btn = st.download_button(
                                                         label=f"Download Bar PLOT Predominant Binding Mode n°{st.session_state.temp} ",
                                                         data=file,
                                                         file_name=f"Barplot n°{st.session_state.temp}.jpeg",
                                                         mime="image/jpeg")

                                            st.write(f"Boxplot built following the descending order of the {st.session_state.score_murcko}.")
                                            st.pyplot(st.session_state.box_plot)
                                            with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                                 btn = st.download_button(
                                                         label=f"Download Box PLOT Predominant Binding Mode n°{st.session_state.temp} ",
                                                         data=file,
                                                         file_name=f"Boxplot n°{st.session_state.temp}.jpeg",
                                                         mime="image/jpeg")
                                            st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score_murcko}.")
                                            st.pyplot(st.session_state.scatterplot)
                                            with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                                 btn = st.download_button(
                                                         label=f"Download Scatter Plot Predominant Binding Mode n°{st.session_state.temp} ",
                                                         data=file,
                                                         file_name=f"Scatter_Plot n°{st.session_state.temp}.jpeg",
                                                         mime="image/jpeg")
                                        except KeyError :
                                            st.error("The name of the column in your sdf file that contains the names of the molecules doesn't seem to be "
                                                 f"'{st.session_state.molecule_name_murcko}'. Please correct it.") 
                                else :
                                    if os.path.exists(f'Predominant Binding Mode{st.session_state.temp}.sdf') == True :
                                        with open(f'Predominant Binding Mode{st.session_state.temp}.sdf', "rb") as file:
                                             btn = st.download_button(
                                                        label="Download your sdf file",
                                                        data=file,
                                                         file_name=f"Predominant Binding Mode n°{st.session_state.temp}.sdf")

                                    if os.path.exists(f'Barplot{st.session_state.temp}.jpeg') == True :
                                        st.pyplot(st.session_state.barplot)

                                        st.write("Ratio of each compounds between the number of poses in the predominant binding mode selected"
                                                 " and the number of total poses.")
                                        with open(f"Barplot{st.session_state.temp}.jpeg", "rb") as file:
                                             btn = st.download_button(
                                                     label=f"Download Bar PLOT Predominant Binding Mode n°{st.session_state.temp} ",
                                                     data=file,
                                                     file_name=f"Barplot n°{st.session_state.temp}.jpeg",
                                                     mime="image/jpeg")

                                    if os.path.exists(f'Barplot{st.session_state.temp}.jpeg') == True :
                                        st.pyplot(st.session_state.box_plot)
                                        
                                        st.write(f"Boxplot built following the descending order of the {st.session_state.score_murcko}.")
                                        with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                             btn = st.download_button(
                                                     label=f"Download Box Plot Predominant Binding Mode n°{st.session_state.temp} ",
                                                     data=file,
                                                     file_name=f"Boxplot n°{st.session_state.temp}.jpeg",
                                                     mime="image/jpeg")
                                    
                                    if os.path.exists(f'Scatter_Plot{st.session_state.temp}.jpeg') == True :    
                                        st.pyplot(st.session_state.scatterplot)
                                        
                                        st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score_murcko}.")
                                        with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                             btn = st.download_button(
                                                     label=f"Download Scatter Plot Predominant Binding Mode n°{st.session_state.temp} ",
                                                     data=file,
                                                     file_name=f"Scatter_Plot n°{st.session_state.temp}.jpeg",
                                                     mime="image/jpeg")

                                    
            else:
                if "cluster_hierarchy_heatmap" in st.session_state :
                    del st.session_state.cluster_hierarchy_heatmap
                    if "n_conformations" in st.session_state :
                        del st.session_state.n_conformations
                    if 'temp' in st.session_state:
                        del st.session_state.temp
                    if 'RMSD_threshold_conformation' in st.session_state:
                        del st.session_state.RMSD_threshold_conformation
                    if os.path.exists(f'Sorted_Heatmap.jpeg') == True :
                        os.remove('Sorted_Heatmap.jpeg')
                    if os.path.exists(f'Cluster_Hierarchy_Heatmap.jpeg') == True :
                        os.remove('Cluster_Hierarchy_Heatmap.jpeg')
                    if os.path.exists(f'Histograms_Best_Score n°{i+1}.jpeg') == True :
                        os.remove(f'Histograms_Best_Score n°{i+1}.jpeg')
                    if os.path.exists(f'Sample_Best_PLPScore_Poses.sdf') == True :
                        os.remove('Sample_Best_PLPScore_Poses.sdf')
                    if os.path.exists(f'Best_PLPScore_Poses.sdf') == True :
                        os.remove('Best_PLPScore_Poses.sdf')

                if "n_clusters" in st.session_state :
                    for i in range(st.session_state.n_clusters):
                        if os.path.exists(f'Sample_Predominant_Binding_Mode{i+1}.sdf') == True :
                            os.remove(f'Sample_Predominant_Binding_Mode{i+1}.sdf')
                        if os.path.exists(f'Predominant Binding Mode n°{i+1}.sdf') == True :
                            os.remove(f'Predominant Binding Mode n°{i+1}.sdf')
                        if os.path.exists(f'Barplot{i+1}.jpeg') == True :
                            os.remove(f'Barplot{i+1}.jpeg')
                        if os.path.exists(f'Box_Plot{i+1}.jpeg') == True :
                            os.remove(f'Box_Plot{i+1}.jpeg')
                        if os.path.exists(f'Scatter_Plot{i+1}.jpeg') == True :
                            os.remove(f'Scatter_Plot{i+1}.jpeg')
                
    else :
        if 'murcko_delete_activated' in st.session_state:
            if os.path.exists(f'Sorted_Heatmap.jpeg') == True :
                os.remove('Sorted_Heatmap.jpeg')
            if os.path.exists(f'Cluster_Hierarchy_Heatmap.jpeg') == True :
                os.remove('Cluster_Hierarchy_Heatmap.jpeg')
            if os.path.exists(f'Sample_Best_PLPScore_Poses.sdf') == True :
                os.remove('Sample_Best_PLPScore_Poses.sdf')
            if os.path.exists(f'Best_PLPScore_Poses.sdf') == True :
                os.remove('Best_PLPScore_Poses.sdf')

            if "n_conformations" in st.session_state :
                for i in range(st.session_state.n_conformations):
                    if os.path.exists(f'Sample_Predominant_Binding_Mode{i+1}.sdf') == True :
                        os.remove(f'Sample_Predominant_Binding_Mode{i+1}.sdf')
                    if os.path.exists(f'Histograms_Best_Score n°{i+1}.jpeg') == True :
                        os.remove(f'Histograms_Best_Score n°{i+1}.jpeg')
                    if os.path.exists(f'Predominant Binding Mode n°{i+1}.sdf') == True :
                        os.remove(f'Predominant Binding Mode n°{i+1}.sdf')
                    if os.path.exists(f'Barplot{i+1}.jpeg') == True :
                        os.remove(f'Barplot{i+1}.jpeg')
                    if os.path.exists(f'Box_Plot{i+1}.jpeg') == True :
                        os.remove(f'Box_Plot{i+1}.jpeg')
                    if os.path.exists(f'Scatter_Plot{i+1}.jpeg') == True :
                        os.remove(f'Scatter_Plot{i+1}.jpeg')
                        
            if "n_clusters" in st.session_state :
                for i in range(st.session_state.n_clusters):
                    if os.path.exists(f'Sample_Predominant_Binding_Mode{i+1}.sdf') == True :
                        os.remove(f'Sample_Predominant_Binding_Mode{i+1}.sdf')
                    if os.path.exists(f'Histograms_Best_Score n°{i+1}.jpeg') == True :
                        os.remove(f'Histograms_Best_Score n°{i+1}.jpeg')
                    if os.path.exists(f'Predominant Binding Mode n°{i+1}.sdf') == True :
                        os.remove(f'Predominant Binding Mode n°{i+1}.sdf')
                    if os.path.exists(f'Barplot{i+1}.jpeg') == True :
                        os.remove(f'Barplot{i+1}.jpeg')
                    if os.path.exists(f'Box_Plot{i+1}.jpeg') == True :
                        os.remove(f'Box_Plot{i+1}.jpeg')
                    if os.path.exists(f'Scatter_Plot{i+1}.jpeg') == True :
                        os.remove(f'Scatter_Plot{i+1}.jpeg')

            for key in st.session_state.keys():
                del st.session_state[key]
