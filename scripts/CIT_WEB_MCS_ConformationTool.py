"""
#####################################################
##                                                 ##
##         -- CIT CONFORMATIONTOOL V1.1 --         ##
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
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdMolAlign import CalcRMS
from ChemoInfoTool import ConformationTool

###############################################
#                                             #
#            APPLICATION SECTION              #
#                                             #                                                      
###############################################

def render_mol(xyz):
    xyzview = py3Dmol.view()#(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=1000)

def main():

    st.header('MCS ConformationTool !')

    st.markdown('Welcome to **MCS ConformationTool** ! Here, the processing and analysis of the docking simulation'
                ' results is done through the **Maximum Common Substructure** (MCS).')

    #SDF FILE SECTION#
    sdf = st.file_uploader("Upload the coordinates of the docked ligand in SDF format:",
                                        type = ["sdf"], key = 'CIT_WEB_MCS_ConformationTool')
    if sdf:
        molecule_name = st.text_input("What is the name of the column in your sdf file that contains the names of the molecules"
                          " (and not the names of each poses resulting from the docking simulations)?", 'Compound Name',
                                       key = 'CIT_WEB_MCS_ConformationTool')
        st.session_state.molecule_name = molecule_name
        score = st.text_input(
             'What is the scoring function used in your sdf file ?',
             'Gold.PLP.Fitness', key = 'CIT_WEB_MCS_ConformationTool')
        st.session_state.score = score
        if 'sdf_file_stock' not in st.session_state :
            st.session_state.sdf_file_stock = sdf
            with open("sdf_file.sdf", "wb") as f:
                f.write(st.session_state.sdf_file_stock.getbuffer())
            mols = [x for x in Chem.SDMolSupplier("sdf_file.sdf")]
            st.session_state.mol1 = mols[0]
            del mols
            os.remove("sdf_file.sdf")
    else:
        try : 
            del st.session_state.sdf_file_stock
            del st.session_state.mol1
        except :
            pass

    #PDB PROTEIN SECTION#

    pdb = st.file_uploader("Upload a pdb file for viewing purposes. (FACULTATIVE)",
                                        type = ["pdb"],  key = 'CIT_WEB_MCS_ConformationTool')
    if pdb :
        st.session_state.pdb = pdb
        with open("pdb_file.pdb", "wb") as pdb_file:
            pdb_file.write(pdb.getbuffer())

    #PERCENTAGE PARAMETER SECTION#

    help_ringMatchesRingOnly = ('If "no", depending on the molecules used, it is possible that the RMSD cannot be'
                                ' calculated afterwards. "Yes" may solve this problem in return for having a less'
                                ' relevant RMSD.')

    ringMatchesRingOnly = st.radio(
         "Should the Maximum Common Substructure take into account uniquely complete rings ?",
         ('No', 'Yes'), help=help_ringMatchesRingOnly)

    if  ringMatchesRingOnly == "Yes":
        st.session_state.ringMatchesRingOnly = True
    else:
        st.session_state.ringMatchesRingOnly = False

    help_percentage = ('In case you consider that the maximum common substructure is not enough to be effective,'
                       ' you can lower this parameter. This will result in removing from your sdf those molecules'
                       ' that share THE LEAST common substructure with the other molecules.')

    st.session_state.percentage = st.slider('Selects the percentage of molecules (not poses) for which a maximum'
                                            ' common substructure will be determined.', 0, 100, 100,
                                            help=help_percentage,  key = 'CIT_WEB_MCS_ConformationTool')

    ###############################################
    #--         CHECKBOX "GET MCS SDF"          --#                                                     
    ###############################################

    first_checkbox = st.checkbox(
        'Get the Maximum Common Substructure between molecules of your SDF file (Attention ! Before closing this app,'
        ' please, UNCHECK THIS BOX)',  key = 'CIT_WEB_MCS_ConformationTool')

    if first_checkbox :
        st.session_state.MCS_delete_activated = True
        with st.spinner('Please wait, the maximum common substructure is coming...'):
            try : 
                if 'ConformationClass' not in st.session_state :
                    with open("sdf_file.sdf", "wb") as sdf:
                        sdf.write(st.session_state.sdf_file_stock.getbuffer())
                    st.session_state.sdf_file = "sdf_file.sdf"        
                    st.session_state.ConformationClass = ConformationTool(st.session_state.sdf_file,
                                                                          st.session_state.score,
                                                                          conformation_tool = "MCS",
                                                                          streamlit = True)
                if 'error_mols' not in st.session_state:
                    st.session_state.ConformationClass.get_MCS_SDF(percentage = st.session_state.percentage,
                                                                   ringMatchesRingOnly=st.session_state.ringMatchesRingOnly)

            except AttributeError:
                st.error("OOPS ! Did you forget to upload your SDF file ?")
                #pass

            render_mol(Chem.MolToMolBlock(st.session_state.fusion))
            st.write("Maximum common substructure between the different molecules of your SDF file")
            #st.image(st.session_state.fusion_im, caption='Maximum common substructure between the different molecules'
            #         ' of your SDF file')   

        show_molecules = st.checkbox('Show molecules which will be not included in the algorithm')
        if show_molecules:
            if st.session_state.error_mols == None :
                st.write('All molecules are good !')
            if st.session_state.error_mols :
                unique_name = []
                unique_mol = []
                for mol in st.session_state.error_mols:
                    if mol.GetProp(st.session_state.molecule_name) in unique_name:
                        continue
                    else:
                        unique_name.append(mol.GetProp(st.session_state.molecule_name))
                        unique_mol.append(mol)
                col1, col2 = st.columns(2)
                with col1 :
                    st.write("Molecule Name")
                with col2 :
                    st.write("Chemical structure of the molecule")          
                for mol in unique_mol :
                    AllChem.Compute2DCoords(mol)
                    col1, col2 = st.columns(2)
                    with col1 :
                        st.write(mol.GetProp(st.session_state.molecule_name))
                    with col2 :
                        st.image(Draw.MolToImage(mol))
                st.warning("These molecules are the ones which share the least common substructure with"
                           " the other molecules in your SDF file.")

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
                                help='If you want to change this setting during the program, make sure the box below is unchecked!',
                                key = 'CIT_WEB_MCS_ConformationTool')

        #SECOND BOX
        second_checkbox = st.checkbox('Get the sample heatmap',  key = 'CIT_WEB_MCS_ConformationTool')
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

        RMSD_Target = st.slider('RMSD threshold: Select the maximum RMSD threshold that should constitute a conformation.'
                                ' Default RMSD threshold = 2 A',
                                 0.0, 15.0, 2.0,
                                help='If you want to change this setting during the program, make sure the box below'
                                ' is unchecked!',  key = 'CIT_WEB_MCS_ConformationTool')

        Proportion = st.slider('Minimum size of the sample defining a conformation. Default proportion = 0.05',
                             0.0, 1.0, 0.05,
                               help=('This setting define the minimum proportion (value between 0 and 1) of individuals'
                                     ' in a group within the sample to consider that group large enough to be'
                                     ' representative of a full conformation.'),  key = 'CIT_WEB_MCS_ConformationTool')

    ###############################################
    #--   CHECKBOX "GET THE SORTED HEATMAP"     --#                                                     
    ############################################### 

        third_checkbox = st.checkbox('Get the sorted heatmap',  key = 'CIT_WEB_MCS_ConformationTool')
        if third_checkbox :
            try:
                st.warning(
                    f"Attention. The sorting process discarded {st.session_state.indviduals_deleted} individuals")

                st.pyplot(st.session_state.sorted_heatmap)
                with open("Sorted_Heatmap.jpeg", "rb") as file:
                     btn = st.download_button(
                             label="Download PLOT sorted heatmap",
                             data=file,
                             file_name="Sorted_Heatmap.jpeg",
                             mime="image/jpeg")
                st.info(f"There is (are) {st.session_state.n_conformations} predominant pose(s) among all poses.\n")

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
                        st.write(f'Conformation n°{i+1}')
                        with open(f"Sample_Conformation{i+1}.sdf", "rb") as file:
                             btn = st.download_button(
                                        label=f"Download all the poses of the conformation n°{i+1} from the SAMPLE",
                                        data=file,
                                         file_name=f"Sample_Conformation{i+1}.sdf")

                with open("Best_PLPScore_Poses.sdf", "rb") as file:
                     btn = st.download_button(
                                label="Download the SDF file including each of the representatives of a conformation",
                                data=file,
                                file_name="Best_Score_Poses.sdf")

                for i, predominant_pose in enumerate(st.session_state.predominant_poses) :
                    st.write(
                        f"The predominant conformation n°{i+1} represents {len(predominant_pose)/len(st.session_state.sample)*100:.1f}" 
                        f"% of the sample, i.e. {len(predominant_pose)} on {len(st.session_state.sample)} poses in total.")

                st.write("\nIn order to check that each group is different from each other, a table taking " 
                          "the first individual from each group and calculating the RMSD between each was constructed :\n")

                st.dataframe(st.session_state.df1)

                st.write("RMSD value between each representative of each conformation.")

                st.pyplot(st.session_state.histplot)

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

                if st.session_state.n_conformations != 1 :
                    temp_options = range(1, st.session_state.n_conformations + 1)
                    st.session_state.temp = st.select_slider("You want a sdf file or plots including molecules in the conformation n°",
                                                             options=temp_options, value = st.session_state.temp)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plots', 0.0, 10.0, 1.75,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15)

                    else :
                        pass


                    if st.button('Prepare your sdf file and build plots'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation
                            )

                        with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                             btn = st.download_button(
                                        label="Download your sdf file",
                                        data=file,
                                         file_name=f"Conformation n°{st.session_state.temp}.sdf")

                        if "barplot" in st.session_state :
                            del st.session_state.barplot
                        if "box_plot" in st.session_state :
                            del st.session_state.box_plot
                        if "scatterplot" in st.session_state :
                            del st.session_state.scatterplot 

                        st.session_state.ConformationClass.get_plots(
                            st.session_state.temp,
                            st.session_state.molecule_name,
                            aspect_plot = st.session_state.aspect_plot,
                            height_plot = st.session_state.height_plot,
                            size_xlabels = st.session_state.size_xlabels,
                            size_ylabels = st.session_state.size_ylabels)

                        st.pyplot(st.session_state.barplot)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")

                        with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n°{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n°{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")

                        st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                        st.pyplot(st.session_state.box_plot)
                        with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Box PLOT Conformation n°{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Boxplot2_Conformation n°{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")
                        st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score}.")
                        st.pyplot(st.session_state.scatterplot)
                        with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Scatter Plot Conformation n°{st.session_state.temp} ",
                                     data=file,
                                     file_name=f"Scatter_Plot n°{st.session_state.temp}.jpeg",
                                     mime="image/jpeg")
                    else :
                        try :
                            with open(f'Conformation{st.session_state.temp}.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                             file_name=f"Conformation n°{st.session_state.temp}.sdf")

                            st.pyplot(st.session_state.barplot)

                            st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")
                            with open(f"Barplot_Conformation{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Bar PLOT Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Barplot_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.box_plot)
                            with open(f"Box_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box Plot Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")
                            st.write(f"Scatter plot built with the ratio as function of the {st.session_state.score}.")
                            st.pyplot(st.session_state.scatterplot)
                            with open(f"Scatter_Plot{st.session_state.temp}.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Scatter Plot Conformation n°{st.session_state.temp} ",
                                         data=file,
                                         file_name=f"Scatter_Plot n°{st.session_state.temp}.jpeg",
                                         mime="image/jpeg")

                        except :
                            pass

                else :                              
                    st.session_state.RMSD_Target_conformation = st.slider(
                        'You want a sdf file and/or plot analysis including molecules in the unique predominant conformation with all poses under a RMSD =',
                        0.0, 15.0, 2.0)

                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')

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


                    bouton = st.button('I want to have the sdf file of poses in this conformation AND/OR the plots.')
                    if bouton :
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_Target_conformation)

                        with open('Conformation1.sdf', "rb") as file:
                         btn = st.download_button(
                                    label="Download your sdf file",
                                    data=file,
                                     file_name=f"Unique Conformation.sdf")

                        if "barplot" in st.session_state :
                            del st.session_state.barplot
                        if "box_plot" in st.session_state :
                            del st.session_state.box_plot
                        if "scatterplot" in st.session_state :
                            del st.session_state.scatterplot 


                        st.session_state.ConformationClass.get_plots(
                            1,
                            st.session_state.molecule_name,
                            aspect_plot = st.session_state.aspect_plot,
                            height_plot = st.session_state.height_plot,
                            size_xlabels = st.session_state.size_xlabels,
                            size_ylabels = st.session_state.size_ylabels)

                        st.pyplot(st.session_state.barplot)
                        st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")

                        with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Bar PLOT Conformation n°1 ",
                                     data=file,
                                     file_name=f"Barplot_Conformation n°1.jpeg",
                                     mime="image/jpeg")

                        st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                        st.pyplot(st.session_state.box_plot)
                        with open(f"Box_Plot1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Box Plot Conformation n°1 ",
                                     data=file,
                                     file_name=f"Boxplot2_Conformation n°1.jpeg",
                                     mime="image/jpeg")

                        st.write(f"Scatterplot built with the ratio as a function of the {st.session_state.score}.")
                        st.pyplot(st.session_state.scatterplot)
                        with open(f"Scatter_Plot1.jpeg", "rb") as file:
                             btn = st.download_button(
                                     label=f"Download Scatter Plot Conformation n°1 ",
                                     data=file,
                                     file_name=f"Scatter_Plot n°1.jpeg",
                                     mime="image/jpeg")
                    else :
                        try :
                            with open(f'Conformation1.sdf', "rb") as file:
                                 btn = st.download_button(
                                            label="Download your sdf file",
                                            data=file,
                                             file_name=f"Conformation n°1.sdf")

                            st.pyplot(st.session_state.barplot)
                            st.write("Ratio of each compounds between the number of poses in the conformation selected and the number of total poses.")

                            with open(f"Barplot_Conformation1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Bar PLOT Conformation n°1 ",
                                         data=file,
                                         file_name=f"Barplot_Conformation n°1.jpeg",
                                         mime="image/jpeg")

                            st.write(f"Boxplot built following the descending order of the {st.session_state.score}.")
                            st.pyplot(st.session_state.box_plot)
                            with open(f"Box_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Box Plot Conformation n°1 ",
                                         data=file,
                                         file_name=f"Boxplot2_Conformation n°1.jpeg",
                                         mime="image/jpeg")

                            st.write(f"Scatterplot built with the ratio as a function of the {st.session_state.score}.")
                            st.pyplot(st.session_state.scatterplot)
                            with open(f"Scatter_Plot1.jpeg", "rb") as file:
                                 btn = st.download_button(
                                         label=f"Download Scatter Plot Conformation n°1 ",
                                         data=file,
                                         file_name=f"Scatter_Plot n°1.jpeg",
                                         mime="image/jpeg")
                        except :
                            pass


            except KeyError :
                st.error("The name of the column in your sdf file that contains the names of the molecules doesn't seem to be "
                           f"'{st.session_state.molecule_name}'. Please correct it.")
            except AttributeError as e :
                #st.exception(e)
                pass


    ###############################################
    #--        BEGINNING THIRDCHECKBOX          --#                                                     
    ###############################################

            if 'sorted_heatmap' not in st.session_state:
                with st.spinner('Please wait, the sorted heatmap is coming...'):
                    st.session_state.ConformationClass.get_sorted_heatmap(
                        individuals,
                        RMSD_Target,
                        loop = 1,
                        p = Proportion)

                if st.session_state.n_conformations != 1:
                    temp_options = range(1, st.session_state.n_conformations + 1)
                    st.session_state.temp = st.select_slider("You want a sdf file and/or a anlysis plots including molecules in the conformation n°",
                                                             options=temp_options)
                    st.write(f"The Conformation selected is {st.session_state.temp}")

                    st.session_state.RMSD_Target_conformation = st.slider('... With all poses under a RMSD =', 0.0, 15.0, 2.0)
                    st.write(f"The RMSD Target selected is {st.session_state.RMSD_Target_conformation}")

                    if "aspect_plot" not in st.session_state :
                        st.session_state.aspect_plot = 1.75
                    if "height_plot" not in st.session_state :
                        st.session_state.height_plot = 18
                    if "size_xlabels" not in st.session_state :
                        st.session_state.size_xlabels = 25
                    if "size_ylabels" not in st.session_state :
                        st.session_state.size_ylabels = 15

                    st.session_state.help_paragraph = (
                        "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                        "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                        "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                        " Height = 18, Xlabels Size = 25, Ylabels Size = 15")

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plots', 0.0, 10.0, 1.75,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 18,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 15)

                    if st.button('Prepare your sdf file and build plots'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            st.session_state.temp,
                            st.session_state.RMSD_Target_conformation
                            )
                else :
                    st.info('There is only one predominant conformation. Do you want to have the sdf file of poses in this conformation OR see analysis of this conformation ? ')
                    st.session_state.RMSD_Target_conformation = st.slider('You want a sdf file and/or a analysis plots including molecules in the unique predominant conformation with all poses under a RMSD =', 0.0, 15.0, 2.0)

                    if "aspect_plot" not in st.session_state :
                        st.session_state.aspect_plot = 1.75
                    if "height_plot" not in st.session_state :
                        st.session_state.height_plot = 18
                    if "size_xlabels" not in st.session_state :
                        st.session_state.size_xlabels = 25
                    if "size_ylabels" not in st.session_state :
                        st.session_state.size_ylabels = 15

                    st.session_state.help_paragraph = (
                        "To give an idea, if your number of molecules (not number of poses) = 15 :\n"
                        "- Aspect ratio = 3, Height = 5, Xlabels Size = 25, Ylabels Size = 30\n "
                        "\nif your number of molecules (not number of poses) = 75 :\n - Aspect ratio = 1.75,"
                        " Height = 18, Xlabels Size = 25, Ylabels Size = 15")

                    settings_checkbox = st.checkbox('Plot Settings (to configure size and some elements of the plots) *Facultative',
                                                    help=st.session_state.help_paragraph)
                    if settings_checkbox :
                        st.session_state.aspect_plot = st.slider(
                            'Configure the aspect ratio of the plots', 0.0, 10.0, 2.5,
                            help="Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.")

                        st.session_state.height_plot = st.slider(
                            'Configure the height of the plots', 0, 50, 7,
                            help="Height (in inches) of each facet.")

                        st.session_state.size_xlabels = st.slider('Configure the size of the axis X labels', 0, 50, 25)
                        st.session_state.size_ylabels = st.slider('Configure the size of the axis Y labels', 0, 50, 25)


                    if st.button('I want to have the sdf file of poses in this conformation AND/OR the plots.'):             
                        st.session_state.ConformationClass.get_sdf_conformations(
                            1, st.session_state.RMSD_Target_conformation)


        else :
            if 'n_conformations' in st.session_state:
                try :
                    os.remove('Sorted_Heatmap.jpeg')
                    os.remove('Histograms_Best_Score.jpeg')
                    os.remove('Sample_Best_PLPScore_Poses.sdf')
                    os.remove('Best_PLPScore_Poses.sdf')
                    for i in range(st.session_state.n_conformations):
                        os.remove(f'Sample_Conformation{i+1}.sdf')
                        try :
                            os.remove(f'Conformation{i+1}.sdf')
                            os.remove(f'Barplot_Conformation{i+1}.jpeg')
                            os.remove(f'Scatter_Plot{i+1}.jpeg')
                            os.remove(f'Box_Plot{i+1}.jpeg')
                        except :
                            pass
                except :
                    pass        

            if 'indviduals_deleted' in st.session_state:
                del st.session_state.indviduals_deleted
            if 'sorted_heatmap' in st.session_state:
                del st.session_state.sorted_heatmap
            if 'predominant_poses' in st.session_state:
                del st.session_state.predominant_poses
            if 'df1' in st.session_state:
                del st.session_state.df1
            if 'histplot' in st.session_state:
                del st.session_state.histplot
            if 'output_liste' in st.session_state:
                del st.session_state.output_liste 
            if 'sample_predominant_poses' in st.session_state:
                del st.session_state.sample_predominant_poses
            if 'sample_indice_best_score' in st.session_state:
                del st.session_state.sample_indice_best_score
            if 'best_PLP_poses' in st.session_state:
                del st.session_state.best_PLP_poses
            if 'barplot' in st.session_state:
                del st.session_state.barplot
            if 'box_plot' in st.session_state:
                del st.session_state.box_plot
            if 'scatterplot' in st.session_state:
                del st.session_state.scatterplot



    else :
        if 'MCS_delete_activated' in st.session_state:
            if 'n_conformations' in st.session_state:
                try :
                    os.remove('Sorted_Heatmap.jpeg')
                    os.remove('Histograms_Best_Score.jpeg')
                    os.remove('Sample_Best_PLPScore_Poses.sdf')
                    os.remove('Best_PLPScore_Poses.sdf')
                    for i in range(st.session_state.n_conformations):
                        try :
                            os.remove(f'Sample_Conformation{i+1}.sdf')
                            os.remove(f'Conformation{i+1}.sdf')
                            os.remove(f'Barplot_Conformation{i+1}.jpeg')
                            os.remove(f'Scatter_Plot{i+1}.jpeg')
                            os.remove(f'Box_Plot{i+1}.jpeg')
                        except :
                            pass
                except :
                    pass
            

            for key in st.session_state.keys():
                del st.session_state[key]
