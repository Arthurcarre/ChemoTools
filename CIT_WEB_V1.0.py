"""
#####################################################
##                                                 ##
##         -- STREAMLIT CHEMOINFOTOOLS --          ##
##                                                 ##
#####################################################
"""

import streamlit as st
from scripts import CIT_WEB_MCS_ConformationTool, CIT_WEB_Murcko_ConformationTool, CIT_WEB_Unique_Molecule_ConformationTool, Coming_Soon

def main():


    st.set_page_config(page_title = "CIT : ChemoInfoTools !",
                       page_icon = ":computer:",
                       layout = "wide",
                       initial_sidebar_state = "expanded",
                       menu_items = {"Get Help": "https://github.com/Arthurcarre/ChemoTools",
                                     "Report a bug": "https://github.com/Arthurcarre/ChemoTools/issues"}
                       )

    pages = ("CIT: ConformationTool", "CIT: Coming Soon")

    title = st.sidebar.title("ChemoInfoTools Application")


    logo = st.sidebar.image("img/logo_pmu.png")
    logo2 = st.sidebar.image("img/ua_h_noir_ecran.png",                            
                             caption = "CIT was developed in cooperation with the Institute of Pharmacy of the Paracelsus Medical"
                            " Private University Salzburg (Austria) and the Institute of Pharmacy of the University of Angers"
                            " (France).")

    page = st.sidebar.selectbox(label = "Select a ChemoInfoTool:",
                                options = pages,
                                index = 0,
                                help = "Select a tool that you want to use."
                                )

    doc_str = "**CIT** - short for **Chemo-Informatic Tools** - is a set of tools for chemo-informatics in order to manipulate or "
    doc_str += "analyse chemo-informatics data. To get started make sure to read the documentation in the [PIA Wiki](https://github.com/Arthurcarre/ChemoTools/wiki)."
    doc_str += " For general help, questions, suggestions or any other feedback please refer "
    doc_str += "to the [GitHub repository](https://github.com/Arthurcarre/ChemoTools) or contact us directly!"
    doc = st.sidebar.markdown(doc_str)

    contact_str = "**Contact:** [Arthur Carré](mailto:arthur.carre@icloud.com)"
    contact = st.sidebar.markdown(contact_str)

    license_str = "**License:** [MIT License](https://github.com/Arthurcarre/ChemoTools/blob/main/LICENSE.md)"
    license = st.sidebar.markdown(license_str)
    
    st.sidebar.info('''
        Made by [Arthur Carré](mailto:arthur.carre@icloud.com) with **_Streamlit_**.
    ''')
    
    if page == "CIT: ConformationTool":
        st.header("CIT : ConformationTool !")
        st.write("""
    This tool aims to isolate, within the results of docking simulations, the different consensual conformations and
    to quantify the consistency of the poses for each molecule of the same family.
    """)
        tab1, tab2, tab3 = st.tabs(["CIT : Maximum Common Substructure", "CIT : Murcko Scaffold", "CIT : Unique Molecule"])
        with tab1:
            CIT_WEB_MCS_ConformationTool.main()
        with tab2:
            CIT_WEB_Murcko_ConformationTool.main()
        with tab3:
            CIT_WEB_Unique_Molecule_ConformationTool.main()
    elif page == "CIT: Coming Soon":
        Coming_Soon.main()



if __name__ == "__main__":
    main()
    
