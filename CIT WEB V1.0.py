"""
#####################################################
##                                                 ##
##      -- STREAMLIT CHEMOINFOTOOLS V1.0 --        ##
##                                                 ##
#####################################################
"""

import streamlit as st
from scripts import CIT_WEB_ConformationTool, Coming_Soon

def main():

    about_str = \
    """
    **CIT : ChemoInfoTools 1.1**
    **Contact:** [Arthur Carré](mailto:arthur.carre@icloud.com)
    **License:** [MIT License](https://github.com/Arthurcarre/CIT/blob/master/LICENSE.md)
    """

    st.set_page_config(page_title = "CIT : ChemoInfoTools !",
                       page_icon = ":test_tube:",
                       layout = "centered",
                       initial_sidebar_state = "expanded",
                       menu_items = {"Get Help": "https://github.com/Arthurcarre/CIT/discussions",
                                     "Report a bug": "https://github.com/Arthurcarre/CIT/issues",
                                     "About": about_str}
                       )

    pages = ("CIT: ConformationTool", "CIT: Coming Soon")

    title = st.sidebar.title("ChemoInfoTools !")


    logo = st.sidebar.image("img/pmu_logo.jpg",
                            caption = "CIT was developed in cooperation with the Institute of Pharmacy of the Paracelsus Medical"
                            " Private University Salzburg (Austria) and the the Institute of Pharmacy of the University of Angers"
                            "(France).")

    page = st.sidebar.selectbox(label = "Select a ChemoInfoTool:",
                                options = pages,
                                index = 0,
                                help = "Select a tool that you want to use."
                                )

    doc_str = "**CIT** - short for **Chemo-Informatic Tools** - is a set of tools for chemo-informatics in order to manipulate or"
    doc_str += "analyse chemo-informatics data. To get started make sure to read the documentation in the [PIA Wiki](https://github.com/Arthurcarre/CIT/wiki)."
    doc_str += "For general help, questions, suggestions or any other feedback please refer "
    doc_str += "to the [GitHub repository](https://github.com/Arthurcarre/CIT/discussions) or contact us directly!"
    doc = st.sidebar.markdown(doc_str)

    contact_str = "**Contact:** [Arthur Carré](mailto:arthur.carre@icloud.com)"
    contact = st.sidebar.markdown(contact_str)

    license_str = "**License:** [MIT License](https://github.com/Arthurcarre/CIT/blob/master/LICENSE.md)"
    license = st.sidebar.markdown(license_str)
    
    logo2 = st.sidebar.image("img/ua_logo.jpg")
    
    if page == "CIT: ConformationTool":
        CIT_WEB_ConformationTool.main()
    elif page == "CIT: Coming Soon":
        Coming_Soon.main()
    else:
        CIT_WEB_ConformationTool.main()

if __name__ == "__main__":
    main()
