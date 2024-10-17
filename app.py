import streamlit as st
from GA_functions import GA_functions
from streamlit_extras.app_logo import add_logo
import utils as utl
from streamlit.components.v1 import html

DB_NAME = "data/RAW_recipes.zip"

# https://github.com/streamlit/streamlit/issues/4832
# this hack allows for a button to switch page.
def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

# initialization
def PageSetup():
    # create an instance of GA_functions class and store it
    # in session_state so that it can be referenced later.
    if 'FoodDataBase' not in st.session_state:
        st.session_state['FoodDataBase'] = GA_functions(DB_NAME)

    # the utl class is from https://github.com/BugzTheBunny/streamlit_custom_gui.git
    # it allows for custom css file to be used.
    utl.set_page_title('Meal-O-Meter')
    utl.local_css("frontend/css/streamlit.css")
    utl.remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

    add_logo("Logo.png", height=400)
    # add sliders to the sidebar.
    # we use a form so that the sliders can be manipulated
    # without triggering a rerun until users clicks "run"
    with st.sidebar.form("my_form"):
        calories_goal = st.slider(
            'Select Total Calories (Per Plan)',
            1400, 4500, (2600), key="cal_slider"
        )
        prot_goal = st.slider(
            'Select Total Protein (g) (Per Plan)',
            0, 400, (100), key="prot_slider"
        )
        fat_goal = st.slider(
            'Select Total Fat (g) (Per Plan)',
            0, 200, (50), key="fat_slider"
        )
        carbs_goal = st.slider(
            'Select Total Carbs (g) (Per Plan)',
            0, 500, (200), key="carb_slider"
        )

        # here we put the run button side to side with the progress bar
        col1, col2 = st.columns([1,3])
        with col1:
            if st.form_submit_button("Run"):
                    nav_page("Run_Plan")
            
        progress_bar_placeholder = col2.empty()
        st.session_state["progress_placeholder"] = progress_bar_placeholder

    # main page links
    pg = st.navigation([st.Page("pages/Home.py", title=f"🏚️ Home"), st.Page("pages/1_Setup_Plan.py", title="⚙️ Setup Diet"), st.Page("pages/2_Run_Plan.py", title="🚀 Run Meal Plan")])
    pg.run()
    
    
PageSetup()

