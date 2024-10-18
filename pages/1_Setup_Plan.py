# Laboration 3

import streamlit as st
import pandas as pd
import numpy as np
import ast
import streamlit as st
from streamlit.hello.utils import show_code
from GA_functions import GA_functions
from typing import cast
### PLOTTING
#from st_aggrid import AgGrid, GridOptionsBuilder 
#import matplotlib.pyplot as plt
import altair as alt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.chart_container import chart_container


# this function is used to cache the data so that it is not reloaded every time the page is rerun
# also changing sliders etc will be much faster.
@st.cache_data
def cached_data(_GA: GA_functions = None):

    
    df = _GA.data

    # clean up data, remove rows with empty fields
    # https://stackoverflow.com/questions/29314033/drop-rows-containing-empty-cells-from-a-pandas-dataframe
    df['description'].replace('', np.nan, inplace=True)
    df.dropna(subset=['description'], inplace=True)

    df['name'].replace('', np.nan, inplace=True)
    df.dropna(subset=['name'], inplace=True)

    # make a new dataframe with only the columns where interested in
    cropped_data = df[["id", "name","minutes", "nutrition","ingredients"]]

    # this converts a "string representation of a list" into an actual list
    # then we can use it in a multiselect widget to filter individual ingredients
    # FIXME: however the ingredients are not very well structured, for instance there is "milk", "milk 0.1%" etc
    cropped_data['ingredients_list'] = cropped_data['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    cropped_data['ingredients_list'] = cropped_data['ingredients_list'].apply(lambda x: [i.strip().lower() for i in x])


    # to be able to filter rows according to ingredients, each ingredient has to
    # be on its own row. this is what explode does.
    all_ingredients = cropped_data.explode('ingredients_list')
    # remove duplicates using unique()


    # The nutrition holds an list of numbers for calories, proteins etc... 
    # Make them each their own column and convert string to number...
    # calories
    all_ingredients['calories'] = all_ingredients['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_ingredients['calories'] = all_ingredients['calories'].apply(lambda x: x[0])

    # proteins
    all_ingredients['protein'] = all_ingredients['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_ingredients['protein'] = all_ingredients['protein'].apply(lambda x: x[4])

    # fat
    all_ingredients['fat'] = all_ingredients['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_ingredients['fat'] = all_ingredients['fat'].apply(lambda x: x[1])

    # carbs
    all_ingredients['carbs'] = all_ingredients['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_ingredients['carbs'] = all_ingredients['carbs'].apply(lambda x: x[2])

    return all_ingredients

def main_page():
    # use cast only to make it readable for type checkers
    if 'FoodDataBase' not in st.session_state:
                raise SystemError
    
    GA = cast(GA_functions, st.session_state.FoodDataBase)

    with st.form("setup_form"):
        placeholder = st.empty()
        with st.spinner('Wait for it...'):
            # https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags

            all_ingredients = cached_data(GA)
            
            unique_selection = sorted(all_ingredients.index.unique()) # used in the multiselect box below

            # here the user can set a range from low to high of how many calories that each recipe should fall into
            #with cent_co:
            with placeholder:
                left_co, cent_co,last_co = st.columns(3)
                calories_per_recipe = cent_co.slider(
                'Recipes by Caloric Range (not what you think)',
                1, 2000, (1, 900)
                )
                
                with last_co:
                    if st.form_submit_button("Apply"):
                        st.success("Filter Applied!")
        
                rng = range(calories_per_recipe[0],calories_per_recipe[1])
                df_selection = all_ingredients.loc[all_ingredients.calories.isin(rng)]

                # FIXME: allow for user to set how many recipes should be included from the dataset
                # with last_co:
                #     max_recipes_slider = st.slider(
                #         'Limit Amount of Recipes',
                #         100, 100000, (5000), key="max_recipes"
                #     )


                # here the user can filter recipes according to ingredients
                unique_ingredients = sorted(all_ingredients["ingredients_list"].unique())
                with left_co:
                    ingredients = st.multiselect(
                        "Ingredients (optional and buggy)",
                        options=unique_ingredients)
                
                if ingredients:
                    df_selection = df_selection.query(
                        "ingredients_list == @ingredients & index == @unique_selection"
                    )
                else:
                        df_selection = df_selection.query(
                        "index == @unique_selection"
                    )


            # this is the final resulting dataframe after cleaning and filtering
            df_grouped = df_selection.groupby(["id", "name" ,"minutes","nutrition", "calories", "protein", "fat", "carbs"])['ingredients_list'].apply(list).reset_index()


    # FIXME: AGrid not working when deploying to streamlit web
    # so comment it out and use the standard instead

    # gb = GridOptionsBuilder()
    # # makes columns resizable, sortable and filterable by default
    # gb.configure_default_column(
    #     resizable=True,
    #     filterable=True,
    #     sortable=True,
    #     editable=False,
    # )
    # gb.configure_column(
    # field="name", header_name="Recipe", width=450, tooltipField="Recipe"
    # )
    # gb.configure_column(
    # field="calories", header_name="Kcal", width=75, tooltipField="Calories"
    # )
    # gb.configure_column(
    # field="protein", header_name="Protein", width=100, tooltipField="Protein"
    # )
    # gb.configure_column(
    # field="fat", header_name="Fat", width=75, tooltipField="Fat"
    # )
    # gb.configure_column(
    # field="carbs", header_name="Sugar", width=75, tooltipField="Sugar"
    # )
    # gb.configure_column(
    # field="ingredients_list", header_name="Ingredients", width=500, tooltipField="Ingredients"
    # )
    # #makes tooltip appear instantly
    # gb.configure_grid_options(tooltipShowDelay=0)
    # gb.configure_pagination(enabled=True)
    # go = gb.build()

    #with placeholder_table:
        # ag = AgGrid(df_grouped[[ "name", "calories", "protein", "fat", "carbs", "ingredients_list",]],gridOptions=go, width=800, height=400)
        #st.write(df_grouped[[ "name", "calories", "protein", "fat", "carbs", "ingredients_list"]])

    GA.setSubSelection(df_grouped)

    with chart_container(df_grouped):
        #st.plotly_chart(fig, use_container_width=False)
        fig = ff.create_distplot([df_grouped['calories']], ["Calories"], bin_size=10,show_rug=False)
        fig.update_layout(
            title=f'Average Calories',
            xaxis_title='Calories',
            yaxis_title='Number of Recipes',
            bargap=0.1,
            template='plotly_white',
        )
        st.plotly_chart(fig, use_container_width=False)

main_page()

