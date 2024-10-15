# GA_main.py

from GA_functions import Recipe, GA_functions
import numpy as np
from GA_functions import MAX_FITNESS_CALLS
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
import altair as alt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import time
import plotly.express as px
from streamlit_extras.colored_header import colored_header

# GA_main.py (add these parameters)
MAX_RECIPES_PER_INDIVIDUAL = 6  # Adjust based on your needs

def make_donut(input_response, input_text, input_color):
  input_response = int(input_response)
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text

def update_plot(placeholder, populationFitness, bin_size, generation):
    x_range = [ st.session_state['cal_slider'] * -1, st.session_state['cal_slider']]

    # Create a DataFrame for Plotly Express
    df = pd.DataFrame({
        'Calorie Difference': populationFitness
    })

    # Define binning parameters
    min_diff = df['Calorie Difference'].min()
    max_diff = df['Calorie Difference'].max()

   # Handle case where all data points are identical
    if min_diff == max_diff:
        # Define a small range around the identical value
        aligned_min = min_diff - 1
        aligned_max = max_diff + 1

    else:
        # Ensure that 0 is within the bin range
        num_bins_neg = int(np.ceil(abs(min_diff) / bin_size))
        num_bins_pos = int(np.ceil(abs(max_diff) / bin_size))

        # Calculate the new min and max to align bins symmetrically around 0 if possible
        aligned_min = -num_bins_neg * bin_size
        aligned_max = num_bins_pos * bin_size

        # Handle case where min_diff < aligned_min or max_diff > aligned_max
        if min_diff < aligned_min:
            aligned_min = min_diff - bin_size
        if max_diff > aligned_max:
            aligned_max = max_diff + bin_size

    # Calculate the number of bins as an integer
    total_bins = int(np.ceil((aligned_max - aligned_min) / bin_size))
    if total_bins < 1:
        total_bins = 1  # Ensure at least one bin

    # If data is too sparse, adjust the bin size or range
    if total_bins < 2:
        aligned_min = min_diff - bin_size
        aligned_max = max_diff + bin_size
        total_bins = 2  # Ensure at least two bins


    # Create histogram with density
    fig = px.histogram(
        df, 
        x='Calorie Difference',
        nbins=total_bins,
        histnorm='density',
        opacity=0.7,
        title=f'Population Calorie Difference at Generation {generation + 1}',
        labels={'Calorie Difference': 'Calorie Difference From Goal'},
        template='plotly_white',
        color_discrete_sequence=px.colors.diverging.__dict__['RdBu']
    )


    fig.update_xaxes(range=x_range)
    fig.update_layout(
        title=f'Best Recipe Combination Try: {generation + 1}',
        xaxis_title='Calorie Difference',
        yaxis_title='Number of Individuals',
        bargap=0.1,
        template='plotly_white'
    )

    # Display plot in Streamlit
    placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(0.01) 

def main_page():
    progress_place = st.session_state["progress_placeholder"]
    progress_bar = progress_place.progress(0)

    # Visualization parameters
    showBestSolution = 1
    showPopulationDistribution = 1
    bestFitness = float('inf')

    # Upload the recipe database
    if 'FoodDataBase' not in st.session_state:
        raise SystemError
    
    GA = st.session_state.FoodDataBase
    try:
        recipes, numRecipes = GA.updateDataBase()
    except:
        st.warning("Error reading database! Did you forget to setup your diet?")
        return

    # GA parameters
    populationSize = 1000
    numNewOffspring = 100  # Must ensure (maxGeneration * numNewOffspring + populationSize) <= 10,000
    maxGeneration = (MAX_FITNESS_CALLS - populationSize) // numNewOffspring
    mutationProbability = 0.05  # Probability for each mutation
    numberMutations = 2 # Number of mutations per individual
    tournamentSize = 3

    GA.checkErrorsInParameters(maxGeneration, populationSize, numNewOffspring, mutationProbability, numberMutations, tournamentSize)

    # Initialize population
    population = GA.newPopulation(populationSize, numRecipes, MAX_RECIPES_PER_INDIVIDUAL)
    # Print first 5 individuals
    for i, individual in enumerate(population[:5], 1):
        selected = np.where(individual == 1)[0]
        print(f"Individual {i}: Selected Recipes = {selected}")


    targetCalories = st.session_state["cal_slider"]  # Example target
    targetFat = st.session_state["fat_slider"]  # Example target
    targetCarbs = st.session_state["carb_slider"]  # Example target
    targetProteins = st.session_state["prot_slider"]  # Example target

    dietaryRestrictions = ['vegetarian']  # Example restrictions (can be empty list if none)
    lowerBound = targetCalories - 200 # accept 200 calories below target
    upperBound = targetCalories + 200 # the same but 200 calories above target is ok too.

    # Calculate initial fitness
    populationFitness, calorieDiffs = GA.calculateFitness(population, recipes, targetCalories, targetProteins,targetFat,targetCarbs, dietaryRestrictions, lowerBound, upperBound)
    placeholder = st.sidebar.empty()
    placeholder2 = st.empty()

    # Main GA loop
    with st.spinner('Busy Doing Sexy Sciency Stuff...'):
        for g in range(maxGeneration):
            offsprings = []
            offspringFitness = []

            # Generate offspring
            for _ in range(numNewOffspring // 2):  # Each crossover produces two offspring
                parents = GA.parentSelectionTournament(population, populationFitness, tournamentSize)
                offspring1, offspring2 = GA.crossover(parents, MAX_RECIPES_PER_INDIVIDUAL)
                offspring1 = GA.mutation(offspring1, mutationProbability, numberMutations, MAX_RECIPES_PER_INDIVIDUAL)
                offspring2 = GA.mutation(offspring2, mutationProbability, numberMutations, MAX_RECIPES_PER_INDIVIDUAL)
                offsprings.extend([offspring1, offspring2])

            # Calculate fitness for offspring
            offspringFitness, calorieDiffs = GA.calculateFitness(offsprings, recipes, targetCalories, targetProteins,targetFat,targetCarbs, dietaryRestrictions, lowerBound, upperBound, MAX_RECIPES_PER_INDIVIDUAL)

            # Update population
            population, populationFitness = GA.updatePopulation(population, populationFitness, offsprings, offspringFitness)

            # Visualize progress
            bestFitness = GA.visualization(population, populationFitness, recipes, g, maxGeneration, showBestSolution, showPopulationDistribution, bestFitness, targetCalories)

            update_plot(placeholder2, calorieDiffs, populationSize//10, g)
            
            completion = float(g+1) / float(maxGeneration)
            progress_bar.progress(completion)


    #print("\nGA run complete.")
    progress_bar.progress(1.0)
    st.sidebar.success("Plan completed!")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Meal Plan Overview", "Recipe 1", "Recipe 2", "Recipe 3", "Recipe 4","Recipe 5" ,"Recipe 6"])

    # Result
    with tab1:
        final_solution = GA.getBestSolution()
        selectedRecipes = [recipes[i] for i, gene in enumerate(final_solution) if gene]
        totalCalories = float(sum(recipe.calories for recipe in selectedRecipes))
        totalProtein = float(sum(recipe.protein for recipe in selectedRecipes))
        totalFat = float(sum(recipe.fat for recipe in selectedRecipes))
        totalCarbs = float(sum(recipe.carbs for recipe in selectedRecipes))

        col1, col2 = st.columns([2, 1])
        with col1:
            colored_header(
                    label="Best Meal Plan",
                    description="According to Science",
                    color_name="red-70",
                )
            for recipe in selectedRecipes:
                st.markdown(f"""
                            - **{recipe.name}** ({recipe.calories} kcal)\n
                            *[Protein: {recipe.protein}g / Fat: {recipe.fat}g / Carbs: {recipe.carbs}g]*
                            """)
            st.header("Total Sum", divider="grey")
            st.write(f"Calories: {totalCalories} kcal (Target: {targetCalories} kcal)")
            st.write(f"Protein: {totalProtein}g (Target: {targetProteins}g)")
            st.write(f"Fat: {totalFat}g (Target: {targetFat}g)")
            st.write(f"Carbs: {totalCarbs}g (Target: {targetCarbs}g)")
            
        with col2:
            col3, col4 = st.columns(2)
            with col3:
                st.write(make_donut(min (100, (totalCalories / targetCalories) *100.0), "Calories", "red"))
                st.write(make_donut(min ( 100, ((totalProtein / targetProteins) *100.0) ) , "Protein", "green"))
            with col4:
                st.write(make_donut( min (100, (totalFat / targetFat)*100.0), "Fat", "orange"))
                st.write(make_donut( min (100, (totalCarbs / targetCarbs)*100.0), "Carbohydrates", "blue"))


#if __name__ == "__main__":
main_page()
