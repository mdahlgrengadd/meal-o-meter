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
from streamlit.components.v1 import html
# GA_main.py (add these parameters)
MAX_RECIPES_PER_INDIVIDUAL = 6  # Adjust based on your needs


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

# these are from the class video https://www.youtube.com/watch?v=asFqpMDSPdM
# https://github.com/dataprofessor/population-dashboard
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

    # create a dataframe for plotly Express
    df = pd.DataFrame({
        'Calorie Difference': populationFitness
    })

    nbins = int((df['Calorie Difference'].max() - df['Calorie Difference'].min()) / bin_size)
    # Create histogram with density
    fig = px.histogram(
        df, 
        x='Calorie Difference',
        nbins=nbins,
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
    # time.sleep(0.01) # slow down visualization

def placeholder_recipe() -> tuple[str, str]:
    recipe ="""
# Chocolate Chip Banana Bread
## 📝 Steps to Make
1. Preheat your oven to 350°F (175°C). Grease a 9x5 inch loaf pan.
2. In a large bowl, mix the mashed bananas and melted butter.
3. Add the sugar, beaten egg, and vanilla extract. Mix well.
4. Sprinkle the baking soda and salt over the mixture, then mix in the flour until just combined.
5. Fold in the chocolate chips.
6. Pour the batter into the prepared loaf pan.
7. Bake in the preheated oven for 50 minutes, or until a toothpick inserted into the center comes out clean.
8. Let cool in the pan for 10 minutes, then transfer to a wire rack to cool completely.

## 🏷️ Tags
- #Dessert
- #Vegetarian
- #Baking

---

"""
    ingredients = """
## 🕒 Time to Cook
- **Preparation Time**: 10 minutes
- **Cooking Time**: 50 minutes
- **Total Time**: 1 hour

## 🍽️ Ingredients
- **Servings**: 8 servings

### Main Ingredients
- 3 ripe bananas, mashed
- 1/3 cup melted butter
- 1/2 cup sugar
- 1 egg, beaten
- 1 teaspoon vanilla extract
- 1 teaspoon baking soda
- Pinch of salt
- 1 1/2 cups all-purpose flour
- 1/2 cup chocolate chips
"""
    return recipe, ingredients

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
        if st.button("Take me there!"):
            nav_page("Setup_Plan")
        return

    # get GA parameters from sliders
    populationSize = st.session_state.populationSize_slider #1000
    numNewOffspring = st.session_state.numNewOffspring_slider # 100  # Must ensure (maxGeneration * numNewOffspring + populationSize) <= 10,000
    maxGeneration = (MAX_FITNESS_CALLS - populationSize) // numNewOffspring
    mutationProbability = st.session_state.mutationProbability_slider # 0.05  # Probability for each mutation
    numberMutations = st.session_state.numberMutations_slider # 2 # Number of mutations per individual
    tournamentSize = st.session_state.tournamentSize_slider # 3

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


    final_solution = GA.getBestSolution()
    selectedRecipes = [recipes[i] for i, gene in enumerate(final_solution) if gene]
    totalCalories = float(sum(recipe.calories for recipe in selectedRecipes))
    totalProtein = float(sum(recipe.protein for recipe in selectedRecipes))
    totalFat = float(sum(recipe.fat for recipe in selectedRecipes))
    totalCarbs = float(sum(recipe.carbs for recipe in selectedRecipes))


    colored_header(
            label="Best Meal Plan",
            description="According to Science",
            color_name="red-70",
        )
    # create a tab menu for displaying the recipes
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Meal Plan Overview", "Recipe 1", "Recipe 2", "Recipe 3", "Recipe 4","Recipe 5" ,"Recipe 6"])
    for t in [tab2, tab3, tab4, tab5, tab6, tab7]:
        txt_rcp, txt_ing = placeholder_recipe()
        c1, c2 = t.columns([2,1])
        c1.markdown(txt_rcp)
        c2.markdown(txt_ing)

    
    # main result
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            for recipe in selectedRecipes:
                st.markdown(f"""
                            - **{recipe.name}** ({recipe.calories} kcal)\n
                            *[Protein: {recipe.protein}g / Fat: {recipe.fat}g / Carbs: {recipe.carbs}g]*
                            """)
            st.header("Nutrients", divider="grey")

            # display the result numerically
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric(
                label=f'Calories',
                value=f'{totalCalories:,.0f}',
                delta=totalCalories-targetCalories,
                delta_color='normal'
            )
            metric2.metric(
                label=f'Protein',
                value=f'{totalProtein:,.0f}',
                delta=totalProtein-targetProteins,
                delta_color='normal'
            )
            metric3.metric(
                label=f'Fat',
                value=f'{totalFat:,.0f}',
                delta=totalFat-targetFat,
                delta_color='normal'
            )
            metric4.metric(
                label=f'Carbs',
                value=f'{totalCarbs:,.0f}',
                delta=totalCarbs-targetCarbs,
                delta_color='normal'
            )
         
        # this are the circle plots showing percentages   
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
