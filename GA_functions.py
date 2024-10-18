# GA_functions.py
# Original code written by Carl Ahlberg
# Adapted for the problem at hand by Martin Dahlgren
# IMPORTANT! ChatGPT _has_ been used as a coding assistant

import pandas as pd
import numpy as np
import random
import sys
import ast
import streamlit as st

MAX_FITNESS_CALLS = 15000 # how many fitness calls are allowed
MAX_DATA_ENTRIES = 10000 # limit the number of recipes in database. "Food.com" dataset has about 200.000 recipes.

# we store the fields we are interested in from the database in this class
class Recipe:
    def __init__(self, id, name, calories, protein, fat, carbs, tags):
        self.id = id
        self.name = name
        self.calories = calories
        self.protein = protein
        self.fat = fat
        self.carbs = carbs
        self.tags = tags # like "vegetarian", "breakfast" etc

# we use a seperate function to load the dataframe so that streamlit can cache it.
# this avoids the database having to reload when something changes that would trigger a streamlit rerun.
@st.cache_data
def load_data(_database_filename) -> pd.DataFrame:
    df = pd.read_csv(_database_filename)
    return df

# GA class is just a collection of functions for a Genetic Algorithm made by Carl Ahlberg.
# Using a class so that they form a logical group to be used elsewhere.
class GA_functions:
    def __init__(self, database_filename):
        self.filename = database_filename # keep a reference to filename in case we need it later

        # here we load the database and store it as a dataframe 
        self.data = load_data(database_filename)
        # to make it more managable the data is reduced. "Food.com" datasset has about 200.000 recepies.
        self.data = self.data.sample(MAX_DATA_ENTRIES) # FIXME: make this user configurable from streamlit page.
        self.data_sub = self.data.copy() # self.data_sub is a copy that we can edit and alter (we still have access to unedited data in self.data)
        self.bestSolution = None # here we store the "individual" that ends up with the collection recipes that best match what the user requested.

    # here we convert the dataframe into objects of the Recipe class
    def updateDataBase(self):
        """
        Loads the recipe database from a CSV file using Pandas and returns a list of Recipe objects.

        Parameters:
        - filename (str): Path to the recipes CSV file.

        Returns:
        - recipes (list of Recipe): List containing all Recipe objects.
        - numRecipes (int): Total number of recipes.
        """
        
        # The "Food.com" dataset has a column "nutrition" where a value in a row might look like "234, 3, 5, 2"
        # The first value is calories. The following code creates new column called "calories" holding just that value.
        self.data_sub['calories'] = self.data_sub['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.data_sub['calories'] = self.data_sub['calories'].apply(lambda x: x[0])

        # a small safety check to see that our database holds all the columns we need.
        required_columns = ['id', 'name', 'nutrition', 'ingredients_list'] # FIXME: "tags" not implemented yet!
        for column in required_columns:
            if column not in self.data_sub.columns:
                print(f"Error: Missing required column '{column}' in the dataset.")
                sys.exit()

        # loop over our data and create a Recipe object for each.
        recipes = []
        for _, row in self.data_sub.iterrows():
            if pd.isnull(row['nutrition']) or pd.isnull(row['name']) or pd.isnull(row['id']):
                continue  # Skip incomplete data

            # FIXME: not implemented yet.
            # tags = row['dietary_tags']
            # if pd.isnull(tags):
            #     tags = []
            # else:
            #     tags = [tag.strip().lower() for tag in tags.split(';')]

            recipe = Recipe(
                id=int(row['id']),
                name=row['name'],
                calories=float(row['calories']),
                protein=float(row['protein']),
                fat=float(row['fat']),
                carbs=float(row['carbs']),
                tags=[] # FIXME: not implemented yet
            )
            print(f"Loaded Recipe: {recipe.name}, Calories: {recipe.calories}, Protein: {recipe.protein}, Fat: {recipe.fat}, Carbs: {recipe.carbs}")
            recipes.append(recipe) # finally store each Recipe object in a list

        numRecipes = len(recipes)
        print(f"Loaded {numRecipes} recipes from {self.filename}.")
        return recipes, numRecipes
    
    # let the user make changes to our data, ie filter by ingredients etc..
    def setSubSelection(self, selection: pd.DataFrame):
        self.data_sub = selection
        self.updateDataBase()
        return
    
    def getSubSelection(self) -> pd.DataFrame:
        return self.data_sub

    # check that the parameters to the genetic algorithm are reasonable
    def checkErrorsInParameters(self, maxGeneration, populationSize, numNewOffspring, mutationProbability, numberMutations, tournamentSize):
        error = 0
        numFitnessCalls = maxGeneration * numNewOffspring + populationSize
        if numFitnessCalls > MAX_FITNESS_CALLS:
            print(f'You are calculating the fitness {numFitnessCalls} times. The maximum is {MAX_FITNESS_CALLS}.')
            error = 1
        if maxGeneration < 10:
            print(f'You set \'maxGeneration\' to {maxGeneration}. The minimum is 10.')
            error = 1
        if populationSize < 5:
            print(f'You set \'populationSize\' to {populationSize}. The minimum is 5.')
            error = 1
        if numNewOffspring < 1:
            print(f'You set \'numNewOffspring\' to {numNewOffspring}. The minimum is 1.')
            error = 1
        if numberMutations < 1:
            print(f'You set \'numberMutations\' to {numberMutations}. The minimum is 1.')
            error = 1
        if not (0 <= mutationProbability <= 1):
            print(f'You set \'mutationProbability\' to {mutationProbability}. The correct range is [0, 1].')
            error = 1
        if not (2 <= tournamentSize <= populationSize):
            print(f'You set \'tournamentSize\' to {tournamentSize}. The correct range is [2, {populationSize}].')
            error = 1
        if error:
            sys.exit()
            

    def newPopulation(self, populationSize, numRecipes, maxRecipes=12):
        """
        Initializes a new population for the GA with a limited number of recipes selected per individual.

        Parameters:
        - populationSize (int): Number of individuals in the population.
        - numRecipes (int): Total number of available recipes.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - population (list of numpy arrays): List containing all individuals.
        """
        # here we initialize our "population".
        # "population" is just a list of "individuals".
        # "individuals" is just list of zeroes or ones meaning if a recipe of that "index" is included or not.
        # the index is referring to the main "recipes" list returned from self.updateDataBase at startup
        population = []
        for _ in range(populationSize):
            # first set all to zero
            individual = np.zeros(numRecipes, dtype=int) 
            selected_indices = np.random.choice(numRecipes, size=maxRecipes, replace=False)
            # then select random indexes to be equal to 1.
            # we dont use bool because we want to calculate the sum of "ones" later to check we dont have to many recipes per individual.
            # FIXME: an idea i have is that some recipes could have higher value depending on user rating for instance to make it more
            # likely to be included.
            individual[selected_indices] = 1 
            population.append(individual)
        return population
    
    # here we calculate how much "individuals" collection of recipes differ from the target goals
    def calculateFitness(self, individuals, recipes, targetCalories, targetProteins, targetFat, targetCarbs, dietaryRestrictions=[], lowerBound=None, upperBound=None, maxRecipes=12):
        """
        Calculates the fitness of each individual in the population.

        Parameters:
        - individuals (list of numpy arrays): Current population.
        - recipes (list of Recipe): List of all available recipes.
        - targetCalories (float): Desired total calorie intake.
        # FIXME: implement dietaryRestrictions (list of str): List of dietary restrictions (e.g., ['vegetarian']).
        - lowerBound (float): Minimum acceptable total calories.
        - upperBound (float): Maximum acceptable total calories.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - populationFitness (list of float): Fitness scores for each individual.
        - calorieDiffs (list of float): Signed difference between total calories and target calories for each individual.
        """
        populationFitness = []
        calorieDiffs = []  # to store calorie differences for each individual

        # loop over all individuals and calculate how well their collection of recipes match the target goal.
        for individual in individuals:
            totalCalories = 0.0
            totalProtein = 0.0
            totalFat = 0.0
            totalCarbs = 0.0
            valid = True # FIXME: implement this to have different diets, like "vegetarian"

            # "recipes" is an array 
            # "individual" is an array of same length as recipes that holds a number (0 or 1) indicating if the recipe at equal index should be included or not
            # the zip function will combine the two so we can know which recipe is include. very pythonic!
            for gene, recipe in zip(individual, recipes):
                if gene:
                    # FIXME: implement different diets!
                    # Check dietary restrictions
                    # if dietaryRestrictions:
                    #    if not all(restriction in recipe.tags for restriction in dietaryRestrictions):
                    #        valid = False
                    #        break
                    totalCalories += recipe.calories
                    totalProtein += recipe.protein
                    totalFat += recipe.fat
                    totalCarbs += recipe.carbs

            numSelectedRecipes = np.sum(individual)

            # this first check for correct diet is not implemented yet and will not ever happen.
            # FIXME: not yet implemented (Dietary restrictions)
            if not valid: 
                # set to high value. float('inf') is better, but might make visualization harder.
                fitness = 1e6 # float('inf')  # Penalize invalid individuals
                calorie_diff = 1e6 # float('inf')  # No meaningful calorie difference
            # the diet is ok, carry on...
            else:
                # calculate signed difference between totalCalories and targetCalories
                # this is used only for visuals so that our histogram plot shows
                # values on both sides of the peak.
                calorie_diff = totalCalories - targetCalories

                # fitness is the absolute difference, we only care how big the
                # difference is from the goal, not if it's above or below the target value.
                fitness = abs(calorie_diff)

                # penalty for violating calorie bounds
                if lowerBound and totalCalories < lowerBound:
                    fitness += (lowerBound - totalCalories) * 10
                if upperBound and totalCalories > upperBound:
                    fitness += (totalCalories - upperBound) * 10

                # penalize for exceeding maxRecipes
                if numSelectedRecipes > maxRecipes:
                    fitness += abs(numSelectedRecipes - maxRecipes) * 100

                # penalize for missing protein, fat and carb targets.
                protein_diff = abs(targetProteins - totalProtein) * 100
                fat_diff = abs(targetFat - totalFat) * 100
                carbs_diff = abs(targetCarbs - totalCarbs) * 100

                fitness += protein_diff + fat_diff + carbs_diff

            populationFitness.append(fitness) # for algorithm
            calorieDiffs.append(calorie_diff) # for better visuals

        return populationFitness, calorieDiffs

    # each generation some individuals are set out for tournament,
    # the winners will be used to create new individuals for the next generation.
    def parentSelectionTournament(self, population, populationFitness, tournamentSize=3):
        """
        Selects two parents using tournament selection.

        Parameters:
        - population (list of numpy arrays): Current population.
        - populationFitness (list of float): Fitness scores.
        - tournamentSize (int): Number of individuals competing in each tournament.

        Returns:
        - parents (list of numpy arrays): Two selected parents.
        """
        parents = []
        for _ in range(2):
            # very pythonic! pick the 2 best individuals from a random set consisting of "tournamentSize" number of contestants. 
            tournament_indices = random.sample(range(len(population)), tournamentSize)
            tournament_fitness = [populationFitness[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]  # Lower fitness is better
            parents.append(population[winner_index].copy())

        #print(f"Selected Parent Fitness: {parents[0]} - Sum: {sum(parents[0])}")
        return parents

    # here we take two "parent" individuals and switch there values from a random point until the end.
    # doing so could potentially create an individual with more than "maxRecipes" of ones, so we need 
    # to handle that too.
    def crossover(self, parents, maxRecipes=12):
        """
        Performs single-point crossover between two parents and ensures offspring do not exceed maxRecipes.

        Parameters:
        - parents (list of numpy arrays): Two parent individuals.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - offspring1, offspring2 (numpy arrays): Two offspring individuals.
        """
        crossover_point = random.randint(1, len(parents[0]) - 1)
        offspring1 = np.concatenate([parents[0][:crossover_point], parents[1][crossover_point:]])
        offspring2 = np.concatenate([parents[1][:crossover_point], parents[0][crossover_point:]])

        # ensure offspring do not exceed maxRecipes
        for offspring in [offspring1, offspring2]:
            num_selected = np.sum(offspring)
            if num_selected > maxRecipes:
                # randomly deselect excess recipes
                excess = int(num_selected - maxRecipes)
                selected_indices = np.where(offspring == 1)[0]
                deselect_indices = np.random.choice(selected_indices, size=excess, replace=False)
                offspring[deselect_indices] = 0

        return offspring1, offspring2

    # just flip the value at random point but check so that maxRecipes is not exceeded
    def mutation(self, individual, mutationProbability, numMutations, maxRecipes=12):
        """
        Applies bit-flip mutation to an individual without exceeding the maximum number of recipes.

        Parameters:
        - individual (numpy array): The individual to mutate.
        - mutationProbability (float): Probability of each mutation.
        - numMutations (int): Number of mutations to apply.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - individual (numpy array): The mutated individual.
        """
        for _ in range(numMutations):
            if random.random() < mutationProbability:
                gene_idx = random.randint(0, len(individual) - 1)
                if individual[gene_idx] == 1:
                    individual[gene_idx] = 0
                else:
                    # Ensure not to exceed maxRecipes
                    if np.sum(individual) < maxRecipes:
                        individual[gene_idx] = 1
        return individual

    def updatePopulation(self, population, populationFitness, offsprings, offspringsFitness):
        """
        Updates the population by selecting the best individuals from the combined pool.

        Parameters:
        - population (list of numpy arrays): Current population.
        - populationFitness (list of float): Fitness scores of current population.
        - offsprings (list of numpy arrays): New offspring.
        - offspringsFitness (list of float): Fitness scores of offspring.

        Returns:
        - new_population (list of numpy arrays): Updated population.
        - new_fitness (list of float): Updated fitness scores.
        """
        combined = list(zip(population, populationFitness)) + list(zip(offsprings, offspringsFitness))
        # Sort by fitness (lower is better)
        combined.sort(key=lambda x: x[1])
        # Select the top individuals to form the new population
        new_population = [ind for ind, fit in combined[:len(population)]]
        new_fitness = [fit for ind, fit in combined[:len(population)]]
        return new_population, new_fitness

    def visualization(self, population, populationFitness, recipes, g, maxGeneration, showBestSolution, showPopulationDistribution, bestFitness, targetCalories):
        """
        Visualizes the GA progress by printing the best meal plan and optionally plotting fitness distributions.

        Parameters:
        - population (list of numpy arrays): Current population.
        - populationFitness (list of float): Fitness scores.
        - recipes (list of Recipe): List of all recipes.
        - g (int): Current generation.
        - maxGeneration (int): Total number of generations.
        - showBestSolution (int): Flag to show the best solution.
        - showPopulationDistribution (int): Flag to show population fitness distribution.
        - bestFitness (float): Best fitness found so far.
        - name (str): User's name.
        - targetCalories (float): Desired calorie intake.

        Returns:
        - bestFitness (float): Updated best fitness.
        """
        currentBest = min(populationFitness)
        indexBestSolution = populationFitness.index(currentBest)

        if currentBest < bestFitness or ((g+1) == maxGeneration):
            print(f'\nGeneration {g+1}/{maxGeneration}')
            print(f'New Best Fitness: {currentBest} calorie difference')

            bestFitness = currentBest
            if showBestSolution or ((g+1) == maxGeneration):
                self.bestSolution = population[indexBestSolution]
                selectedRecipes = [recipes[i] for i, gene in enumerate(self.bestSolution) if gene]
                totalCalories = sum(recipe.calories for recipe in selectedRecipes)

                print("\nBest Meal Plan:")
                for recipe in selectedRecipes:
                    print(f"- {recipe.name} ({recipe.calories} kcal)")
                print(f"Total Calories: {totalCalories} kcal (Target: {targetCalories} kcal)")

                # Debugging: Print individual calories and verify
                for recipe in selectedRecipes:
                    print(f"- {recipe.name} ({recipe.calories} kcal)  ({recipe.protein} g)  ({recipe.fat} g)  ({recipe.carbs} g)")

        return bestFitness

    def getBestSolution(self) -> list[any]:
        return self.bestSolution