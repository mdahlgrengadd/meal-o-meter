# GA_functions.py
# Original code written by Carl Ahlberg
# Adapted for the problem at hand by Martin Dahlgren


import pandas as pd
import numpy as np
import random
import sys
import ast
import streamlit as st

MAX_FITNESS_CALLS = 15000 # how many fitness calls are allowed
MAX_DATA_ENTRIES = 10000 # number of recipes in database

class Recipe:
    def __init__(self, id, name, calories, protein, fat, carbs, tags):
        self.id = id
        self.name = name
        self.calories = calories
        self.protein = protein
        self.fat = fat
        self.carbs = carbs
        self.tags = tags

@st.cache_data
def load_data(_database_filename) -> pd.DataFrame:
    df = pd.read_csv(_database_filename)
    return df

class GA_functions:
    def __init__(self, database_filename):
        self.filename = database_filename
        self.data = load_data(database_filename)
        self.data = self.data.sample(MAX_DATA_ENTRIES)
        self.data_sub = self.data
        self.bestSolution = None

    def updateDataBase(self):
        """
        Loads the recipe database from a CSV file using Pandas and returns a list of Recipe objects.

        Parameters:
        - filename (str): Path to the recipes CSV file.

        Returns:
        - recipes (list of Recipe): List containing all Recipe objects.
        - numRecipes (int): Total number of recipes.
        """
        try:
            # then we can use it in a multiselect widget to filter individual ingredients
            self.data_sub['calories'] = self.data_sub['nutrition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            self.data_sub['calories'] = self.data_sub['calories'].apply(lambda x: x[0])
        except FileNotFoundError:
            print(f"Error: The file {self.filename} was not found.")
            sys.exit()
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
            sys.exit()
        except pd.errors.ParserError:
            print("Error: The CSV file is malformed.")
            sys.exit()

        required_columns = ['id', 'name', 'nutrition']
        for column in required_columns:
            if column not in self.data_sub.columns:
                print(f"Error: Missing required column '{column}' in the dataset.")
                sys.exit()

        recipes = []
        for _, row in self.data_sub.iterrows():
            if pd.isnull(row['nutrition']) or pd.isnull(row['name']) or pd.isnull(row['id']):
                continue  # Skip incomplete data

            #tags = row['dietary_tags']
            #if pd.isnull(tags):
            #    tags = []
            #else:
            #    tags = [tag.strip().lower() for tag in tags.split(';')]

            recipe = Recipe(
                id=int(row['id']),
                name=row['name'],
                calories=float(row['calories']),
                protein=float(row['protein']),
                fat=float(row['fat']),
                carbs=float(row['carbs']),
                tags=[]
            )
            print(f"Loaded Recipe: {recipe.name}, Calories: {recipe.calories}, Protein: {recipe.protein}, Fat: {recipe.fat}, Carbs: {recipe.carbs}")
            recipes.append(recipe)

        numRecipes = len(recipes)
        print(f"Loaded {numRecipes} recipes from {self.filename}.")
        return recipes, numRecipes
    
    def setSubSelection(self, selection: pd.DataFrame):
        self.data_sub = selection
        self.updateDataBase()
        return
    
    def getSubSelection(self) -> pd.DataFrame:
        return self.data_sub

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
            

    # GA_functions.py (updated)

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
        population = []
        for _ in range(populationSize):
            individual = np.zeros(numRecipes, dtype=int)
            selected_indices = np.random.choice(numRecipes, size=maxRecipes, replace=False)
            individual[selected_indices] = 1
            population.append(individual)
        return population
    
    def calculateFitness(self, individuals, recipes, targetCalories, targetProteins, targetFat, targetCarbs, dietaryRestrictions=[], lowerBound=None, upperBound=None, maxRecipes=12):
        """
        Calculates the fitness of each individual in the population.

        Parameters:
        - individuals (list of numpy arrays): Current population.
        - recipes (list of Recipe): List of all available recipes.
        - targetCalories (float): Desired total calorie intake.
        - dietaryRestrictions (list of str): List of dietary restrictions (e.g., ['vegetarian']).
        - lowerBound (float): Minimum acceptable total calories.
        - upperBound (float): Maximum acceptable total calories.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - populationFitness (list of float): Fitness scores for each individual.
        - calorieDiffs (list of float): Signed difference between total calories and target calories for each individual.
        """
        populationFitness = []
        calorieDiffs = []  # To store calorie differences for each individual

        for individual in individuals:
            totalCalories = 0.0
            totalProtein = 0.0
            totalFat = 0.0
            totalCarbs = 0.0
            valid = True


            for gene, recipe in zip(individual, recipes):
                if gene:
                    # Check dietary restrictions
                    #if dietaryRestrictions:
                    #    if not all(restriction in recipe.tags for restriction in dietaryRestrictions):
                    #        valid = False
                    #        break
                    totalCalories += recipe.calories
                    totalProtein += recipe.protein
                    totalFat += recipe.fat
                    totalCarbs += recipe.carbs

            numSelectedRecipes = np.sum(individual)

            if not valid: # Not yet implemented (Dietary restrictions)
                fitness = 1e6# float('inf')  # Penalize invalid individuals
                calorie_diff =1e6# float('inf')  # No meaningful calorie difference
            else:
                # Calculate signed difference between totalCalories and targetCalories
                calorie_diff = totalCalories - targetCalories

                # Fitness is the absolute difference
                fitness = abs(calorie_diff)

                # Penalty for violating calorie bounds
                if lowerBound and totalCalories < lowerBound:
                    fitness += (lowerBound - totalCalories) * 10
                if upperBound and totalCalories > upperBound:
                    fitness += (totalCalories - upperBound) * 10

                # Penalize for exceeding maxRecipes
                if numSelectedRecipes > maxRecipes:
                    fitness += (numSelectedRecipes - maxRecipes) * 100

                protein_diff = abs(targetProteins - totalProtein)
                fat_diff = abs(targetFat - totalFat)
                carbs_diff = abs(targetCarbs - totalCarbs)

                fitness += (protein_diff * 0.5)*100 + (fat_diff * 0.5)*100 + (carbs_diff * 0.5)*100

            populationFitness.append(fitness)
            calorieDiffs.append(calorie_diff)  # Store the signed calorie difference

        # Return both the original populationFitness and the caloric differences
        return populationFitness, calorieDiffs


    def calculateFitnessAbsolute(self, individuals, recipes, targetCalories, dietaryRestrictions=[], lowerBound=None, upperBound=None, maxRecipes=12):
        """
        Calculates the fitness of each individual in the population.

        Parameters:
        - individuals (list of numpy arrays): Current population.
        - recipes (list of Recipe): List of all available recipes.
        - targetCalories (float): Desired total calorie intake.
        - dietaryRestrictions (list of str): List of dietary restrictions (e.g., ['vegetarian']).
        - lowerBound (float): Minimum acceptable total calories.
        - upperBound (float): Maximum acceptable total calories.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - populationFitness (list of float): Fitness scores for each individual.
        """
        populationFitness = []
        for individual in individuals:
            totalCalories = 0
            totalProtein = 0
            totalFat = 0
            totalCarbs = 0
            valid = True

            for gene, recipe in zip(individual, recipes):
                if gene:
                    # Check dietary restrictions
                    #if dietaryRestrictions:
                    #    if not all(restriction in recipe.tags for restriction in dietaryRestrictions):
                    #        valid = False
                    #        break
                    totalCalories += recipe.calories
                    totalProtein += recipe.protein
                    totalFat += recipe.fat
                    totalCarbs += recipe.carbs

            numSelectedRecipes = np.sum(individual)

            if not valid:
                fitness = 1e6 # Use a large value instead of float('inf')  # Penalize invalid individuals
            else:
                calorie_diff = abs(targetCalories - totalCalories)
                fitness = calorie_diff  # Primary fitness metric

                # Penalty for violating calorie bounds
                if lowerBound and totalCalories < lowerBound:
                    fitness += (lowerBound - totalCalories) * 10
                if upperBound and totalCalories > upperBound:
                    fitness += (totalCalories - upperBound) * 10

                # Penalize for exceeding maxRecipes
                if numSelectedRecipes > maxRecipes:
                    fitness += (numSelectedRecipes - maxRecipes) * 100  # Adjust penalty weight as needed

                # Optional: Penalize for nutritional imbalances
                desiredProtein = 50  # Example target in grams
                desiredFat = 70      # Example target in grams
                desiredCarbs = 250   # Example target in grams

                protein_diff = max(0, desiredProtein - totalProtein)
                fat_diff = max(0, desiredFat - totalFat)
                carbs_diff = max(0, desiredCarbs - totalCarbs)

                #fitness += (protein_diff * 0.5) + (fat_diff * 0.3) + (carbs_diff * 0.2)

            populationFitness.append(fitness)
        return populationFitness

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
            tournament_indices = random.sample(range(len(population)), tournamentSize)
            tournament_fitness = [populationFitness[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]  # Lower fitness is better
            parents.append(population[winner_index].copy())

        #print(f"Selected Parent Fitness: {parents[0]} - Sum: {sum(parents[0])}")
        return parents

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

        # Ensure offspring do not exceed maxRecipes
        for offspring in [offspring1, offspring2]:
            num_selected = np.sum(offspring)
            if num_selected > maxRecipes:
                # Randomly deselect excess recipes
                excess = int(num_selected - maxRecipes)
                selected_indices = np.where(offspring == 1)[0]
                deselect_indices = np.random.choice(selected_indices, size=excess, replace=False)
                offspring[deselect_indices] = 0

        return offspring1, offspring2

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
        if showPopulationDistribution:
            # plt.figure(2)
            # plt.clf()
            # plt.hist(populationFitness, bins=20, color='blue', alpha=0.7)
            # plt.title(f'Population Fitness at Generation {g+1}')
            # plt.xlabel('Calorie Difference')
            # plt.ylabel('Number of Individuals')
            # plt.pause(0.01)
            # plt.draw()
            pass

        # if (g+1) == maxGeneration:
        #     print('\nFinal Best Fitness:', currentBest)
        #     if showBestSolution:
        #         self.bestSolution = population[indexBestSolution]
        #         selectedRecipes = [recipes[i] for i, gene in enumerate(self.bestSolution) if gene]
        #         totalCalories = sum(recipe.calories for recipe in selectedRecipes)

        #         print("\nFinal Best Meal Plan:")
        #         for recipe in selectedRecipes:
        #             print(f"- {recipe.name} ({recipe.calories} kcal)")
        #         print(f"Total Calories: {totalCalories} kcal (Target: {targetCalories} kcal)")

        return bestFitness

    def getBestSolution(self) -> list[any]:
        return self.bestSolution