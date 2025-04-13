# GA_functions.py
# Original code written by Carl Ahlberg
# Adapted for the problem at hand by Martin Dahlgren
# PyTorch optimization added for performance improvement
# IMPORTANT! ChatGPT _has_ been used as a coding assistant

import pandas as pd
import numpy as np
import random
import sys
import ast
import streamlit as st
import torch  # Import PyTorch for tensor operations
import os
import json

MAX_FITNESS_CALLS = 205001  # how many fitness calls are allowed
# limit the number of recipes in database. "Food.com" dataset has about 200.000 recipes.
MAX_DATA_ENTRIES = 201000

# we store the fields we are interested in from the database in this class


class Recipe:
    def __init__(self, id, name, calories, protein, fat, carbs, tags):
        self.id = id
        self.name = name
        self.calories = calories
        self.protein = protein
        self.fat = fat
        self.carbs = carbs
        self.tags = tags  # like "vegetarian", "breakfast" etc

# we use a seperate function to load the dataframe so that streamlit can cache it.
# this avoids the database having to reload when something changes that would trigger a streamlit rerun.


@st.cache_data
def load_data(_database_filename) -> pd.DataFrame:
    df = pd.read_csv(_database_filename)
    return df

# GA class is just a collection of functions for a Genetic Algorithm made by Carl Ahlberg.
# Using a class so that they form a logical group to be used elsewhere.


class GA_functions:
    def __init__(self, database_filename, use_pytorch=False, force_gpu=False):
        # keep a reference to filename in case we need it later
        self.filename = database_filename
        self.use_pytorch = use_pytorch  # Flag to enable or disable PyTorch optimizations

        # here we load the database and store it as a dataframe
        self.data = load_data(database_filename)
        # to make it more managable the data is reduced. "Food.com" datasset has about 200.000 recepies.
        # FIXME: make this user configurable from streamlit page.
        self.data = self.data.sample(MAX_DATA_ENTRIES)
        # self.data_sub is a copy that we can edit and alter (we still have access to unedited data in self.data)
        self.data_sub = self.data.copy()
        # here we store the "individual" that ends up with the collection recipes that best match what the user requested.
        self.bestSolution = None

        # PyTorch specific setup

        if self.use_pytorch:
            # Set device to GPU if available; else CPU
            if force_gpu:
                # Force GPU if requested
                self.device = torch.device("cuda")
                print(f"Forcing GPU usage: {self.device}")
            else:
                # Check if CUDA is available
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = torch.device("cpu")
                    print(f"Using device: cpu (GPU not available)")

                    # Print more diagnostic info about why GPU might not be available
                    if hasattr(torch, 'version'):
                        print(f"PyTorch version: {torch.__version__}")
                        if torch.version.cuda:
                            print(f"CUDA version: {torch.version.cuda}")
                        else:
                            print("CUDA not available in this PyTorch build")

                    # Print if CUDA is visible but not being used
                    if not torch.cuda.is_available() and hasattr(torch.cuda, 'is_available'):
                        print(
                            f"CUDA visible but not available: {torch.cuda.is_available()}")

            # After processing the data, load the recipes and also store the nutritional tensor
            self.recipes, self.numRecipes = self.updateDataBase()
            # Create nutrition tensor for faster calculations
            if hasattr(self, 'recipes') and len(self.recipes) > 0:
                nutrient_list = [[r.calories, r.protein, r.fat, r.carbs]
                                 for r in self.recipes]
                self.nutrients = torch.tensor(
                    nutrient_list, dtype=torch.float32, device=self.device)
            else:
                self.nutrients = None
        else:
            # Standard initialization without PyTorch optimization
            self.recipes, self.numRecipes = self.updateDataBase()

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
        if 'nutrition' in self.data_sub.columns:
            # Convert nutrition strings to lists if needed
            if self.data_sub['nutrition'].dtype == 'object':
                try:
                    self.data_sub['nutrition'] = self.data_sub['nutrition'].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except:
                    # If conversion fails, keep as is
                    pass

            # Now extract calories from nutrition
            self.data_sub['calories'] = self.data_sub['nutrition'].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x)

        # Extract protein, fat, and carbs from nutrition if they don't exist
        if 'protein' not in self.data_sub.columns and 'nutrition' in self.data_sub.columns:
            self.data_sub['protein'] = self.data_sub['nutrition'].apply(
                lambda x: x[4] if isinstance(x, (list, tuple)) and len(x) > 4 else 0)  # protein is at index 4

        if 'fat' not in self.data_sub.columns and 'nutrition' in self.data_sub.columns:
            self.data_sub['fat'] = self.data_sub['nutrition'].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else 0)  # fat is at index 1

        if 'carbs' not in self.data_sub.columns and 'nutrition' in self.data_sub.columns:
            self.data_sub['carbs'] = self.data_sub['nutrition'].apply(
                lambda x: x[2] if isinstance(x, (list, tuple)) and len(x) > 2 else 0)  # carbs is at index 2

        # Create ingredients_list if missing - this is required by the app
        if 'ingredients_list' not in self.data_sub.columns and 'ingredients' in self.data_sub.columns:
            # Convert ingredients strings to lists if needed
            if self.data_sub['ingredients'].dtype == 'object':
                try:
                    self.data_sub['ingredients_list'] = self.data_sub['ingredients'].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    self.data_sub['ingredients_list'] = self.data_sub['ingredients_list'].apply(
                        lambda x: [i.strip().lower() for i in x] if isinstance(x, list) else [])
                except:
                    # If conversion fails, create empty list
                    self.data_sub['ingredients_list'] = [[]]

        # a small safety check to see that our database holds all the columns we need.
        # Removed 'ingredients_list' from required columns
        required_columns = ['id', 'name', 'nutrition']
        for column in required_columns:
            if column not in self.data_sub.columns:
                print(
                    f"Error: Missing required column '{column}' in the dataset.")
                sys.exit()

        # loop over our data and create a Recipe object for each.
        recipes = []
        for _, row in self.data_sub.iterrows():
            # Check for null values safely (handling arrays)
            skip_row = False

            # Safe null checking for each required field
            for field in ['nutrition', 'name', 'id']:
                if isinstance(row[field], (list, np.ndarray)):
                    if len(row[field]) == 0:  # Empty array/list
                        skip_row = True
                        break
                elif pd.isna(row[field]):  # Regular null check
                    skip_row = True
                    break

            if skip_row:
                continue  # Skip incomplete data

            # Extract calories, protein, fat, carbs safely
            try:
                calories = float(row['calories']) if not pd.isna(
                    row['calories']) else 0
                protein = float(row['protein']) if not pd.isna(
                    row['protein']) else 0
                fat = float(row['fat']) if not pd.isna(row['fat']) else 0
                carbs = float(row['carbs']) if not pd.isna(row['carbs']) else 0
            except (ValueError, TypeError):
                # If conversion fails, skip this row
                continue

            recipe = Recipe(
                id=int(row['id']),
                name=row['name'],
                calories=calories,
                protein=protein,
                fat=fat,
                carbs=carbs,
                tags=[]  # FIXME: not implemented yet
            )
            recipes.append(recipe)

        numRecipes = len(recipes)
        print(f"Loaded {numRecipes} recipes from {self.filename}.")

        # If using PyTorch, update the nutrients tensor
        if self.use_pytorch and hasattr(self, 'device') and numRecipes > 0:
            nutrient_list = [[r.calories, r.protein, r.fat, r.carbs]
                             for r in recipes]
            self.nutrients = torch.tensor(
                nutrient_list, dtype=torch.float32, device=self.device)

        return recipes, numRecipes

    # let the user make changes to our data, ie filter by ingredients etc..
    def setSubSelection(self, selection: pd.DataFrame):
        self.data_sub = selection
        self.recipes, self.numRecipes = self.updateDataBase()
        return

    def getSubSelection(self) -> pd.DataFrame:
        return self.data_sub

    def save_to_nosql(self, db_name="filtered_recipes"):
        """
        Saves the filtered dataset to a JSON file in the data folder.

        Parameters:
        - db_name (str): Base name for the database file (without extension)

        Returns:
        - str: Path to the saved database
        """
        # Ensure data folder exists
        data_folder = os.path.dirname(self.filename)
        db_path = os.path.join(data_folder, f"{db_name}.json")

        # Convert DataFrame to records
        df_to_save = self.data_sub.copy()

        # Handle non-serializable objects (lists, dicts, etc.)
        for col in df_to_save.columns:
            if df_to_save[col].apply(lambda x: isinstance(x, (list, dict))).any():
                # Convert lists/dicts to JSON strings
                df_to_save[col] = df_to_save[col].apply(
                    lambda x: json.dumps(x) if isinstance(
                        x, (list, dict)) else x
                )

        # Convert to records
        records = df_to_save.to_dict('records')

        # Save to JSON file
        try:
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(records)} filtered recipes to {db_path}")
            st.success(f"Saved {len(records)} filtered recipes to database")
            return db_path

        except Exception as e:
            print(f"Error saving database: {str(e)}")
            st.error(f"Error saving database: {str(e)}")
            return None

    def load_from_nosql(self, db_name="filtered_recipes"):
        """
        Loads data from a JSON file in the data folder.

        Parameters:
        - db_name (str): Base name for the database file (without extension)

        Returns:
        - pd.DataFrame: DataFrame with the loaded data
        """
        data_folder = os.path.dirname(self.filename)
        db_path = os.path.join(data_folder, f"{db_name}.json")

        if not os.path.exists(db_path):
            print(f"Database {db_path} does not exist")
            return None

        try:
            # Load JSON data
            with open(db_path, 'r', encoding='utf-8') as f:
                records = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(records)

            # Convert serialized JSON back to Python objects
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Try to convert JSON strings back to Python objects
                        df[col] = df[col].apply(
                            lambda x: json.loads(x) if isinstance(x, str) and
                            (x.startswith('[') and x.endswith(']')) or
                            (x.startswith('{') and x.endswith('}'))
                            else x
                        )
                    except:
                        pass  # If conversion fails, keep as is

            print(f"Loaded {len(records)} recipes from {db_path}")
            return df

        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return None

    # check that the parameters to the genetic algorithm are reasonable
    def checkErrorsInParameters(self, maxGeneration, populationSize, numNewOffspring, mutationProbability, numberMutations, tournamentSize):
        error = 0
        numFitnessCalls = maxGeneration * numNewOffspring + populationSize
        if numFitnessCalls > MAX_FITNESS_CALLS:
            print(
                f'You are calculating the fitness {numFitnessCalls} times. The maximum is {MAX_FITNESS_CALLS}.')
            error = 1
        if maxGeneration < 10:
            print(
                f'You set \'maxGeneration\' to {maxGeneration}. The minimum is 10.')
            error = 1
        if populationSize < 5:
            print(
                f'You set \'populationSize\' to {populationSize}. The minimum is 5.')
            error = 1
        if numNewOffspring < 1:
            print(
                f'You set \'numNewOffspring\' to {numNewOffspring}. The minimum is 1.')
            error = 1
        if numberMutations < 1:
            print(
                f'You set \'numberMutations\' to {numberMutations}. The minimum is 1.')
            error = 1
        if not (0 <= mutationProbability <= 1):
            print(
                f'You set \'mutationProbability\' to {mutationProbability}. The correct range is [0, 1].')
            error = 1
        if not (2 <= tournamentSize <= populationSize):
            print(
                f'You set \'tournamentSize\' to {tournamentSize}. The correct range is [2, {populationSize}].')
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
        - population: List of numpy arrays or PyTorch tensor based on optimization setting.
        """
        if self.use_pytorch:
            # PyTorch optimized version
            population = torch.zeros(
                (populationSize, numRecipes), dtype=torch.int32, device=self.device)
            for i in range(populationSize):
                # Randomly select maxRecipes indices from 0 to numRecipes-1 without replacement
                indices = torch.randperm(numRecipes, device=self.device)[
                    :maxRecipes]
                population[i, indices] = 1
            return population
        else:
            # Original numpy version
            population = []
            for _ in range(populationSize):
                individual = np.zeros(numRecipes, dtype=int)
                selected_indices = np.random.choice(
                    numRecipes, size=maxRecipes, replace=False)
                individual[selected_indices] = 1
                population.append(individual)
            return population

    def calculateFitness(self, population, recipes, targetCalories, targetProteins, targetFat, targetCarbs,
                         dietaryRestrictions=[], lowerBound=None, upperBound=None, maxRecipes=12):
        """
        Calculates the fitness of each individual in the population.

        Parameters:
        - population: Either a PyTorch tensor or list of numpy arrays based on optimization setting.
        - recipes (list of Recipe): List of all available recipes.
        - targetCalories (float): Desired total calorie intake.
        - targetProteins (float): Desired protein intake.
        - targetFat (float): Desired fat intake.
        - targetCarbs (float): Desired carbs intake.
        - dietaryRestrictions (list of str): List of dietary restrictions (e.g., ['vegetarian']).
        - lowerBound (float): Minimum acceptable total calories.
        - upperBound (float): Maximum acceptable total calories.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - populationFitness: Fitness scores for each individual.
        - calorieDiffs: Signed difference between total calories and target calories for each individual.
        """
        # Check if population is a PyTorch tensor or a list
        is_pytorch_tensor = self.use_pytorch and hasattr(population, 'float')

        if is_pytorch_tensor:
            # PyTorch optimized version - vectorized operations
            # Convert binary population to float and do matrix multiplication with nutrients tensor
            totals = population.float() @ self.nutrients  # shape: (popSize, 4)
            totalCalories = totals[:, 0]
            totalProtein = totals[:, 1]
            totalFat = totals[:, 2]
            totalCarbs = totals[:, 3]

            # Calculate fitness components
            calorie_diff = totalCalories - targetCalories
            fitness = calorie_diff.abs()

            # Apply penalties
            if lowerBound is not None:
                fitness += torch.clamp(lowerBound -
                                       totalCalories, min=0.0) * 10
            if upperBound is not None:
                fitness += torch.clamp(totalCalories -
                                       upperBound, min=0.0) * 10

            # Penalty for exceeding maxRecipes
            # count of ones per individual
            num_selected = population.sum(dim=1)
            fitness += torch.clamp(num_selected - maxRecipes, min=0) * 100

            # Penalties for macronutrient targets
            protein_diff = (targetProteins - totalProtein).abs() * 100
            fat_diff = (targetFat - totalFat).abs() * 100
            carbs_diff = (targetCarbs - totalCarbs).abs() * 100
            fitness += protein_diff + fat_diff + carbs_diff

            return fitness, calorie_diff
        else:
            # When offsprings are passed as a list (which happens in Run_Plan.py), convert them to tensor
            if self.use_pytorch and isinstance(population, list):
                # Convert list of numpy arrays to PyTorch tensor
                try:
                    population_tensor = torch.stack([
                        torch.as_tensor(ind, dtype=torch.int32,
                                        device=self.device)
                        for ind in population
                    ])
                    # Now use the PyTorch version with the converted tensor
                    return self.calculateFitness(population_tensor, recipes, targetCalories, targetProteins,
                                                 targetFat, targetCarbs, dietaryRestrictions, lowerBound,
                                                 upperBound, maxRecipes)
                except:
                    # If conversion fails for any reason, fall back to numpy version
                    pass

            # Original numpy version
            populationFitness = []
            calorieDiffs = []

            for individual in population:
                totalCalories = 0.0
                totalProtein = 0.0
                totalFat = 0.0
                totalCarbs = 0.0
                valid = True

                for gene, recipe in zip(individual, recipes):
                    if gene:
                        totalCalories += recipe.calories
                        totalProtein += recipe.protein
                        totalFat += recipe.fat
                        totalCarbs += recipe.carbs

                numSelectedRecipes = np.sum(individual)

                if not valid:
                    fitness = 1e6
                    calorie_diff = 1e6
                else:
                    calorie_diff = totalCalories - targetCalories
                    fitness = abs(calorie_diff)

                    # penalty for violating calorie bounds
                    if lowerBound and totalCalories < lowerBound:
                        fitness += (lowerBound - totalCalories) * 10
                    if upperBound and totalCalories > upperBound:
                        fitness += (totalCalories - upperBound) * 10

                    # penalize for exceeding maxRecipes
                    if numSelectedRecipes > maxRecipes:
                        fitness += abs(numSelectedRecipes - maxRecipes) * 100

                    # penalize for missing protein, fat and carb targets
                    protein_diff = abs(targetProteins - totalProtein) * 100
                    fat_diff = abs(targetFat - totalFat) * 100
                    carbs_diff = abs(targetCarbs - totalCarbs) * 100
                    fitness += protein_diff + fat_diff + carbs_diff

                populationFitness.append(fitness)
                calorieDiffs.append(calorie_diff)

            return populationFitness, calorieDiffs

    def parentSelectionTournament(self, population, fitness, tournamentSize=3):
        """
        Selects two parents using tournament selection.

        Parameters:
        - population: Either a PyTorch tensor or list of numpy arrays.
        - fitness: List of fitness scores or PyTorch tensor.
        - tournamentSize (int): Number of individuals competing in each tournament.

        Returns:
        - parents: Two selected parents.
        """
        # Check if we're using PyTorch tensor or list/numpy array
        is_pytorch_tensor = self.use_pytorch and hasattr(population, 'size')

        if is_pytorch_tensor:
            # PyTorch optimized version
            popSize = population.size(0)
            parents = []
            for _ in range(2):
                indices = torch.randperm(popSize, device=self.device)[
                    :tournamentSize]
                tournamentFitness = fitness[indices]
                winner = indices[torch.argmin(tournamentFitness)]
                parents.append(population[winner].clone())
            return torch.stack(parents)  # shape: (2, numRecipes)
        else:
            # Convert lists to PyTorch tensors if PyTorch is enabled but input is a list
            if self.use_pytorch and isinstance(population, list) and isinstance(fitness, list):
                try:
                    pop_tensor = torch.stack([torch.tensor(ind, dtype=torch.int32, device=self.device)
                                              for ind in population])
                    fit_tensor = torch.tensor(
                        fitness, dtype=torch.float32, device=self.device)
                    return self.parentSelectionTournament(pop_tensor, fit_tensor, tournamentSize)
                except:
                    # Fall back to numpy version if conversion fails
                    pass

            # Original numpy version
            parents = []
            for _ in range(2):
                tournament_indices = random.sample(
                    range(len(population)), tournamentSize)
                tournament_fitness = [fitness[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmin(
                    tournament_fitness)]
                parents.append(population[winner_index].copy())
            return parents

    def crossover(self, parents, maxRecipes=12):
        """
        Performs single-point crossover between two parents.

        Parameters:
        - parents: Two parent individuals (PyTorch tensor or list of numpy arrays).
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - Two offspring individuals after crossover.
        """
        # Check if we're using PyTorch tensor or list/numpy array
        is_pytorch_tensor = self.use_pytorch and hasattr(parents, 'size')

        if is_pytorch_tensor:
            # PyTorch optimized version
            numGenes = parents.size(1)
            crossover_point = random.randint(1, numGenes - 1)
            offspring1 = torch.cat(
                [parents[0, :crossover_point], parents[1, crossover_point:]])
            offspring2 = torch.cat(
                [parents[1, :crossover_point], parents[0, crossover_point:]])

            # Ensure offspring do not exceed maxRecipes
            for offspring in [offspring1, offspring2]:
                if offspring.sum() > maxRecipes:
                    excess = int(offspring.sum().item() - maxRecipes)
                    ones_indices = (offspring == 1).nonzero(as_tuple=True)[0]
                    perm = torch.randperm(len(ones_indices))
                    indices_to_zero = ones_indices[perm[:excess]]
                    offspring[indices_to_zero] = 0

            return offspring1, offspring2
        else:
            # Convert list/numpy to PyTorch if PyTorch is enabled
            if self.use_pytorch and isinstance(parents, list):
                try:
                    # Try to convert parents to PyTorch tensors
                    parents_tensor = torch.stack([torch.tensor(parent, dtype=torch.int32, device=self.device)
                                                 for parent in parents])
                    return self.crossover(parents_tensor, maxRecipes)
                except:
                    # Fall back to numpy version if conversion fails
                    pass

            # Original numpy version
            crossover_point = random.randint(1, len(parents[0]) - 1)
            offspring1 = np.concatenate(
                [parents[0][:crossover_point], parents[1][crossover_point:]])
            offspring2 = np.concatenate(
                [parents[1][:crossover_point], parents[0][crossover_point:]])

            # Ensure offspring do not exceed maxRecipes
            for offspring in [offspring1, offspring2]:
                num_selected = np.sum(offspring)
                if num_selected > maxRecipes:
                    excess = int(num_selected - maxRecipes)
                    selected_indices = np.where(offspring == 1)[0]
                    deselect_indices = np.random.choice(
                        selected_indices, size=excess, replace=False)
                    offspring[deselect_indices] = 0

            return offspring1, offspring2

    def mutation(self, individual, mutationProbability, numMutations, maxRecipes=12):
        """
        Applies bit-flip mutation to an individual.

        Parameters:
        - individual: The individual to mutate (PyTorch tensor or numpy array).
        - mutationProbability (float): Probability of each mutation.
        - numMutations (int): Number of mutations to apply.
        - maxRecipes (int): Maximum number of recipes per individual.

        Returns:
        - The mutated individual.
        """
        if self.use_pytorch:
            # PyTorch optimized version
            numGenes = individual.size(0)
            # We still use a Python loop for the low number of mutations
            for _ in range(numMutations):
                if random.random() < mutationProbability:
                    gene_idx = random.randint(0, numGenes - 1)
                    if individual[gene_idx] == 1:
                        individual[gene_idx] = 0
                    else:
                        # Ensure not to exceed maxRecipes
                        if individual.sum() < maxRecipes:
                            individual[gene_idx] = 1
            return individual
        else:
            # Original numpy version
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

    def batch_mutation(self, population, mutationProbability, numMutations, maxRecipes=12):
        """
        Applies mutations to the entire population in a vectorized manner.

        Parameters:
        - population: Tensor of shape (popSize, numGenes) representing the population
        - mutationProbability: Probability of each mutation
        - numMutations: Maximum number of mutations per individual
        - maxRecipes: Maximum number of recipes per individual

        Returns:
        - Mutated population tensor
        """
        if not self.use_pytorch or not torch.is_tensor(population):
            # Fall back to sequential mutation if not using PyTorch
            return torch.stack([self.mutation(ind, mutationProbability, numMutations, maxRecipes)
                                for ind in population])

        # Get population dimensions
        popSize, numGenes = population.shape

        # Create a copy of the population to modify
        mutated_population = population.clone()

        # Generate random mutation masks for each potential mutation
        for _ in range(numMutations):
            # Generate mutation probability mask (1 where mutation occurs, 0 elsewhere)
            mutation_mask = torch.rand(
                popSize, device=self.device) < mutationProbability

            if not torch.any(mutation_mask):
                continue  # Skip if no mutations in this round

            # For each individual that will be mutated, select a random gene position
            gene_positions = torch.randint(
                0, numGenes, (popSize,), device=self.device)

            # Process only individuals marked for mutation in this round
            for idx in torch.where(mutation_mask)[0]:
                gene_pos = gene_positions[idx]

                if mutated_population[idx, gene_pos] == 1:
                    # Turn off the gene
                    mutated_population[idx, gene_pos] = 0
                else:
                    # Turn on the gene only if we haven't reached maxRecipes
                    if torch.sum(mutated_population[idx]) < maxRecipes:
                        mutated_population[idx, gene_pos] = 1

        return mutated_population

    def updatePopulation(self, population, fitness, offsprings, offsprings_fitness):
        """
        Updates the population by selecting the best individuals from the combined pool.

        Parameters:
        - population: Current population (PyTorch tensor or list of numpy arrays).
        - fitness: Fitness scores of current population.
        - offsprings: New offspring.
        - offsprings_fitness: Fitness scores of offspring.

        Returns:
        - Updated population and fitness scores.
        """
        # Check if we're using PyTorch tensor or list/numpy array
        is_pytorch_tensor = self.use_pytorch and hasattr(population, 'size')

        if is_pytorch_tensor:
            # PyTorch optimized version - ensure offsprings are also tensors
            try:
                # Convert offsprings to tensor if they are lists
                if isinstance(offsprings, list):
                    offsprings_tensor = torch.stack([
                        torch.as_tensor(
                            offspring, dtype=torch.int32, device=self.device)
                        for offspring in offsprings
                    ])
                else:
                    offsprings_tensor = offsprings

                # Convert fitness values to tensors if they are lists
                if isinstance(offsprings_fitness, list):
                    offsprings_fitness_tensor = torch.as_tensor(
                        offsprings_fitness, dtype=torch.float32, device=self.device)
                else:
                    offsprings_fitness_tensor = offsprings_fitness

                if isinstance(fitness, list):
                    fitness_tensor = torch.as_tensor(
                        fitness, dtype=torch.float32, device=self.device)
                else:
                    fitness_tensor = fitness

                # Now concatenate tensors
                combined_population = torch.cat(
                    (population, offsprings_tensor), dim=0)
                combined_fitness = torch.cat(
                    (fitness_tensor, offsprings_fitness_tensor), dim=0)

                # Sort by fitness (ascending order: lower fitness is better)
                sorted_indices = torch.argsort(combined_fitness)
                new_population = combined_population[sorted_indices][:population.size(
                    0)]
                new_fitness = combined_fitness[sorted_indices][:population.size(
                    0)]
                return new_population, new_fitness
            except Exception as e:
                # If there's any error converting to tensor or combining, fall back to numpy version
                print(
                    f"Warning: Falling back to numpy implementation due to: {e}")
                is_pytorch_tensor = False

        # Original numpy version or fallback
        if not is_pytorch_tensor:
            # Convert PyTorch tensors to numpy arrays if needed
            if hasattr(population, 'cpu') and hasattr(population, 'numpy'):
                population = population.cpu().numpy()
            if hasattr(fitness, 'cpu') and hasattr(fitness, 'numpy'):
                fitness = fitness.cpu().numpy()

            # Convert to lists if they're numpy arrays
            if isinstance(population, np.ndarray):
                population = [p for p in population]
            if isinstance(fitness, np.ndarray):
                fitness = fitness.tolist()

            combined = list(zip(population, fitness)) + \
                list(zip(offsprings, offsprings_fitness))
            # Sort by fitness (lower is better)
            combined.sort(key=lambda x: x[1])
            # Select the top individuals to form the new population
            new_population = [ind for ind, fit in combined[:len(population)]]
            new_fitness = [fit for ind, fit in combined[:len(population)]]
            return new_population, new_fitness

    def visualization(self, population, fitness, recipes, g, maxGeneration, showBestSolution=True,
                      showPopulationDistribution=False, bestFitness=float('inf'), targetCalories=0, silent=False):
        """
        Visualizes the GA progress by printing the best meal plan and optionally plotting fitness distributions.

        Parameters:
        - population: Current population (PyTorch tensor or list of numpy arrays).
        - fitness: Fitness scores.
        - recipes (list of Recipe): List of all available recipes.
        - g (int): Current generation.
        - maxGeneration (int): Total number of generations.
        - showBestSolution (bool): Flag to show the best solution.
        - showPopulationDistribution (bool): Flag to show population fitness distribution.
        - bestFitness (float): Best fitness found so far.
        - targetCalories (float): Desired calorie intake.
        - silent (bool): If True, suppresses detailed printing for better performance.

        Returns:
        - bestFitness (float): Updated best fitness.
        """
        if self.use_pytorch and hasattr(population, 'cpu'):
            # PyTorch optimized version - transfer to CPU for display
            fitness_cpu = fitness.cpu().numpy()
            currentBest = np.min(fitness_cpu)
            indexBestSolution = np.argmin(fitness_cpu)

            if currentBest < bestFitness or ((g+1) == maxGeneration):
                if not silent:
                    print(f'\nGeneration {g+1}/{maxGeneration}')
                    print(
                        f'New Best Fitness: {currentBest} calorie difference')
                bestFitness = currentBest

                if showBestSolution or ((g+1) == maxGeneration):
                    # Store the best solution for later retrieval
                    bestSolution = population[indexBestSolution].cpu().numpy()
                    self.bestSolution = bestSolution

                    # Only print detailed recipe information if not in silent mode
                    if not silent:
                        selectedRecipes = [recipes[i]
                                           for i, gene in enumerate(bestSolution) if gene]
                        totalCalories = sum(
                            recipe.calories for recipe in selectedRecipes)

                        print("\nBest Meal Plan:")
                        for recipe in selectedRecipes:
                            print(f"- {recipe.name} ({recipe.calories} kcal)")
                        print(
                            f"Total Calories: {totalCalories} kcal (Target: {targetCalories} kcal)")

                        # Debugging: Print individual calories and verify
                        for recipe in selectedRecipes:
                            print(
                                f"- {recipe.name} ({recipe.calories} kcal)  ({recipe.protein} g)  ({recipe.fat} g)  ({recipe.carbs} g)")
        else:
            # Original numpy version
            currentBest = min(fitness)
            indexBestSolution = fitness.index(currentBest)

            if currentBest < bestFitness or ((g+1) == maxGeneration):
                if not silent:
                    print(f'\nGeneration {g+1}/{maxGeneration}')
                    print(
                        f'New Best Fitness: {currentBest} calorie difference')
                bestFitness = currentBest

                if showBestSolution or ((g+1) == maxGeneration):
                    self.bestSolution = population[indexBestSolution]

                    # Only print detailed recipe information if not in silent mode
                    if not silent:
                        selectedRecipes = [recipes[i] for i, gene in enumerate(
                            self.bestSolution) if gene]
                        totalCalories = sum(
                            recipe.calories for recipe in selectedRecipes)

                        print("\nBest Meal Plan:")
                        for recipe in selectedRecipes:
                            print(f"- {recipe.name} ({recipe.calories} kcal)")
                        print(
                            f"Total Calories: {totalCalories} kcal (Target: {targetCalories} kcal)")

                        # Debugging: Print individual calories and verify
                        for recipe in selectedRecipes:
                            print(
                                f"- {recipe.name} ({recipe.calories} kcal)  ({recipe.protein} g)  ({recipe.fat} g)  ({recipe.carbs} g)")

        return bestFitness

    def getBestSolution(self):
        """
        Returns the best solution found by the GA.

        Returns:
        - bestSolution: The best individual found by the GA.
        """
        return self.bestSolution
