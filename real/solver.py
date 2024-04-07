import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.signal import convolve2d
from functools import reduce
from constraint import Problem, ExactSumConstraint, MaxSumConstraint
from operator import mul
from math import factorial
from itertools import product

from setting import *

# Helper functions
def dilate(bool_arr):
	# Perform binary dilation with a structuring element with connectivity 2.
	return binary_dilation(bool_arr, structure=generate_binary_structure(2, 2))

def neighbors(bool_arr):
	# Return a binary mask marking all squares that neighbor a True cells in the boolean array.
	return bool_arr ^ dilate(bool_arr)

def mask_xy(x, y, shape):
	# Create a binary mask that marks only the square at (x, y).
	mask = np.zeros(shape, dtype=bool)
	mask[x, y] = 1
	return mask

def neighbors_xy(x, y, shape):
	# Return a binary mask marking all squares that neighbor the square at (x, y).
	return neighbors(mask_xy(x, y, shape))

def boundary(state):
	# Return a binary mask marking all closed squares that are adjacent to a number.
	return neighbors(~np.isnan(state))

def count_active_neighbors(bool_arr):
	# Calculate how many True's there are next to a square.
	filter = np.ones((3, 3))
	filter[1, 1] = 0
	return convolve2d(bool_arr, filter, mode='same')

def subtract_known(state, mines=None):
	# Reduce the numbers in the state to represent the number of mines next to it that have not been found yet.
	# mines: The mines to use to reduce numbers.
	no_known_adj_mines = count_active_neighbors(mines)
	state[~np.isnan(state)] -= no_known_adj_mines[~np.isnan(state)]
	return state

def draw_game(state):
	s = ""
	for i in range(state.shape[0]):
		for j in range(state.shape[1]):
			v = state[i, j]
			if v > 0:
				s += f'{int(v)} '
			elif v == 0:
				s += '  '
			elif v == CELL_COVERED:
				s += '- '
			elif v == CELL_FLAG:
				s += 'f '
			else:
				s += 'x '
		s += '\n'
	s += "____________________________________________________________________________________________\n"
	return s

class MinesweeperSolver:
	def __init__(self, height, width, total_mines):
		self._total_mines = total_mines
		# Known mines are 1, known safe squares are 0, uncertain squares are np.nan.
		self.known = np.full((height, width), np.nan)

	def mines_known(self):
		# Returns how many mines we know the location of.
		return (self.known == 1).sum(dtype=int)

	def mines_left(self):
		# Returns the number of mines that we don't know the location of yet.
		return self._total_mines - self.mines_known()

	def solve(self, state):
		# Compute the probability of there being a mine under a given square.
		# state: A 2D nested list representing the minesweeper state.
		# Return an array giving the probability that each unopened square contains a mine.
		# Opened squares or partially computed ones are returned as np.nan.

		# Convert to an easier format to solve: only the numbers remain, the rest is np.nan.
		state = np.array([[state[x, y] if state[x, y] >= 0 else np.nan
			for y in range(state.shape[1])] for x in range(state.shape[0])])
		'''
		print(state)
		print('\n')
		print(self.known)
		print('\n')
		'''
		print(draw_game(state))
		# If not the first move
		if not np.isnan(state).all():
			# Expand the known state with new information from the passed state.
			self.known[~np.isnan(state)] = 0
			# Reduce the numbers of known adjacent mines from each number and find any trivial solutions.
			prob, state = self._gac(state)
			# Stop if we've found a safe square to open.
			if not np.isnan(prob).all() and 0 in prob:
				return prob
			# Compute the probabilities for the remaining, uncertain squares.
			prob = self._csp(state, prob)
			return prob
		else:
			# If no cells are opened yet, just give each cell the same probability.
			return np.full(state.shape, self._total_mines / state.size)
			
	def _gac(self, state):
		# Apply arc consistency on the sum constraints to deduce that all unopened neighbors are mines/safe.
		# Return an array with known mines as 1, safe squares as 0 and everything else as np.nan.
		result = np.full(state.shape, np.nan)
		# This step can be done multiple times, as each time we deduce mines / safe, the numbers can be further reduced.
		new_results = True
		# Subtract all numbers by the amount of neighboring mines we've already found, simplifying the game.
		state = subtract_known(state, self.known == 1)
		# Calculate the unknown square: unopened and we've not previously found their value through deduction.
		unknown_squares = np.isnan(state) & np.isnan(self.known)
		while new_results:
			no_unknown_neighbors = count_active_neighbors(unknown_squares)
			# Find squares with the number N > 0 in it and has N unopened neighbors -> all mines
			solutions = (state > 0) & (state == no_unknown_neighbors)
			# Create a mask for all those squares that we now know are mines.
			known_mines = unknown_squares & reduce(np.logical_or,
				[neighbors_xy(x, y, state.shape) for x, y in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
			# Update our known matrix with these new finding: 1 for mines.
			self.known[known_mines] = 1
			# Further reduce the numbers, since we found new mines.
			state = subtract_known(state, known_mines)
			# Update what is unknown.
			unknown_squares &= ~known_mines
			no_unknown_neighbors = count_active_neighbors(unknown_squares)

			# Find squares with a 0 and any unopened neighbors -> all safe.
			solutions = (state == 0) & (no_unknown_neighbors > 0)
			# Create a mask for all those squares that we now know are safe.
			known_safe = unknown_squares & reduce(np.logical_or,
				[neighbors_xy(x, y, state.shape) for x, y in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
			# Update our known matrix with these new finding: 0 for safe squares.
			self.known[known_safe] = 0
			# Update what is unknown.
			unknown_squares &= ~known_safe

			# Update the result matrix for both steps, 0 for safe squares, 1 for mines.
			result[known_safe] = 0
			result[known_mines] = 1
			new_results = (known_safe | known_mines).any()
		return result, state

	def _csp(self, state, prob):
		def _components(state):
			# Mark all the numbers next to unknown islands.
			numbers_mask = dilate(np.isnan(state) & np.isnan(self.known)) & ~np.isnan(state)
			# Mark all unknown cells next to these numbers.
			boundary_mask = dilate(numbers_mask) & (np.isnan(state) & np.isnan(self.known))
			# Find and number horizontally, vertically or diagonally connected components of these border cells.
			labeled, no_components = label(boundary_mask, structure=generate_binary_structure(2, 2))
			# Connect two components that have some numbers in between them.
			i = 1
			while i < no_components:
				j = i + 1
				while j <= no_components:
					# If there are some numbers connecting the two components...
					if not np.isnan(state[dilate(labeled == i) & dilate(labeled == j)]).all():
						# Merge the components.
						labeled[labeled == j] = i
						labeled[labeled > j] -= 1
						no_components -= 1
						i -= 1
						break
					j += 1
				i += 1
			return labeled, no_components

		def _areas(state, comp):
			# Split the component into areas, in which each square is constrained by the same constraints
			# Return a mapping of constraints to an n-tuple of squares they apply to.
			# Return a list of all constraints applicable in the component.

			# Find all numbers that are constraints for the given component.
			constraints_mask = neighbors(comp) & ~np.isnan(state)
			# Generate a list of all CP constraints corresponding to numbers.
			constraint_list = [ExactSumConstraint(int(num)) for num in state[constraints_mask]]
			# Create an array where these constraints are placed in the corresponding squares.
			constraints = np.full(state.shape, None, dtype=object)
			constraints[constraints_mask] = constraint_list

			# Create an array where a list of applicable constraints is stored for each squares in the component.
			applied_constraints = np.empty(state.shape, dtype=object)
			for x, y in zip(*comp.nonzero()):
				applied_constraints[x, y] = []
			for xi, yi in zip(*constraints_mask.nonzero()):
				targets = neighbors_xy(xi, yi, comp.shape) & comp
				for xj, yj in zip(*targets.nonzero()):
					applied_constraints[xj, yj].append(constraints[xi, yi])

			# Group all squares that have the same constraints applied to them, each one being an area.
			mapping = {}
			for x, y in zip(*comp.nonzero()):
				k = tuple(applied_constraints[x, y]) # Convert to tuple, so we can use it as a hash key.
				if k not in mapping:
					mapping[k] = []
				mapping[k].append((x, y))
			# Turn the list of (x, y) tuples into a tuple, which allows them to be used as hash keys.
			mapping = {k: tuple(v) for k, v in mapping.items()}
			return mapping, constraint_list

		def combinations(n, m):
			# Calculate the number of ways to distribute m mines in n squares.
			return factorial(n)/(factorial(n-m)*factorial(m))


		def _count_models(solution):
			# Count how many models are possible for a solution of the component areas.
			# solution: {area_key: number_or_mines}
			# Return the number of ways the component's areas can be filled to match the solution.
			return reduce(mul, [combinations(len(area), mines_per_area) for area, mines_per_area in solution.items()])

		def _relative_weight(range_no_mines_comps, no_unconstrained):
			# Calculate the relative number of models for unconstrained squares if all components have no_mines_comps mines
			no_mines_left = self.mines_left()
			if no_unconstrained == 0:
				return {no_mines_left: 1}
			rel_no_unconstrained_models = {}
			no_mines_unconstrained = no_mines_left - range_no_mines_comps[0]
			# If there are no_mines_unconstrained mines for no_unconstrained squares,
			# then number of models is C(no_unconstrained, no_mines_unconstrained)
			# For m < n: C(n, m) = C(n, m - 1) * (n - m) / m
			# For n = m: C(n, n) = C(n, n - 1) / n
			weight = 1
			for no_mines_comps in range_no_mines_comps:
				no_mines_unconstrained_next = self.mines_left() - no_mines_comps
				for no_mines_unconstrained in range(no_mines_unconstrained + 1, no_mines_unconstrained_next + 1):
					# Special case: m == n, due to factorial in the derivation.
					if no_unconstrained == no_mines_unconstrained:
						weight /= no_unconstrained
					else:
						weight = weight * (no_unconstrained - no_mines_unconstrained) / no_mines_unconstrained
				rel_no_unconstrained_models[no_mines_comps] = weight
				if no_mines_unconstrained >= no_unconstrained:
					break
			return rel_no_unconstrained_models

		def _combine_components(state, prob, no_models_per_comp, comp_prob_per_comp):
			# Combine the probabilities and model counts into one probability array.
			# state: The reduced state.
			# prob: The already computed probabilities.
			# no_models_per_comp: A list of model count mappings per component, each having the format {no_mines: no_model}
			# comp_prob_per_comp: A list of probability mappings per component, each having the format {no_mines: prob}
			# Return the exact probability for every unknown square.

			# Find the unconstrained squares
			solution_mask = boundary(state) & np.isnan(self.known)
			unconstrained_squares = np.isnan(state) & ~solution_mask & np.isnan(self.known)
			n = unconstrained_squares.sum(dtype=int)
			# It's possible there aren't any components at all
			if comp_prob_per_comp:
				min_no_mines = sum([min(no_mines) for no_mines in comp_prob_per_comp])
				max_no_mines = sum([max(no_mines) for no_mines in comp_prob_per_comp])
				no_mines_left = self.mines_left()
				rel_no_unconstrained_models = _relative_weight(range(min(max_no_mines, no_mines_left), min_no_mines - 1, -1), n)
				total_no_models = 0
				total_prob = np.zeros(prob.shape)
				# Iterate over all combinations of the components.
				for mines_combination in product(*[comp.keys() for comp in no_models_per_comp]):
					no_mines_comps = sum(mines_combination)
					if no_mines_left - n <= no_mines_comps <= no_mines_left:
						# Combine the prob arrays for this component combination.
						combined_prob = reduce(np.add, [comp_prob_per_comp[comp][no_mines_comps]
							for comp, no_mines_comps in enumerate(mines_combination)])
						combined_no_models = reduce(mul, [no_models_per_comp[comp][no_mines_comps]
							for comp, no_mines_comps in enumerate(mines_combination)])
						rel_no_total_models = rel_no_unconstrained_models[no_mines_comps] * combined_no_models
						total_no_models += rel_no_total_models
						total_prob += rel_no_total_models * combined_prob
				# Normalize the probabilities by dividing out the total number of models.
				total_prob /= total_no_models
				# Add result to the prob array.
				prob[solution_mask] = total_prob[solution_mask]
			# If there are any unconstrained mines...
			if n > 0:
				# The amount of remaining mines is distributed evenly over the unconstrained squares.
				prob[unconstrained_squares] = (self.mines_left() - prob[~np.isnan(prob) & np.isnan(self.known)].sum()) / n
			# Remember the certain values.
			certain_mask = np.isnan(self.known) & ((prob == 0) | (prob == 1))
			self.known[certain_mask] = prob[certain_mask]
			return prob

		components, no_components = _components(state)
		'''
		print(components)
		print('\n')
		'''
		no_models_per_comp = []
		comp_prob_per_comp = []
		for c in range(1, no_components + 1):
			areas, constraints = _areas(state, components == c)
			# Create a CP problem to determine which combination of mines per area is possible.
			problem = Problem()
			# Add all variables, each one having a domain [0, no squares in tuple]
			for v in areas.values():
				problem.addVariable(v, range(len(v)+1))
			# Add all constraints
			for constraint in constraints:
				problem.addConstraint(constraint, [v for k, v in areas.items() if constraint in k])
			# Add a constraint so that the maximum number of mines never exceeds the number of mines left.
			problem.addConstraint(MaxSumConstraint(self.mines_left()), list(areas.values()))
			solutions = problem.getSolutions()

			# For each number of mines m in the component,
			# count the number of models that exist and the probability of each square having a mine.
			no_models_by_m = {}
			comp_prob_by_m = {}
			for solution in solutions:
				no_mines = sum(solution.values())
				# Number of models that match this solution.
				no_models = _count_models(solution)
				# Increase the counter for the number of models that have m mines.
				no_models_by_m[no_mines] = no_models_by_m.get(no_mines, 0) + no_models
				# Calculate the probability of each square in the component having a mine.
				comp_prob = np.zeros(prob.shape)
				for area, mines_per_area in solution.items():
					comp_prob[tuple(zip(*area))] = mines_per_area / len(area)
				# Sum up all the models, giving the expected number of mines of all models combined.
				comp_prob_by_m[no_mines] = comp_prob_by_m.get(no_mines, np.zeros(prob.shape)) + no_models * comp_prob

			comp_prob_by_m = {no_mines: comp_prob/no_models_by_m[no_mines] for no_mines, comp_prob in comp_prob_by_m.items()}
			no_models_per_comp.append(no_models_by_m)
			comp_prob_per_comp.append(comp_prob_by_m)

		return _combine_components(state, prob, no_models_per_comp, comp_prob_per_comp)
		'''
			print(f"Component {c}:")
			for k, v in areas.items():
				for c in k:
					print(c._exactsum, end=" ")
				print(f": {v}")
		print('\n')
		
		r = int(input("Enter row: "))
		c = int(input("Enter col: "))
		b = int(input("Enter button: "))
		result = np.full(state.shape, np.nan)
		result[r, c] = b
		self.known[r, c] = b
		return result
		'''
