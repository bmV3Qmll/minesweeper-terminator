import pyautogui
import pyscreeze
import numpy as np
from PIL import ImageDraw

from setting import *
from solver import *

pyautogui.FAILSAFE = False
np.set_printoptions(precision=3)

class MinesweeperAutomation:
	# Class to play minesweeper from pixels: find the game on the screen, read the cells' values, click and so on
	def __init__(self, settings=SETTINGS_MINESWEEPER_CLASSIC, mines=None, ignoreFlag=True, region=None, speed=0.1):
		# Bot settings, which colors are used to find and read the field
		self.settings = settings
		# The shape of the field
		self.game_shape = None
		# Number of mines in a game
		self.game_mines = mines
		# Coordinates of the game on the screen
		self.cells_coordinates = None
		# Placeholder for the solver
		self.solver = None
		# Cell recognition cache
		self.cell_cache = {}
		# Don't use flags
		self.ignoreFlag = ignoreFlag
		# Screen region that contains the game
		self.game_region = region
		if region == None:
			region == (0, 0, 800, 600)
		self.speed = speed

	def find_game(self, image=None):
		# Find game field by looking for squares of appropriate color placed in a grid.
		# image: PIL Image
		# Return 2d array of (x1, y1, x2, y2) of found cells.

		def find_square(left, top):
			# Check if x, y is a top left corner of a rectangle pixels are from parent method.
			# Square should be the same color as it's top left corner
			color = pixels[left, top]

			# Find width
			right = left
			while right < image.size[0] and pixels[right+1, top] == color:
				right += 1

			# Find height
			bottom = top
			while bottom < image.size[1] and pixels[left, bottom + 1] == color:
				bottom += 1

			# Check if all the pixels are of the needed color
			for i in range(left, right + 1):
				for j in range(top, bottom + 1):
					# This is not a one-color square
					if pixels[i, j] != color:
						return False, False, False, False

			return left, top, right, bottom

		def find_all_squares():
			draw = ImageDraw.Draw(image)
			for i in range(image.size[0]):
				for j in range(image.size[1]):
					if pixels[i, j] in self.settings.field_color:
						left, top, right, bottom = find_square(i, j)

						# If the square is found large and "square-ish" enough, store 4 coordinates in "found"

						if left and \
							right - left > self.settings.minimum_cell_size and \
							(bottom - top) > 0 and \
							1.1 > (right-left) / (bottom - top) > 0.9:
							found.append((left, top, right, bottom))

							# Fill it with black so it would not be found again
							draw.rectangle((left, top, right, bottom), fill="black")
						else:
							# Paint it over, so we will not have to test
							# these pixels again
							draw.line((left, top, right, top), fill="black")
							draw.line((left, top, left, bottom), fill="black")
			return found

		def filter_grid(found):
			# Filter found squares that are on a "grid": repeating int coordinates in the list
			# Count all x and y coordinates of all the squares we found
			x_count, y_count = {}, {}
			for left, top, right, bottom in found:
				x_count[left] = x_count.get(left, 0) + 1
				y_count[top] = y_count.get(top, 0) + 1
				x_count[right] = x_count.get(right, 0) + 1
				y_count[bottom] = y_count.get(bottom, 0) + 1

			# Calculate "weight" - how often this squares coordinates are present in other squares
			found_with_weights = {}
			all_weights = []
			for left, top, right, bottom in found:
				weight = x_count[left] + y_count[top] + \
						x_count[right] + y_count[bottom]
				found_with_weights[(left, top, right, bottom)] = weight
				all_weights.append(weight)

			# Find median of all weights. Anything higher or equal to than will be in the final grid
			all_weights.sort()
			threshold = all_weights[len(all_weights) // 2]

			new_found = [coordinates
						 for coordinates, weight in found_with_weights.items()
						 if weight >= threshold]

			return new_found

		def deduce_game_parameters(found):
			# From the found squares, deduce game dimensions and the number of mines.

			game_width = len(set((left for left, _, _, _ in found)))
			game_height = len(set((top for _, top, _, _ in found)))
			game_mines = int(2.5 * (game_width * game_height)**(1/2))

			# Mine counts to recognize
			game_presets = {(8, 8): 10, (9, 9): 10, (16, 16): 40, (30, 16): 99}

			if (game_width, game_height) in game_presets:
				game_mines = game_presets[(game_width, game_height)]

			return (game_height, game_width), game_mines

		def arrange_cells(found):
			'''Arrange all found cells into a grid, in a form of NumPy array
			'''
			grid = np.array(found, dtype=object)
			grid = np.reshape(grid, list(self.game_shape[::-1]) + [4])
			grid = np.transpose(grid, axes = [1, 0, 2])
			return grid

		# Take a screenshot, if needed
		if image is None:
			image = pyscreeze.screenshot(region=self.game_region)

		# Pixels of the input image
		pixels = image.load()

		# We'll be putting found squares here:
		found = []

		# Find all potential squares
		found = find_all_squares()

		if len(found) < 10:
			print("Cannot find the game")
			return False

		# Filter those that are on the same grid
		found = filter_grid(found)

		# Determine game parameters (size, mines), from the found grid
		self.game_shape, deduced_mines = deduce_game_parameters(found)
		print(f"Found game of the size {self.game_shape}")

		# If no mine count passed to the bot - try to assume from the game size
		if self.game_mines is None:
			self.game_mines = deduced_mines
			print(f"Assuming {self.game_mines} mines")
		else:
			print(f"Mines are set to {self.game_mines}")

		# Sort them into rows and columns, store it in self.cells_coordinates
		self.cells_coordinates = arrange_cells(found)

		# Initiate solver
		self.solver = MinesweeperSolver(self.game_shape[0], self.game_shape[1], self.game_mines)

		return True

	def read_field(self, image):
		# Read the information from the field: covered and uncovered cells, numbers, mines, etc.
		# Return numpy array.
		def get_difference(image1, image2):
			p1 = image1.load()
			p2 = image2.load()
			diff = 0
			for i in range(min(image1.size[0], image2.size[0])):
				for j in range(min(image1.size[1], image2.size[1])):
					for c in range(3):
						diff += abs(p1[i, j][c] - p2[i, j][c])
			return diff

		def get_image_hash(img):
			data = []
			pixels = img.load()
			for i in range(img.size[0]):
				for j in range(img.size[1]):
					data.append(pixels[i, j])
			return hash(tuple(data))

		def read_cell(img):
			# Check if we saw this one before
			image_hash = get_image_hash(img)
			if image_hash in self.cell_cache:
				return self.cell_cache[image_hash]

			# Compare the image with known cell samples
			best_fit_difference = None
			best_fit_value = None

			for sample, value in self.settings.samples:
				# Calculate difference with a sample
				difference = get_difference(sample, img)

				# Check with all and use the closest one, but only if difference is smaller than sensitivity.
				if difference < self.settings.sample_sensitivity:
					if best_fit_difference is None or difference < best_fit_difference:
						best_fit_difference = difference
						best_fit_value = value

			if best_fit_value is not None:
				# Store the result in cache
				self.cell_cache[image_hash] = best_fit_value
				return best_fit_value

			return None

		field = np.zeros(self.game_shape, dtype=int)
		for i in range(self.game_shape[0]):
			for j in range(self.game_shape[1]):
				left, top, right, bottom = self.cells_coordinates[i, j]
				cell = image.crop((left - 2, top - 2, left + 15, top + 15))
				cell_value = read_cell(cell)
				
				if cell_value is None:
					cell_value = 0
					print(f"Can't read cell at ({i}, {j})")
					filename = f"./errors/unknown-{i}-{j}.png"
					cell.save(filename)
					raise Exception(f"Can't read cell at ({i}, {j})," + f"saved as {filename}")

				field[i, j] = cell_value

		return field

	def do_clicks(self, prob):
		def _click_at(coord, button):
			left, top, right, bottom = self.cells_coordinates[coord]
			x_coord = (left + right) // 2
			y_coord = (top + bottom) // 2
			pyautogui.moveTo(x_coord, y_coord, self.speed)
			pyautogui.click(button=button)

		# Find certain mines / safe if any
		safe = [(x, y) for x, y in zip(*np.nonzero(prob == 0))]
		mines = [(x, y) for x, y in zip(*np.nonzero(prob == 1))]

		# Given the safe and mines coordinates, do the clicks
		if self.ignoreFlag:
			mines = []

		for button, coord_list in zip(("right", "left"), (mines, safe)):
			if not coord_list:
				continue
			for coord in coord_list:
				_click_at(coord, button)

		if not mines and not safe:
			# Click the cell with the lowest nonzero probability, may use preference rule to break equality
			np.nan_to_num(prob, copy=False, nan = 1)
			prob[prob == 0] = 1
			print(prob)
			print('\n')
			_click_at(np.unravel_index(np.argmin(prob, axis=None), prob.shape), "left")

		pyautogui.moveTo(0, 0, self.speed)

	def is_dead(self, field):
		# Check if there is an exploded mine on the field, which means the game is over
		return (field == CELL_EXPLODED_MINE).any()

	def finish(self, field):
		# Check if there aren't any covered cells left
		return not (field == CELL_COVERED).any()

	def make_a_move(self, screenshot=None):
		# Read the situation on the board, run a solver for the next move, click the cells
		if screenshot is None:
			screenshot = pyscreeze.screenshot(region=self.game_region)

		# Read the field
		field = self.read_field(screenshot)

		# Check if the game is over
		if self.is_dead(field):
			return STATUS_DEAD
		if self.finish(field):
			return STATUS_WON

		# Get the solution to the current field
		self.do_clicks(self.solver.solve(field))
		return STATUS_ALIVE