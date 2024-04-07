from PIL import Image

# Cell types
CELL_MINE = -1
CELL_COVERED = -2
CELL_FLAG = -3
CELL_EXPLODED_MINE = -4
# MinesweeperGame.status: returned by the do_move, tells the result of the move
STATUS_ALIVE = 0
STATUS_DEAD = 1
STATUS_WON = 2

class MinesweeperSettings():
	def __init__(self, field_color, samples_files, sample_sensitivity=35000):
		# Color used to find a grid. This should be the most central color of a closed cell
		self.field_color = field_color
		# Load sample pictures of cells
		self.samples = [(Image.open(file), value) for file, value in samples_files.items()]
		# How many pixels to pad when cut out a cell picture
		self.sample_sensitivity = sample_sensitivity
		# Minimum size to be considered a potential cell
		self.minimum_cell_size = 10

# Settings for classic minesweeper versions
SETTINGS_MINESWEEPER_CLASSIC = MinesweeperSettings(
	field_color=[(192, 192, 192), (192, 192, 192, 255)],
	samples_files={
		"./rsrc/0.png": 0,
		"./rsrc/1.png": 1,
		"./rsrc/2.png": 2,
		"./rsrc/3.png": 3,
		"./rsrc/4.png": 4,
		"./rsrc/5.png": 5,
		"./rsrc/6.png": 6,
		"./rsrc/mine.png": CELL_MINE,
		"./rsrc/covered.png": CELL_COVERED,
		"./rsrc/flag.png": CELL_FLAG,
		"./rsrc/explosion.png": CELL_EXPLODED_MINE
		}
	)