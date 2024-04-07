from setting import *
import interface

def main():
	# Create a new automation object
	bot = interface.MinesweeperAutomation()
	# Find the game on the screen
	game_found = bot.find_game()
	if not game_found:
		return
	while True:
		# Read a screen, do clicks
		result = bot.make_a_move()
		if result == STATUS_DEAD or result == STATUS_WON:
			break;

if __name__ == "__main__":
	main()