import random
import cv2
import OCR
import pyautogui
import numpy
import keyboard
import copy
BOARD_LENGTH = 9

class Sudoku:
    #--------------------Sudoku class functions--------------------
    def __init__(self):
        self.grid = [[0 for _ in range(BOARD_LENGTH)] for _ in range(BOARD_LENGTH)]

    def print_board(self):
        for row in range(BOARD_LENGTH):
            for column in range(BOARD_LENGTH):
                if self.grid[row][column] != 0:
                    print(self.grid[row][column], end=" ")
                else:
                    print(" ", end=" ")
            print()
    #-------------------puzzle generation--------------------------
    def generate_puzzle(self, difficulty="medium"):
        #we fill the diagonal blocks, to reduce the number of branches
        #the backtracking algorithm will have
        self.fill_diagonal_blocks()

        # Fill in the grid via back tracking
        self.fill_via_backtracking()

        if difficulty == "easy":
            self.remove_numbers(35)  
        elif difficulty == "medium":
            self.remove_numbers(45)
        elif difficulty == "hard":
            self.remove_numbers(55)

    def remove_numbers(self, amount_to_remove):
        for _ in range(amount_to_remove):
            row, column = random.randint(0, 8), random.randint(0, 8)
            while self.grid[row][column] == 0:
                row, column = random.randint(0, 8), random.randint(0, 8)
            self.grid[row][column] = 0

    def fill_diagonal_blocks(self):
        for block in range(0,9,3):
            subgrid_numbers = [_ for _ in range(1, 10)]
            random.shuffle(subgrid_numbers)
            for row in range(3):
                for column in range(3):
                    self.grid[block + row][block + column] = subgrid_numbers[row * 3 + column]
    #--------------------------------------back_tracking and its insert validation----------------------
    def fill_via_backtracking(self, row = 0, column = 0):
        #if end of row return true to end this recursive branch
        if row == 9:
            return True
        #if end of column start on the next recursive branch 
        if column == 9:
            return self.fill_via_backtracking(row + 1, 0)
        # skip the entries in which that already have a number in them
        if self.grid[row][column] != 0:
            return self.fill_via_backtracking(row, column + 1)
        #generate random numbers for the row
        numbers = [_ for _ in range(1, 10)]
        random.shuffle(numbers)
        
        #check to see if the number can be inserted, if it can insert it
        # check to see that at the end of the branching its a valid sudoku puzzle
        # other wise back track and set all the filled in entries to 0
        for num in numbers:
            if self.valid_number_insertion(row, column, num):
                self.grid[row][column] = num
                if self.fill_via_backtracking(row, column + 1):
                    return True
                self.grid[row][column] = 0
        
        return False
    
    def valid_number_insertion(self, row, column, number_to_insert):
        return (
            self.valid_row_insertion(row, number_to_insert) and
            self.valid_column_insertion(column, number_to_insert) and
            self.valid_subgrid_insertion(row - row % 3, column - column % 3, number_to_insert)
        )
    
    def valid_row_insertion(self, row, number_to_insert):
        return number_to_insert not in self.grid[row]
    
    def valid_column_insertion(self, column, number_to_insert):
        return all(self.grid[i][column] != number_to_insert for i in range(9))
    
    def valid_subgrid_insertion(self, start_row, start_column, number_to_insert):
        return all(
            self.grid[row][column] != number_to_insert
            for row in range(start_row, start_row + 3)
            for column in range(start_column, start_column + 3)
        )

#this does not work as intended

copy_of = Sudoku()
def screen_solve():
    sudoku = Sudoku()
    screenshot = pyautogui.screenshot()

    # Convert screenshot to OpenCV format
    image = numpy.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        table, x_offset, y_offset = OCR.return_puzzle(image)
        if(table is not None):
            for row in range(BOARD_LENGTH):
                for column in range(BOARD_LENGTH):
                    sudoku.grid[row][column] = int(table[BOARD_LENGTH*row + column].text)
            sudoku.print_board()
            copy_of = copy.deepcopy(sudoku)
            print("--------------------------------------------------------")
            sudoku.fill_via_backtracking()
            sudoku.print_board()
        for row in range(BOARD_LENGTH):
                for column in range(BOARD_LENGTH):
                    x = x_offset + table[BOARD_LENGTH*row + column].topleft[0]
                    y = y_offset + table[BOARD_LENGTH*row + column].topleft[1]
                    pyautogui.click(x, y)
                    print(copy_of.grid[row][column])
                    if(copy_of.grid[row][column] == 0):
                        print("it has made it here")
                        pyautogui.press(str(sudoku.grid[row][column]))
    except Exception as e:
        print("An error occurred:", e)
if __name__ == "__main__":
    while True:
        # Check if the user presses CTRL + 5
        if keyboard.is_pressed('ctrl') and keyboard.is_pressed('5'):
            screen_solve()