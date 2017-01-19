
def create_board(rows, columns, rows_list):
    board = {}
    for row in range(rows):
        for column in range(columns):
            board[(row, column)] = rows_list[row].split(' ')[column] 
    return board

class ChessTest(object):

    def __init__(self, piece, number_length, valid_starting_digits, rows, columns, board):
        """
        Initializes a ChessTest object.
        piece - can be either 'knight' or 'bishop'
        number_length - length of the numbers to be found
        valid_starting_digits - list of valid starting digits
        board - the board represented as a dictionary
        """
        self.knight_moves = [(2, -1), (2, 1), (-2, -1), (-2, 1),
                            (1, 2), (-1, 2), (1, -2), (-1, -2)]
        self.bishop_steps = [(1, 1), (1, -1), (-1 , 1), (-1, -1)]
        self.bishop_moves = []
        for step in range(1, max(rows, columns)):
            self.bishop_moves.extend([tuple(x * step for x in bishop_step) for bishop_step in self.bishop_steps])
        self.piece = piece
        if self.piece == 'knight':
            self.possible_moves = self.knight_moves
        if self.piece == 'bishop':
            self.possible_moves = self.bishop_moves
        self.number_length = number_length
        self.valid_starting_digits = valid_starting_digits
        self.board = board
        self.possible_numbers= {}

    def available_moves(self, starting_position):
        """
        Given a position provided as a tuple (row, column) returns all the
        available moves destinations given the board.
        """
        available_destinations = []
        for move in self.possible_moves:
            new_destination = (starting_position[0] + move[0], starting_position[1] + move[1])
            if new_destination in self.board.keys():
                try:
                    int(self.board[new_destination])
                    available_destinations.append(new_destination)
                except ValueError:
                    pass
        return available_destinations

    def add_moves(self, starting_sequence, current_place, step_to_go):
        """
        Given a sequence, the current spot on the board and the number of steps to go
        recursively appends all the available values obtained with a move

        Every sequence gets stored in the possible_numbers dictionary.

        It will keep appending moves until the counter is <= 1.
        """
        if step_to_go <= 1:
            self.possible_numbers[starting_sequence] = 1
            return
        available_destinations = self.available_moves(current_place)
        for destination in available_destinations:
            new_sequence = starting_sequence + self.board[destination]
            self.add_moves(new_sequence, destination, step_to_go - 1)

    def find_all_numbers(self, step_to_go):
        """
        It resets the possible_numbers dictionary
        and for every character in the board
        uses the add_moves method to start the 
        recursive search if the number in self.valid_starting_digits.
        """
        self.possible_numbers = {}
        
        for position, character in self.board.items():
            if character in self.valid_starting_digits:
                self.add_moves(character, position, step_to_go)

    def print_possible_numbers(self):
        """
        It prints to standard output the number of possible_numbers found.
        """
        print(len(self.possible_numbers))



if __name__ == '__main__':
    
    #managing inputs
    input_piece = input()
    input_number_length = int(input())
    input_starting_digits = input().split(' ')
    input_rows = int(input())
    input_columns = int(input())
    input_rows_list = []
    for row in range(input_rows):
        row = input()
        input_rows_list.append(row)

    board = create_board(input_rows, input_columns, input_rows_list)
    chess_test = ChessTest(input_piece, input_number_length, input_starting_digits, input_rows, input_columns, board)
    chess_test.find_all_numbers(input_number_length)
    chess_test.print_possible_numbers()









