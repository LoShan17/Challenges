
class TetrisEngine(object):


    def __init__(self, width=10):
        '''
        Initializes a TetrisEngine with a default 10 width
        '''
        self.pieces_dictionary = {'Q': [[1, 1], [1, 1]], 'Z': [[0,1,1], [1,1,0]],
                                  'S': [[1,1,0],[0,1,1]], 'T': [[0,1,0], [1,1,1]], 
                                  'I': [[1,1,1,1]], 'L': [[1,1], [1,0], [1,0]],
                                  'J': [[1,1], [0,1], [0,1]]}
        self.width = 10
        self.grid = []
        self.results = []


    def get_blocking_row(self, piece, leftmost_column):
        '''
        Identifies the row where the piece is going to land
        and returns row's index, -1 is returned if no rows
        are blocking the piece
        '''
        current_piece = self.pieces_dictionary[piece]
        if not self.grid:
            return -1        
        else:
            for index, row in enumerate(self.grid[::-1]):
                original_row_index = self.grid_height() - 1 - index
                for piece_row_index in range(min(index + 1, len(current_piece))):
                    for piece_column_index, piece_square in enumerate(current_piece[piece_row_index]):
                        if self.grid[original_row_index + piece_row_index][leftmost_column + piece_column_index] and piece_square:
                            return original_row_index
            return -1


    def update_grid(self, blocking_row, piece, leftmost_column):
        '''
        Updates the existing grid with piece and adds rows on top if needed
        '''
        current_piece = self.pieces_dictionary[piece]
        rows_to_update = self.grid_height() - (blocking_row + 1)
        for piece_row_index in range(min(len(current_piece), rows_to_update)):
            for piece_column_index, piece_square in enumerate(current_piece[piece_row_index]):
                if piece_square:
                    self.grid[piece_row_index + blocking_row + 1][piece_column_index + leftmost_column] = piece_square

        rows_to_add = len(current_piece) - (self.grid_height() - (blocking_row + 1))
        for index in range(rows_to_add):
            piece_row = current_piece[index + rows_to_update]
            new_grid_row = self.add_empty_spaces(leftmost_column, piece_row)
            self.grid.append(new_grid_row)
        return


    def clear_full_rows(self):
        '''
        Removes all full rows from grid
        '''
        cleared_grid=False
        while not cleared_grid:
            if not self.grid:
                cleared_grid=True
            for index, row in enumerate(self.grid):
                if all(row):
                    self.grid.pop(index)
                    break
                if index == self.grid_height() - 1:
                    cleared_grid=True
        return


    def add_empty_spaces(self, leftmost_column, piece_row):
        '''
        Returns new line to be added to the grid
        '''
        right_spaces = self.width - (leftmost_column + len(piece_row))
        return [0] * leftmost_column + piece_row + [0] * right_spaces


    def run_game(self, pieces_list):
        '''
        Runs a single game on Tetris going through pieces_list
        '''
        for piece in pieces_list:
            piece, leftmost_column = piece[0], int(piece[1])
            blocking_row = self.get_blocking_row(piece, leftmost_column)
            self.update_grid(blocking_row, piece, leftmost_column)
            self.clear_full_rows()


    def reset_grid(self):
        '''
        Resets TetrisEngine emptying the grid
        '''
        self.grid = []


    def reset_results(self):
        '''
        Resets TetrisEngine emptying the results list
        '''
        self.results = []


    def grid_height(self, add_to_results=False):
        '''
        Returns grid height
        '''
        height = len(self.grid)
        if add_to_results:
            self.results.append(height)
        return height


    def output_results(self, output_file_name='output.txt'):
        '''
        Creates output file from results list
        '''
        with open(output_file_name, 'w') as output_file:
            for result in self.results:
                output_file.write(str(result) + '\n')
                output_file.write('\n')


if __name__ == '__main__':
    with open('input.txt') as input:
        lines = [line.rstrip('\n') for line in input]
        lines = [line.split(',') for line in lines if line]

    tetris_engine = TetrisEngine()
    for line in lines:
        #print(line)
        tetris_engine.run_game(line)
        tetris_engine.grid_height(True)
        #print(tetris_engine.grid)
        tetris_engine.reset_grid()
        #print('#########################')
        #print('')
    tetris_engine.output_results()
