class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.dict = {}  # Use a dictionary to store non-zero elements

    def set(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.dict[(row, col)] = value
        else:
            raise ValueError("Invalid row or column index")

    def get(self, row, col):
        return self.dict.get((row, col),0)

    def recommend(self, vector):
        if len(vector) != self.cols:
            raise ValueError("Vector dimension does not match matrix columns")
        result = [0] * self.rows
        for (row, col), value in self.dict.items():
            result[row] += value * vector[col]
        return result

    def add_movie(self, matrix):
        if self.rows != matrix.rows or self.cols != matrix.cols:
            raise ValueError("Matrix dimensions do not match")
        result = SparseMatrix(self.rows, self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                value1 = self.get(row, col)
                value2 = matrix.get(row, col)
                if value2 is None:
                    value2 = 0  # Use default value of 0 if value2 is None
                result.set(row, col, value1 + value2)
        return result

    def to_dense(self):
        dense_matrix = [[0] * self.cols for _ in range(self.rows)]
        for (row, col), value in self.dict.items():
            dense_matrix[row][col] = value
        return dense_matrix
