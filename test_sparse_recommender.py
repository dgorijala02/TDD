import pytest
from sparse_recommender import SparseMatrix


# Test case for initializing a sparse matrix
def test_initialize_sparse_matrix():
    matrix = SparseMatrix(5, 5)
    assert matrix.rows == 5
    assert matrix.cols == 5

matrix = SparseMatrix(3, 4)
def test_set_valid_value():
    matrix.set(0, 0, 1)
    assert matrix.get(0, 0) == 1
    matrix.set(1, 2, 3)
    assert matrix.get(1, 2) == 3

def test_set_invalid_index():
    with pytest.raises(ValueError):
        matrix.set(4, 0, 1)  # Invalid row index
    with pytest.raises(ValueError):
        matrix.set(0, 5, 1)  # Invalid column index

def test_get_existing_value():
    matrix.set(0, 1, 2)
    assert matrix.get(0, 1) == 2
    matrix.set(2, 3, 4)
    assert matrix.get(2, 3) == 4

def test_get_non_existing_value():
    assert matrix.get(1, 1) == 0  # Default value for non-existing entry
    assert matrix.get(2, 0) == 0  # Default value for non-existing entry

def test_recommend():
    matrix = SparseMatrix(3, 4)
    matrix.set(0, 0, 1)
    matrix.set(0, 1, 2)
    matrix.set(1, 2, 3)
    user_vector = [1, 2, 3, 4]
    result = matrix.recommend(user_vector)
    assert result == [5, 9, 0]

# Test case for handling invalid vector dimension in multiplication
def test_recommend_invalid_dimension():
    matrix = SparseMatrix(3, 4)
    user_vector = [1, 2, 3]  # Vector with incorrect dimension
    with pytest.raises(ValueError):
        matrix.recommend(user_vector)

# Test case for matrix addition
def test_add_movie():
    matrix1 = SparseMatrix(3, 4)
    matrix1.set(0, 0, 1)
    matrix1.set(1, 1, 2)
    matrix2 = SparseMatrix(3, 4)
    matrix2.set(0, 0, 3)
    matrix2.set(2, 2, 4)
    result = matrix1.add_movie(matrix2)
    assert result.get(0, 0) == 4
    assert result.get(1, 1) == 2
    assert result.get(2, 2) == 4

# Test case for handling matrix addition with invalid dimensions
def test_add_movie_invalid_dimension():
    matrix1 = SparseMatrix(3, 4)
    matrix2 = SparseMatrix(2, 3)  # Matrices with different dimensions
    with pytest.raises(ValueError):
        matrix1.add_movie(matrix2)

# Test case for converting a sparse matrix to a dense matrix
def test_to_dense():
    matrix = SparseMatrix(2, 3)
    matrix.set(0, 0, 1)
    matrix.set(1, 2, 2)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[1, 0, 0], [0, 0, 2]]

def test_to_dense_non_square_matrix():
    # Test with a non-square sparse matrix
    matrix = SparseMatrix(4, 3)
    matrix.set(0, 0, 1)
    matrix.set(1, 1, 2)
    matrix.set(2, 2, 3)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[1, 0, 0], [0, 2, 0], [0, 0, 3], [0, 0, 0]]

def test_to_dense_negative_values():
    # Test with a sparse matrix containing negative values
    matrix = SparseMatrix(3, 3)
    matrix.set(0, 0, -1)
    matrix.set(1, 1, -2)
    matrix.set(2, 2, -3)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[-1, 0, 0], [0, -2, 0], [0, 0, -3]]

def test_to_dense_duplicate_values():
    # Test with a sparse matrix containing duplicate values
    matrix = SparseMatrix(3, 3)
    matrix.set(0, 0, 1)
    matrix.set(1, 1, 1)
    matrix.set(2, 2, 1)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]





