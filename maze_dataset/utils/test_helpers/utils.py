# Used for comparison of adj_lists
# Adj_list looks like [[[0, 1], [1, 1]], [[0, 0], [0, 1]], ...]
# We don't care about order of coordinate pairs within
# the adj_list or coordinates within each coordinate pair.
def adj_list_to_nested_set(adj_list):
    return {
        frozenset([tuple(start_coord), tuple(end_coord)])
        for start_coord, end_coord in adj_list
    }
