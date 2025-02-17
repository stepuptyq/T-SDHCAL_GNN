def count_max_seq_len(data):
    max_seq_len = 0
    for i in range(len(data)):
        my_len = len(data[i])
        if my_len > max_seq_len:
            max_seq_len = my_len
    return max_seq_len