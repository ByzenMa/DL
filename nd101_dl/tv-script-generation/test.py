import tensorflow as tf


def testTF(batch_size, rnn_size):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2)
    _initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(_initial_state, name="initial_state")
    # outputs, final_state =  tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    # tf.layers.dense(outputs,vocab_size,)
    tf.nn.embedding_lookup()
    return cell, initial_state


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_to_int = {}
    for word in text.split(' '):
        if word not in vocab_to_int:
            vocab_to_int[word] = len(vocab_to_int)
    int_to_vocab = dict((v, k) for k, v in vocab_to_int.iteritems())
    return vocab_to_int, int_to_vocab


if __name__ == '__main__':
    vocab_to_int, int_to_vocab = create_lookup_tables('This is good!')
    print vocab_to_int
    print int_to_vocab
