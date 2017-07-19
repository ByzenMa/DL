import problem_unittests as pu
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense


def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    input_pl = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets_pl = tf.placeholder(tf.int32, shape=[None, None])
    lr_pl = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='source_sequence_length')
    return input_pl, targets_pl, lr_pl, keep_prob, target_sequence_length, max_target_len, source_sequence_length


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    output, final_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_sequence_length,
                                         source_vocab_size, enc_embedding_size)
    target_data_processed = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    training_decoder_output, inference_decoder_output = decoding_layer(target_data_processed, final_state,
                                                                       target_sequence_length,
                                                                       max_target_sentence_length, rnn_size, num_layers,
                                                                       target_vocab_to_int, target_vocab_size,
                                                                       batch_size, keep_prob, dec_embedding_size)
    return training_decoder_output, inference_decoder_output


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

    def build_cell(lstm_size):
        cell = tf.contrib.rnn.LSTMCell(lstm_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return cell

    cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(rnn_size) for _ in range(num_layers)])
    cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
    output, final_state = tf.nn.dynamic_rnn(cell, embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    return output, final_state


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    with tf.variable_scope("decode"):
        training_decoder_output = decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                                                       target_sequence_length, max_target_sequence_length,
                                                       output_layer, keep_prob)
    with tf.variable_scope("decode", reuse=True):
        start_of_sequence_id = target_vocab_to_int['<GO>']
        end_of_sequence_id = target_vocab_to_int['<EOS>']
        inference_decoder_output = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                                                        end_of_sequence_id, max_target_sequence_length,
                                                        target_vocab_size, output_layer, batch_size, keep_prob)

    return training_decoder_output, inference_decoder_output


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens,
                                                                end_of_sequence_id)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)

    inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
    return inference_decoder_output


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=target_sequence_length,
                                                        time_major=False)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       encoder_state,
                                                       output_layer)
    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_summary_length)
    return training_decoder_output


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """

    def process_decoder_input(target_data, target_vocab_to_int, batch_size):
        """
        Preprocess target data for encoding
        :param target_data: Target Placehoder
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :param batch_size: Batch Size
        :return: Preprocessed target data
        """

    go_idx = target_vocab_to_int['<GO>']
    tiled_go_idx = tf.cast(tf.reshape(tf.tile(np.array([go_idx]), np.array([batch_size])), shape=[batch_size, -1]),
                           tf.int32)
    return tf.concat([tiled_go_idx, target_data], axis=1)[:, :-1]


def model_inputs():
    input_pl = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets_pl = tf.placeholder(tf.int32, shape=[None, None])
    lr_pl = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='target_sequence_length')
    max_target_len = tf.placeholder(tf.int32, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='source_sequence_length')
    return input_pl, targets_pl, lr_pl, keep_prob, target_sequence_length, max_target_len, source_sequence_length


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """

    def vocab_to_int(vocab_to_int_dict, text, target=False):
        if target:
            sents = [sent + ' <EOS>' for sent in text.split('\n')]
        else:
            sents = text.split('\n')
        vocab_to_int = []
        for sent in sents:
            ids = [vocab_to_int_dict[word] for word in sent.split(' ')]
            vocab_to_int.append(ids)
        return vocab_to_int

    source_id_text = vocab_to_int(source_vocab_to_int, source_text)
    target_id_text = vocab_to_int(target_vocab_to_int, target_text, target=True)
    return source_id_text, target_id_text


if __name__ == "__main__":
    pu.test_text_to_ids(text_to_ids)
    pu.test_model_inputs(model_inputs)
    pu.test_process_encoding_input(process_decoder_input)
    pu.test_decoding_layer_train(decoding_layer_train)
    pu.test_decoding_layer_infer(decoding_layer_infer)
    pu.test_decoding_layer(decoding_layer)
    pu.test_seq2seq_model(seq2seq_model)
