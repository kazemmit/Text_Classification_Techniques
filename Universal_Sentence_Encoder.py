import tensorflow as tf
import tensorflow_hub as hub

def read_USE_model():
  '''
  To load the model once. Next loading would be much faster
  :return:
  '''
  return hub.Module(use_pretrain_model_url)


def init_USE_model():
  # Create tf graph
  g = tf.Graph()
  with g.as_default():
    # We will be feeding 1D tensors of text into the graph.
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module(use_pretrain_model_url)
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
  g.finalize()

  # Create session and initialize.
  session = tf.Session(graph=g)
  session.run(init_op)
  return session,embedded_text,text_input

