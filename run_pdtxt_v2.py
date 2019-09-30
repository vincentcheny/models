import tensorflow as tf

with tf.compat.v1.Session() as sess:
    with open('./export_graphs/graph.pdtxt', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # output = tf.import_graph_def(graph_def, return_elements=['out:0'])
        x = tf.import_graph_def(graph_def, name='')
        print(x)
        # print(sess.run(x))
