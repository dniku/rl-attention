import tensorflow as tf
import numpy as np

class DeepSaliencyTreeRenderer():
    def __init__(self, callback_locals, feed_dict):
        self.session = callback_locals['self'].sess
        self.graph = callback_locals['self'].graph
        self.env = callback_locals['self'].env
        self.feed_dict = feed_dict


    def find_most_relevant_nodes(self, layer, relevance_values, layer_values, current_batch_index):
        """Find most relevant neurons, using softmax, distance, and/or some cutoff.
        Find tensor based on activation or relevance
        Use scipy.signal.find_peaks with a prominence minimum to find peaks"""

        # peak_indices = scipy.signal.find_peaks(values)    # one day we will use this

        # values.shape = [batch_size, h, w, channels]
        peak_indices = [np.unravel_index(
            relevance_values[current_batch_index, ...].argmax(),
            relevance_values[current_batch_index, ...].shape)]

        relevant_node_infos = []
        for i in peak_indices:
            relevant_node_infos.append({
                'coordinates': [current_batch_index, i[0], i[1], i[2]],
                'node': layer[current_batch_index, i[0], i[1], i[2]],
                'layer': layer,
                'raw_value': layer_values[0][current_batch_index, i[0], i[1], i[2]],
                'activation_value': layer_values[1][current_batch_index, i[0], i[1], i[2]],
                'relevance_value': relevance_values[current_batch_index, i[0], i[1], i[2]],
                'children': [],
            })

        return relevant_node_infos


    def update_relevant_children_for_node(self, node_info, child_layers, child_layer_values, current_batch_index):
        # 1. Calculate relevance scores for all input nodes to this node
        # 2. Select relevant input nodes based on relevance scores
        # 3. Call update_relevant_children_for_node on each of the new children
        # 4. Don't continue if input_layer is None
        
        gradients = self.session.run(tf.gradients(node_info['node'], child_layers[-1][1]), self.feed_dict)[0]
        
        #print("gradients to all shape", gradients_to_all.shape)
        #gradients_to_node = gradients_to_all[node_info['coordinates'][0], # current_batch_index
        #                                     node_info['coordinates'][1], # node y coordinate
        #                                     node_info['coordinates'][2], # node x coordinate
        #                                     node_info['coordinates'][3]] # node channel 
        
        relevance = gradients * child_layer_values[1] / node_info['raw_value'] # has values for all batches of input
        relevant_child_nodes = self.find_most_relevant_nodes(child_layers[-1][0], relevance, child_layer_values, current_batch_index)

        if len(child_layers) >= 2: # if there's another layer below the children we just found:
            grandchild_layer_values = self.session.run(child_layers[-2], self.feed_dict)
            for child_node_info in relevant_child_nodes:
                child_node_info['children'] = self.update_relevant_children_for_node(child_node_info,
                                                                                     child_layers[:-1],
                                                                                     grandchild_layer_values,
                                                                                     current_batch_index)

        return relevant_child_nodes


    def get_relevance_tree_for_layers(self, layers):
        # Evaluate relevance score for each unit.
        #   - If attention layer is given, use activations and/or coordinates to find top k % of relevant neurons.
        # Pick the top k percent of neurons.
        # Create a list of those neurons, and also record their local coordinates, global coordinates, before_activation_value, and activation_value

        # Then, for each of the neurons, compute children.
        # Find children for each neuron by running get_relevance_tree_for_neuron
        #

        # layers looks like [ [layer1_raw, layer1_relu], [layer2_raw, layer2_relu], ... [layer_n_raw, layer_n_relu]]
        trees_by_batch = []
        for batch_index in range(self.session.run(layers[0][0], self.feed_dict).shape[0]):
    
            layer_values = self.session.run(layers[-1], self.feed_dict) # np.array
            child_layer_values = self.session.run(layers[-2], self.feed_dict)

            relevant_node_infos = self.find_most_relevant_nodes(layers[-1][0], layer_values[0], layer_values, batch_index)

            for node_info in relevant_node_infos:
                 node_info['children'] = self.update_relevant_children_for_node(node_info, layers[:-1], child_layer_values, batch_index)

            trees_by_batch.append(relevant_node_infos)

        return trees_by_batch
