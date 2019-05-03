import tensorflow as tf
import numpy as np
import math

class DeepSaliencyTreeRenderer():
    def __init__(self, callback_locals, feed_dict):
        self.session = callback_locals['self'].sess
        self.graph = callback_locals['self'].graph
        self.env = callback_locals['self'].env
        self.feed_dict = feed_dict


    def find_most_relevant_nodes(self, layer, relevance_values, layer_values, current_batch_index, ymin, xmin):
        """Find most relevant neurons, using softmax, distance, and/or some cutoff.
        Find tensor based on activation or relevance
        Use scipy.signal.find_peaks with a prominence minimum to find peaks"""

        # peak_indices = scipy.signal.find_peaks(values)    # one day we will use this

        # values.shape = [batch_size, h, w, channels]
        #peak_indices = [np.unravel_index(
        #    relevance_values[current_batch_index, ...].argmax(),
        #    relevance_values[current_batch_index, ...].shape)]
        #if relevance_values.shape[1] == 1 and relevance_values.shape[2] == 1:
        #    peak_indices = list(zip(*np.unravel_index(np.argpartition(relevance_values, -3, axis=None)[-3:], relevance_values[current_batch_index, ...].shape)))
        #else:
        #print("Finding most relevant nodes, layer size", layer_geometry['size'])
        #print("Relevance values has shape", relevance_values.shape)
        top_x = int(np.floor(4*np.exp(-layer['size']*0.1)+2))
        #print("top x is", top_x)
        peak_indices = filter_k(relevance_values, order_method='max', k=2, top_x=top_x, method='peak')

        if len(peak_indices) == 0:
            print("Peak indices:", peak_indices)
            print("Relevance shape:", relevance_values.shape)
            print("relevance values:", relevance_values)
            print("topx", top_x)
            plt.imshow(relevance_values[:, :, 0])
            plt.show()

        # Cluster each filter channel activation into its centroid(s) by magnitude, using gap statistic
        # Find the magnitudes of the relevance at the centroids
        # Pick the top n centroids as sorted by their magnitude

        relevant_node_infos = []
        for i in peak_indices:
            lcc = layer['coordinate_converter']
            yc = i[0] + ymin
            xc = i[1] + xmin
            relevant_node_infos.append({
                'coordinates': [current_batch_index, yc, xc, i[2]],
                'global_coordinates': [current_batch_index, lcc(yc), lcc(xc), i[2]],
                'node': layer['pre_activation'][current_batch_index, yc, xc, i[2]],
                'layer': layer['pre_activation'],
                'weights': layer['input_weights'][:, :, :, i[2]],
                'size': layer['size'],
                'raw_value': layer_values[0][current_batch_index, yc, xc, i[2]].item(),
                'activation_value': layer_values[1][current_batch_index, yc, xc, i[2]].item(),
                'relevance_value': relevance_values[yc, xc, i[2]].item(),
                'children': [],
            })

        return relevant_node_infos


    def update_relevant_children_for_node(self, node_info, child_layers, child_layer_values, current_batch_index):
        # 1. Calculate relevance scores for all input nodes to this node
        # 2. Select relevant input nodes based on relevance scores
        # 3. Call update_relevant_children_for_node on each of the new children
        # 4. Don't continue if input_layer is None

        input_weights = node_info['weights'] # 3D tensor, size (kernel_size, kernel_size, child_filters)

        # For a given layer and a given filter, the weights are the same

        hs = (node_info['size']-1)/2
        rf_ymin = int(node_info['coordinates'][1] - hs)
        rf_ymax = int(node_info['coordinates'][1] + hs + 1)
        rf_xmin = int(node_info['coordinates'][2] - hs)
        rf_xmax = int(node_info['coordinates'][2] + hs + 1)

        #print("HS:", hs, "yminmax, xminmax", rf_ymin, rf_ymax, rf_xmin, rf_xmax)

        #node_info is the top layer node, now we try to find children for it.
        #input activations

        input_activations = child_layer_values[1][current_batch_index, rf_ymin:rf_ymax, rf_xmin:rf_xmax, :]

        input_relevance_values = input_activations * input_weights

        print("relevance", input_relevance_values)

        #print("INPUT WEIGHTS", input_weights.shape, "INPUT ACTRIVATIONS", input_activations.shape, "RELEVANCE", input_relevance_values.shape)

        relevant_child_nodes = self.find_most_relevant_nodes(child_layers[-1], input_relevance_values, child_layer_values, current_batch_index, rf_ymin, rf_xmin)

        if (len(relevant_child_nodes) == 0):
            print("No child nodes found")
            print("child layers:", len(child_layers))
            print("relevance values:", input_relevance_values.shape)
            print("child layer values:", len(child_layer_values))
            print("batch index:", current_batch_index)
            print("rfyxmin", rf_ymin, rf_xmin)

        #print("Found relevant child nodesL:", relevant_child_nodes)

        if len(child_layers) >= 2: # if there's another layer below the children we just found:
            grandchild_layer_values = self.session.run([child_layers[-2]['pre_activation'], child_layers[-2]['activation']], self.feed_dict)
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

        trees_by_batch = []
        for batch_index in range(self.session.run(layers[0]['pre_activation'], self.feed_dict).shape[0]):
            #print("Finding branch clusters for batch", batch_index)
            layer_values = self.session.run([layers[-1]['pre_activation'], layers[-1]['activation']], self.feed_dict) # np.array
            child_layer_values = self.session.run([layers[-2]['pre_activation'], layers[-2]['activation']], self.feed_dict)

            # use layer_values[1] here to use the softmax of the attention2
            relevant_node_infos = self.find_most_relevant_nodes(layers[-1], layer_values[1][batch_index, ...], layer_values, batch_index, 0, 0)

            for node_info in relevant_node_infos:
                 node_info['children'] = self.update_relevant_children_for_node(node_info, layers[:-1], child_layer_values, batch_index)

            trees_by_batch.append(relevant_node_infos)

        return trees_by_batch

    def add_layer_coordinate_converters(self, layers):
        def create_converter(lg, pl):
            conv = lambda c: ((lg['size']-1)/2 + lg['strides']*c) * pl['global_strides'] + (pl['global_size']-1)/2
            return conv

        for (li, lg) in enumerate(layers):
            lg['coordinate_converter'] = create_converter(lg, layers[li-1] if li >= 1 else {'global_strides': 1, 'global_size': 1})

        return layers

    def add_global_layer_sizes(self, layers):
        layers[0]['global_size'] = layers[0]['size']
        layers[0]['global_strides'] = layers[0]['strides']
        for (li, lg) in enumerate(layers[1:]):
            lg['global_size'] = layers[li]['global_size'] + (lg['size'] - 1) * layers[li]['global_strides']
            lg['global_strides'] = layers[li]['global_strides'] * lg['strides']
        return layers


    def draw_square(self, node, layer, parent_node, **kwargs):
        (y, x, c) = (node['global_coordinates'][1], node['global_coordinates'][2], node['global_coordinates'][3])
        hs = layer['global_size'] / 2
        plt.plot([x-hs, x+hs, x+hs, x-hs, x-hs], [y-hs, y-hs, y+hs, y+hs, y-hs], color='w', **kwargs)

    def draw_node(self, node, layer, parent_node, **kwargs):
        (y, x, c) = (node['global_coordinates'][1], node['global_coordinates'][2], node['global_coordinates'][3])
        plt.plot([x], [y], 'o', color=layer['color'], **kwargs)

    def draw_branch(self, node, layer, parent_node, **kwargs):
        (y, x, c) = (node['global_coordinates'][1], node['global_coordinates'][2], node['global_coordinates'][3])
        (py, px, pc) = (parent_node['global_coordinates'][1], parent_node['global_coordinates'][2], parent_node['global_coordinates'][3])
        plt.plot([x, px], [y, py], color='w', **kwargs)


    def plot_relevant_nodes(self, nodes, layers, parent_node=None):
        # plot a dot where the node is
        for n in nodes:
            #print("Layer", len(layers))
            if len(layers) == 1:
                #print("Center square", n['global_coordinates'])
                self.draw_square(n, layers[-1], parent_node)
            else:
                #print("Center node", n['global_coordinates'])
                self.draw_node(n, layers[-1], parent_node)

            if parent_node is not None:
                #print("and refer to parent", parent_node['global_coordinates'])
                self.draw_branch(n, layers[-1], parent_node)

            self.plot_relevant_nodes(n['children'], layers[:-1], n)

    def save_tree_and_attention(self, frame_index, rts, input_values, layers):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if tf.contrib.framework.is_tensor(obj):
                    return "Tensor (not serialized)"
                try:
                    return json.JSONEncoder.default(self, obj)
                except:
                    return "Could not encode this"

        out_json = {
            'rts': rts,
            'input_values': input_values,
            'frame_index': frame_index,
            'layers': layers,
        }

        file_path = "/tmp/" + str(frame_index) + ".json" ## your path variable
        json.dump(out_json, codecs.open(file_path, 'w', encoding='utf-8'), cls=NumpyEncoder, separators=(',', ':'), sort_keys=True, indent=4)
        ### this saves the array in .json format

    def plot_relevance_tree(self, layers, a2, input_values, frame_index):
        # get relevance tree
        layers = self.add_global_layer_sizes(layers)
        layers = self.add_layer_coordinate_converters(layers)

        rts = self.get_relevance_tree_for_layers(layers)
        # traverse through the tree
        for (batch_index, rt) in enumerate(rts):
            # plot activations
            plt.imshow(a2[batch_index, :, :, 0])
            plt.colorbar()
            plt.show()
            plt.imshow(a2[batch_index, :, :, 1])
            plt.colorbar()
            plt.show()
            # plot input frames

            plt.figure(figsize=(14, 10))
            plt.imshow(np.max(input_values[batch_index, :, :, :], axis=-1), cmap='gray')
            self.plot_relevant_nodes(rt, layers)
            plt.show()

            self.save_tree_and_attention(frame_index, rts, input_values, layers)


class Callback(object):
    def __init__(self, display_frames=False, display_saliency_map=False, display_deep_saliency_tree=False):
        self.display_frames = display_frames
        self.display_saliency_map = display_saliency_map
        self.display_deep_saliency_tree = display_deep_saliency_tree
        self.pbar = None
        self.frame_index = 0

    def __call__(self, _locals, _globals):
        if self.pbar is None:
            self.pbar = tqdm_notebook(total=_locals['nupdates'] * _locals['self'].n_batch)

        self.pbar.update(_locals['self'].n_batch)
        self.pbar.set_postfix_str('{update}/{nupdates} updates'.format(**_locals))

        self.session, self.graph, self.env = _locals['self'].sess, _locals['self'].graph, _locals['self'].env

        #pprint(self.graph.get_operations())
        input_values = self.env.stackedobs
        input_tensor = self.graph.get_tensor_by_name("train_model/input/Ob:0")
        input_cast_tensor = self.graph.get_tensor_by_name("train_model/input/Cast:0")
        c1_activations = self.graph.get_tensor_by_name("train_model/model/Relu:0")
        c2_activations = self.graph.get_tensor_by_name("train_model/model/Relu_1:0")
        c3_activations = self.graph.get_tensor_by_name("train_model/model/Relu_2:0")
        a1_activations = self.graph.get_tensor_by_name("train_model/model/Elu:0")
        a2_activations = self.graph.get_tensor_by_name("train_model/model/a2/add:0")
        a2_softmax = self.graph.get_tensor_by_name("train_model/model/attn:0")
        c1_input_weights = self.graph.get_tensor_by_name("model/c1/w:0")
        c2_input_weights = self.graph.get_tensor_by_name("model/c2/w:0")
        c3_input_weights = self.graph.get_tensor_by_name("model/c3/w:0")
        a1_input_weights = self.graph.get_tensor_by_name("model/a1/w:0")
        a2_input_weights = self.graph.get_tensor_by_name("model/a2/w:0")


        if _locals['update'] == _locals['nupdates']:
            self.pbar.close()
            self.pbar = None

        if _locals['update'] % 1 == 1 or _locals['update'] == _locals['nupdates'] or True:
            if self.display_frames:
                plt.grid(None)
                plt.imshow(_locals["self"].env.render(mode='rgb_array'))
                plt.show()

            if self.display_saliency_map:
                sr = SaliencyRenderer(_locals)
                smap = sr.get_basic_input_saliency_map(
                    input_tensor, input_values, input_cast_tensor, a2_activations,
                    selection_method='SUM', n_gradient_samples=10, gradient_sigma_spread=0.15)
                plt.imshow(smap[0, :, :, 0])
                plt.colorbar()
                plt.show()

            if self.display_deep_saliency_tree:
                layers = [
                   {
                       'pre_activation': c1_activations.op.inputs[0],
                       'activation': c1_activations,
                       'size': 8, 'strides': 4, 'color': 'white',
                       'input_weights': self.session.run(c1_input_weights),
                   },
                   {
                       'pre_activation': c2_activations.op.inputs[0],
                       'activation': c2_activations,
                       'size': 4, 'strides': 2, 'color': 'green',
                       'input_weights': self.session.run(c2_input_weights),
                   },
                   {
                       'pre_activation': c3_activations.op.inputs[0],
                       'activation': c3_activations,
                       'size': 3, 'strides': 1, 'color': 'red',
                       'input_weights': self.session.run(c3_input_weights),
                   },
                   {
                       'pre_activation': a1_activations.op.inputs[0],
                       'activation': a1_activations,
                       'size': 1, 'strides': 1, 'color': 'yellow',
                       'input_weights': self.session.run(a1_input_weights),
                   },
                   {
                       'pre_activation': a2_activations,
                       'activation': a2_softmax,
                       'size': 1, 'strides': 1, 'color': 'magenta',
                       'input_weights': self.session.run(a2_input_weights),
                   },
                ]

                dstr = DeepSaliencyTreeRenderer2(_locals, {input_tensor: input_values})
                dstr.plot_relevance_tree(layers, self.session.run(a2_softmax, {input_tensor: input_values}), input_values, self.frame_index)

            # Save current model
            self.frame_index += 1
            _locals['self'].save(str(output_dir / 'model.pkl'))

        return True

