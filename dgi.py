import dgl
import torch
import torch.nn as nn
import tqdm

from torch.fx import GraphModule , Graph
import operator
from torch.fx.passes.split_utils import split_by_tags
import dgl.nn
from torch.fx import Tracer, GraphModule, Proxy
from torch.fx._compatibility import compatibility
import math
from dgl import DGLHeteroGraph

GET_ATTR = "get_attr"
CALL_MODULE = "call_module"
CALL_FUNCTION = "call_function"
CALL_METHOD = "call_method"
PLACEHOLDER = "placeholder"
OUTPUT = "output"

CONV_BLOCK = "conv_block"


class InferenceHelperBase():
    """Inference helper base class.

    This class is the base class for inference helper. Users can create an inference
    helper to compute layer-wise inference easily. The inference helper provides a
    simple interence, which users only need to call the ``inference`` function after
    create the InferenceHelper object.

    Parameters
    ----------
    root : torch.nn.Module
        The root model to conduct inference.
    device : torch.device
        The device to conduct inference computation.
    use_uva : bool
        Whether store graph and tensors in UVA.
    debug : bool
        Whether display debug messages.
    """
    def __init__(self, root: nn.Module, conv_modules = (), device = "cpu", \
                 use_uva = False, debug = False):
        # add a '_' in order not crash with the origin one.
        self._device = device
        self._use_uva = use_uva
        self._debug = debug
        graph_module = dgl_symbolic_trace(root, conv_modules)
        self._tags, self._splitted = split_module(graph_module, debug)
        self._wrap_conv_blocks()

    def _get_mock_graph(self, graph):
        """Get the mock graph."""
        if self._mock_graph is None:
            self._input_graph = graph
            if graph.is_homogeneous:
                self._mock_graph = dgl.graph(([0], [0]), device=self._device)
            else:
                data_dict = {}
                for canonical_etype in graph.canonical_etypes:
                    data_dict[canonical_etype] = ([0], [0])
                self._mock_graph = dgl.heterograph(data_dict, device=self._device)
        return self._mock_graph

    def _trace_output_shape(self, func, *args):
        """Trace the output shape."""
        mock_input = ()
        for arg in args:
            if isinstance(arg, dgl.DGLHeteroGraph):
                mock_input += (self._get_mock_graph(arg),)
            elif isinstance(arg, torch.Tensor):
                mock_input += (arg[[0]].to(self._device),)
            else:
                raise Exception("Input type not supported yet.")

        assert self._input_graph is not None
        mock_rets = func(*mock_input)

        if not isinstance(mock_rets, tuple):
            mock_rets = (mock_rets,)
        ret_shapes = []
        for mock_ret in mock_rets:
            if isinstance(mock_ret, torch.Tensor):
                ret_shapes.append((self._input_graph.number_of_nodes(),) + mock_ret.size()[1:])
            else:
                raise Exception("Output type not supported yet.")
        return ret_shapes

    def _wrap_conv_blocks(self):
        """Wrap Conv blocks to calls."""
        def _warped_call(self, *args):
            torch.cuda.empty_cache()
            ret_shapes = self.helper._trace_output_shape(self, *args)
            rets = ()
            for ret_shape in ret_shapes:
                rets += (self.helper.init_ret(ret_shape),)

            outputs = self.helper.compute(rets, self, *args)
            if len(outputs) == 1:
                return outputs[0]
            return outputs

        GraphModule.wraped_call = _warped_call
        for tag in self._tags:
            sub_gm = getattr(self._splitted, tag)
            sub_gm.helper = self
            self._splitted.delete_submodule(tag)
            setattr(self._splitted, tag, sub_gm.wraped_call)

    def compute(self, rets, func, *args):
        """Compute function.

        The abstract function for compute one layer convolution. Inside the inference
        function, the compute function is used for compute the message for next layer
        tensors. Users can override this function for their customize requirements.

        Parameters
        ----------
        inference_graph : DGLHeteroGraph
            The graph object for computation.
        rets : Tuple[Tensors]
            The predefined output tensors for this layer.
        layer : dgl.utils.split.schema.GraphLayer
            The layer information in the schema.
        func : Callable
            The function for computation.
        """
        raise NotImplementedError()

    def before_inference(self, graph, *args):
        """What users want to do before inference.

        Parameters
        ----------
        graph : DGLHeteroGraph
            The graph object.
        args : Tuple
            The input arguments, same as ``inference`` function.
        """
        pass

    def after_inference(self):
        """What users want to do after inference."""
        pass

    def init_ret(self, shape):
        """The initization for ret.

        Users can override it if customize initization needs. For example use numpy memmap.

        Parameters
        ----------
        shape : Tuple[int]
            The shape of output tensors.

        Returns
        Tensor : Output tensor (empty).
        """
        return torch.zeros(shape)

    def inference(self, inference_graph, *args):
        """The inference function.

        Call the inference function can conduct the layer-wise inference computation.

        inference_graph : DGLHeteroGraph
            The input graph object.
        args : Tuple
            The input arguments, should be the same as module's forward function.
        """
        self.before_inference(inference_graph, *args)
        self._input_graph = None
        self._mock_graph = None
        if self._use_uva:
            print('Uses UVA')
            for k in list(inference_graph.ndata.keys()):
                inference_graph.ndata.pop(k)
            for k in list(inference_graph.edata.keys()):
                inference_graph.edata.pop(k)

        outputs = self._splitted(inference_graph, *args)

        self.after_inference()

        return outputs


class InferenceHelper(InferenceHelperBase):
    """The InferenceHelper class.

    To construct an inference helper for customized requirements, users can extend the
    InferenceHelperBase class and write their own compute function (which can refer the
    InferenceHelper's implementation).

    Parameters
    ----------
    root : torch.nn.Module
        The model to conduct inference.
    batch_size : int
        The batch size for dataloader.
    device : torch.device
        The device to conduct inference computation.
    num_workers : int
        Number of workers for dataloader.
    use_uva : bool
        Whether store graph and tensors in UVA.
    debug : bool
        Whether display debug messages.
    """
    def __init__(self, root: nn.Module, conv_modules, batch_size, device,\
                 num_workers = 4, debug = False):
        super().__init__(root, conv_modules, device, debug=debug)
        self._batch_size = batch_size
        self._num_workers = 0 

    def compute(self, rets, func, *args):
        """Compute function.

        The basic compute function inside the inference helper. Users should not call this
        function on their own.

        Returns
        ----------
        Tuple[Tensors] : Output tensors.
        """
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            self._input_graph,
            torch.arange(self._input_graph.number_of_nodes()).to(self._input_graph.device),
            sampler,
            batch_size=self._batch_size,
            device=self._device if self._num_workers == 0 else 'cpu',
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers)
    

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            new_args = get_new_arg_input(args, input_nodes, blocks[0], self._device)

            output_vals = func(*new_args)
            del new_args

            rets = update_ret_output(output_vals, rets, output_nodes, blocks)
            del output_vals

        return rets





class Splitter():
    """The function generator class.

    Can split the forward function to layer-wise sub-functions.
    """
    def compute_message_degree(self, traced: GraphModule):
        """Compute message degrees."""
        # Set message degree to zero.
        for node in traced.graph.nodes:
            node.message_degree = 0
        for node in traced.graph.nodes:
            for user in node.users:
                user.message_degree = max(user.message_degree, node.message_degree + node.is_conv)
        # Fixed node that do not tagged (e.g., g.number_of_nodes()).
        for node in traced.graph.nodes.__reversed__():
            for arg in node.all_input_nodes:
                if arg.op == PLACEHOLDER or arg.is_conv:
                    continue
                if arg.message_degree != node.message_degree:
                    arg.message_degree = node.message_degree
        # Remove the last layer.
        output_message = max([node.message_degree for node in traced.graph.nodes])
        for node in traced.graph.nodes:
            if node.message_degree == output_message:
                node.message_degree -= 1
        
                # Print node type and message degree
        for node in traced.graph.nodes:
            print(f"Node Type: {node.is_conv}, Operation: {node.op}, Message Degree: {node.message_degree}")

    def tag_by_message_degree(self, traced):
        """Tag according to the message degrees."""
        tags = []
        for node in traced.graph.nodes:
            if node.op == PLACEHOLDER or node.op == OUTPUT:
                continue
            node.tag = CONV_BLOCK + str(node.message_degree)
            if node.tag not in tags:
                tags.append(node.tag)
        return tags

    def split(self, traced: GraphModule):
        """The split function."""
        self.compute_message_degree(traced)
        # TODO: Input bindings could be done here.
        tags = self.tag_by_message_degree(traced)
        splitted = split_by_tags(traced, tags)
        return tags, splitted

def blocks_to_graph(graph: Graph):
    """Transform blocks to a graph."""
    graph_list = None
    for node in graph.nodes:
        if node.is_conv:
            graph_obj = node.args[0]
            if graph_obj.op == CALL_FUNCTION and graph_obj.target == operator.getitem:
                graph_list = graph_obj.args[0]
                break
    if graph_list is not None:
        for node in graph.nodes:
            if node.op == CALL_FUNCTION and node.target == operator.getitem \
                and node.args[0] == graph_list:
                node.replace_all_uses_with(graph_list)
                graph.erase_node(node)
        graph.lint()

def split_module(traced: GraphModule, debug=True):
    """The module split function.

    Split the forward function of the input module.
    """
    if debug:
        print("-------- Origin forward function -------")
        print(traced.code.strip())
        print("-"*40)

    blocks_to_graph(traced.graph)
    traced.recompile()

    if debug:
        print("------- Modified forward function ------")
        print(traced.code.strip())
        print("-"*40)

    splitter = Splitter()
    tags, splitted = splitter.split(traced)

    if debug:
        print("------------ Main function -------------")
        print(splitted.code.strip())
        print("-"*40)
        for layer_id, tag in enumerate(tags):
            print("--------- Layer {} conv function --------".format(layer_id))
            print(getattr(splitted, tag).code.strip())
            print("-"*40)

    print('Tags' , tags)
    print('Split ' , splitted)

    return tags, splitted




class DGLTracer(Tracer):
    """The DGL Tracer Class. Extended from torch.fx.tracer.
    The DGL Tracer can trace a nn.module forward function to a computation graph.
    Arguments are the same as `torch.fx.tracer`.
    Parameters
    ----------
    autowrap_modules : Tuple[ModuleType]
        Defaults to `(math, )`, Python modules whose functions should be wrapped automatically
        without needing to use fx.wrap(). Backward-compatibility for this parameter is guaranteed.
    autowrap_function : Tuple[Callable, ...]
        Python functions that should be wrapped automatically without needing to use fx.wrap().
        Backward compabilibility for this parameter is guaranteed.
    param_shapes_constant : bool
        When this flag is set, calls to shape, size and a few other shape like attributes of a
        module's parameter will be evaluted directly, rather than returning a new Proxy value for
        an attribute access. Backward compatibility for this parameter is guaranteed.
    """
    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules = (math, ),
                 autowrap_functions = (),
                 param_shapes_constant = False):
        self.graph_proxy = None
        self.conv_modules = dgl.nn.conv.__dict__["__all__"]
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def set_conv_modules(self, modules):
        """Set Conv modules."""
        if isinstance(modules, (list, tuple)):
            for module in modules:
                self.set_conv_module(module)
        else:
            self.set_conv_module(modules)

    def set_conv_module(self, module):
        """Set Conv module."""
        is_module = False
        for clazz in module.__mro__:
            if clazz == torch.nn.modules.module.Module:
                is_module = True
        if not is_module:
            raise Exception("Conv Modules must be torch.nn.module.")
        self.conv_modules.append(module.__name__)

    @compatibility(is_backward_compatible=True)
    def call_module(self, m: torch.nn.Module, forward, args, kwargs):
        """Call modules."""
        def tag_conv_fn(node):
            node.is_conv = True
            return Proxy(node)

        if m.__class__.__name__ in self.conv_modules:
            module_qualified_name = self.path_of_module(m)
            return self.create_proxy('call_module', module_qualified_name, args, kwargs, \
                proxy_factory_fn=tag_conv_fn)
        super().call_module(m, forward, args, kwargs)


@compatibility(is_backward_compatible=True)
def dgl_symbolic_trace(root, conv_modules = (), concrete_args=None):
    """DGL symbolic trace function.
    We use this function to trace the nn.module to a computation graph. The output is
    a `torch.fx.GraphModule` object.

    Parameters
    ----------
    root : nn.Module
        Module or function to be traced and converted into a Graph representation.
    conv_modules : tuple
        The conv modules that we do not enter.
    concrete_args : Optional[Dict[str, any]]
        Inputs to be partially specialized.
    Returns
    -------
    GraphModule
        a Module created from the recorded operations from ``root``.
    """
    tracer = DGLTracer()
    tracer.set_conv_modules(conv_modules)
    graph = tracer.trace(root, concrete_args)
    for node in graph.nodes:
        if not hasattr(node, "is_conv"):
            node.is_conv = False

    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    graph_module = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(graph_module, key):
            setattr(graph_module, key, getattr(root, key))
    return graph_module



def get_new_arg_input(args, input_nodes, inference_graph, device, use_uva=False):
    """
    Get the new argument inputs indexed by input_nodes.

    Parameters:
    ----------
    args : list
        List of arguments, which can include tensors and graphs.
    input_nodes : torch.Tensor
        Input nodes generated by dataloader.
    inference_graph : DGLHeteroGraph
        DGL graph object.
    device : torch.device
        Device to compute inference.
    use_uva : bool
        Optional; whether to use Universal Virtual Addressing (UVA) for memory access.

    Returns:
    ----------
    tuple:
        Arguments for inference computation, with tensors and graphs properly placed on the specified device.
    """
    new_args = ()
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if use_uva:
                # Assume gather_pinned_tensor_rows is a custom function for handling UVA.
                # Ensure `arg` is a pinned memory tensor if using UVA.
                new_args += (gather_pinned_tensor_rows(arg, input_nodes),)
            else:
                # Move tensor to device if not already there, then index.
                if arg.device != device:
                    arg = arg.to(device)
                new_args += (arg[input_nodes],)
        elif isinstance(arg, DGLHeteroGraph):
            # Ensure the graph is moved to the target device.
            new_args += (inference_graph.to(device),)
    return new_args

def update_ret_output(output_vals, rets, output_nodes, blocks):
    """Update output rets.

    Parameters
    ----------
    output_vals : tuple[torch.Tensor] or torch.Tensor
        Output values.
    rets : Tuple[torch.Tensor]
        Tensor holders for outputs.
    output_nodes : torch.Tensor
        Output nodes generated by dataloader.
    blocks : list[Blocks]
        Blocks generated by dataloader.

    Returns
    ----------
    Tuple : rets.
    """
    if not isinstance(output_vals, tuple):
        output_vals = (output_vals,)
    for output_val, ret in zip(output_vals, rets):
        if output_val.size()[0] == blocks[0].num_dst_nodes():
            update_out_in_chunks(ret, output_nodes, output_val)
        else:
            raise RuntimeError("Can't determine return's type.")
    return rets

def update_out_in_chunks(ret, idx, val):
    """Update output in chunks.

    In pytorch implementation, transfer speed greatly decrease if the
    tensor size larger than 2^25. Here we update them in chunks to
    accelerate it.
    """
    memory_comsuption = 4 # float, TODO
    for dim in range(1, len(val.shape)):
        memory_comsuption *= val.shape[dim]
    num_nodes = val.shape[0]
    num_node_in_chunks = (2**25) // memory_comsuption
    start, end = 0, 0
    while start < num_nodes:
        end = min(start + num_node_in_chunks, num_nodes)
        ret[idx[start:end]] = val[start:end].cpu()
        start = end
