#!/usr/bin/env python3
"""
Abstract class for the models classes
"""

import os
import time
import logging

from openvino.inference_engine import IECore
import ngraph as ng
import numpy as np
import cv2

from utils import timer


class BaseModel:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    model_name = "BaseModel"
    model_src = None
    def __init__(self,
        model: str,
        device: str="CPU",
        batch_size: int=1
    ):
        """
        :param model: Model IR file
        :param model_name: The model name
        :param device: device name
        :param batch_size: Batch size
        """
        self.device = device
        self.batch_size = batch_size
        self.exec_network = None
        self.infer_request = None
        self.infer_request_handle = None

        self.inference_durations = []

        # Initialize the plugin
        self.plugin = IECore()
        # Read the IR as a IENetwork
        self.network = self._read_network(model)
        self.network.batch_size = self.batch_size
        
        # Input layer
        self.input_blob = next(iter(self.network.input_info))
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape
        self.output_name = next(iter(self.network.outputs))
        self.inputs_keys = list(self.network.input_info.keys())
        # logging.info(f"..... {self.output_name}  {self.output_blob}")
        # logging.info(f"===== {self.network.outputs} {self.output_shape}")

    def _read_network(self, model_structure):
        """
        Read the IR as a IENetwork
        :param model_structure (str) Model structure file path
        """
        model_weights = os.path.splitext(model_structure)[0] + ".bin"
        if not os.path.exists(model_structure):
            raise ValueError(
                "Could not Initialise the network. "
                f"The provided model structure path doens't exist: {model_structure}"
            )
        return self.plugin.read_network(
            model=model_structure,
            weights=model_weights)

    @property
    def inputs_shapes(self) -> dict:
        """ Return the shape of each of the inputs
        """
        if self.network:
            return {key: self.network.input_info[key].input_data.shape
                for key in self.inputs_keys}
        else:
            raise Exception("Model Network not properly loaded")


    def get_supported_layers(self):
        """
        Get list of supported layers of the loaded network
        """
        ng_function = ng.function_from_cnn(self.network)
        return [node.get_friendly_name()
                for node in ng_function.get_ordered_ops()]

    def get_unsupported_layers(self):
        """
        Get the list of unsupported layers by the Network
        """
        supported_layers = (self.plugin.query_network(
            self.network, self.device)).keys()

        layers = self.get_supported_layers()
        unsupported_layers = set(layers) - set(supported_layers)
        return list(unsupported_layers)

    def check_model(self):
        """
        Check the supported layers of the network
        """
        unsupported_layers = self.get_unsupported_layers()
        if unsupported_layers:
            str_layers = ', '.join(unsupported_layers)
            logging.exception(f"Following layers are not supported by "
                "the plugin for the specified device {self.device}:\n {str_layers}")

    def load_model(self):
        """
        Load the model
        """
        start_time = time.time()
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(
            network=self.network, device_name=self.device)
        self.loading_time = time.time() - start_time
        return self.exec_network.requests[0].get_perf_counts()

    def preprocess_input(self, image: np.ndarray, input_name: str=None):
        """
        Before feeding the data into the model for inference,
        we need to transform the input image
        """
        # By default the model has one single input
        if not input_name:
            input_shape = list(self.inputs_shapes.values())[0]
        else:
            input_shape = self.inputs_shapes[input_name]

        processed_image = image.copy()
        if image.shape != input_shape:
            processed_image = cv2.resize(image, (input_shape[3], input_shape[2]))
            processed_image = processed_image.transpose((2, 0, 1))
            processed_image = processed_image.reshape(1, *processed_image.shape)
        return processed_image

    def preprocess_output(self, outputs: np.ndarray, image: np.ndarray):
        """ Output pre-processing"""
        raise NotImplementedError

    def predict(self, image: np.ndarray):
        """perform prediction"""

        # could not broadcast input array from shape (1080,1920,3) into shape (1,3,384,672)
        net_input = self.preprocess_input(image)
        # infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        # infer_request_handle.wait()

        start_inference = time.time()
        output = self.exec_async_inference(net_input, 0)
        self.inference_durations.append((time.time() - start_inference))
        output = self.preprocess_output(output, image)
        return output

    @property
    def model_stats(self):
        """
        Inference time and loading time output
        of the model
        """
        return {
            "Model name": self.model_name,
            "Loading time (sec.)": round(self.loading_time, 4),
            "AVG inference time (sec.)": round(np.mean(self.inference_durations), 4)}

    def exec_async_inference(self, image, request_id=0):
        """
        Start an asynchronous inference request, given an input batch of images.
        :param batch: the batch of images
        :return : The handle of the asynchronous request
        """
        self.infer_request_handle = self.exec_network.start_async(
            request_id=request_id, inputs={self.input_blob: image})
        output = self.wait()
        return output

    def wait(self, timeout=-1):
        """
        Wait for an asynchronous request
        :return outputs:
        A dictionary that maps output layer names to
        numpy.ndarray objects with output data of the layer.
        :return openvino.inference_engine.ie_api.Blob object
        """
        status = self.infer_request_handle.wait(timeout=timeout)
        while status != 0:
            logging.info("Waiting for inference ...")
            time.sleep(2)
        res = {output_blob: self.infer_request_handle.output_blobs[output_blob].buffer
            for output_blob in self.network.outputs}
        return res

    @property
    def inference_output(self):
        """
        Extract the output results
        :param request_id: Index of the request value
        :param output: Name of the output layer
        :return
           Inference request result
        """
        # return self.infer_request_handle.output_blobs[self.output_blob].buffer
        return self.infer_request_handle.output_blobs


