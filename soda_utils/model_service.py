import json
import traceback
from decimal import Decimal

import numpy as np

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
try:
    import tritonclient.grpc as grpcclient
except Exception:
    print("Triton Client not imported.")


STRING_TO_NUMPY = {
    "BOOL": np.bool,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "STRING": np.bytes_,
    "BYTES": np.object_,
}


def string_to_numpy(type_string):
    return STRING_TO_NUMPY[type_string]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class SodaModelService:
    def __init__(self, server_url, model_name, model_version=None):
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version if model_version else ""
        self.inputs = list()
        self.outputs = list()

    def add_inputs(self, inputs: list):
        for input in inputs:
            t_shape = tuple(input["shape"])
            tensor_data = np.array(
                np.random.random(t_shape), dtype=string_to_numpy(input["dtype"])
            )
            # tensor_data = np.ones(shape=t_shape, dtype=string_to_numpy(input['dtype']))
            self.add_input(
                name=input["name"],
                shape=t_shape,
                dtype=input["dtype"],
                tensor=tensor_data,
            )

    def add_outputs(self, outputs: list):
        for output in outputs:
            self.add_output(output["name"])

    def add_input(self, name, shape, dtype, tensor):
        triton_input = grpcclient.InferInput(name, shape, dtype)
        triton_input.set_data_from_numpy(tensor)
        self.inputs.append(triton_input)

    def add_output(self, name):
        self.outputs.append(grpcclient.InferRequestedOutput(name))

    def get_model_config(self):
        try:
            server_url = f"{self.server_url}:8001"
            triton_client = grpcclient.InferenceServerClient(
                url=server_url, verbose=False
            )
        except Exception as e:
            print("context creation failed: " + str(e))
            raise e

        # Metadata
        metadata = triton_client.get_model_config(self.model_name, as_json=True)
        # if not (metadata['name'] == self.model_name):
        #     raise Exception("FAILED : get_model_metadata")

        return metadata

    def get_model_meta_data(self):
        try:
            server_url = f"{self.server_url}:8001"
            triton_client = grpcclient.InferenceServerClient(
                url=server_url, verbose=False
            )
        except Exception as e:
            print("context creation failed: " + str(e))
            raise e

        # Metadata
        metadata = triton_client.get_model_metadata(
            model_name=self.model_name, model_version=self.model_version, as_json=True
        )
        if not (metadata["name"] == self.model_name):
            raise Exception("FAILED : get_model_metadata")

        return metadata

    def predict(self):
        try:
            server_url = f"{self.server_url}:8001"
            triton_client = grpcclient.InferenceServerClient(
                url=server_url, verbose=False
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            raise e

        try:
            # Test with outputs
            results = triton_client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=self.inputs,
                outputs=self.outputs,
                client_timeout=None,
            )
        except Exception as e:
            print(traceback.format_exc())
            raise e
        else:
            # out = [{output.name(): json.dumps(results.as_numpy(output.name())[:10], cls=NumpyEncoder)} for output in self.outputs]
            out = results
            return out
