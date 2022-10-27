import json
import os
import sys

APP_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_BASE_PATH)
sys.path.append("/opt/tritonserver/backends/python")

from glob import glob

import numpy as np
import sodaflow
import torch
import triton_python_backend_utils as pb_utils
from mmdet.apis import inference_detector, init_detector

# sodaflow tracking
from omegaconf import DictConfig
from sodaflow import api as soda_api


class TritonPythonModel(soda_api.SodaPythonModel):
    # def initialize(self, args):
    #     self.model_config = json.loads(args["model_config"])
    #     output_img_config = pb_utils.get_output_config_by_name(
    #         self.model_config, "OUTPUT__0"
    #     )
    #     self.output_dtype = pb_utils.triton_string_to_numpy(
    #         output_img_config["data_type"]
    #     )
    #     self.device = torch.device("cuda")
    #     if os.environ.get("TEST"):
    #         self.weight_path = "/artifacts/twincar-damage-model"
    #     saved_model = glob(f"{self.weight_path}/**/*.pth", recursive=True)[0]
    #     print("loading pretrained model from %s" % saved_model)
    #     config_path = glob(f"{self.weight_path}/**/*.py", recursive=True)[0]
    #     self.model = init_detector(config_path, saved_model, device=self.device)

    def load_model(self, args):
        self.model_config = json.loads(args["model_config"])
        output_img_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT__0"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_img_config["data_type"]
        )
        self.device = torch.device("cuda")
        if os.environ.get("TEST"):
            self.weight_path = "/artifacts/twincar-damage-model"
        saved_model = glob(f"{self.weight_path}/**/*.pth", recursive=True)[0]
        print("loading pretrained model from %s" % saved_model)
        config_path = glob(f"{self.weight_path}/**/*.py", recursive=True)[0]
        self.model = init_detector(config_path, saved_model, device=self.device)

    def execute(self, requests):
        responses = []

        for request in requests:
            image_np = soda_api.get_input_tensor_by_name(request, "INPUT__0")
            image_np = image_np.as_numpy()[0]
            # image_tensors = torch.from_numpy(image_np.as_numpy()).to(self.device)
            print(f"shape: {image_np.shape}, {image_np.dtype}")

            # batch_size = image_tensors.size(0)

            # do predict ( single image )
            results = inference_detector(self.model, image_np)
            bbox_result = []
            for cat, bboxes in enumerate(results):
                for box in bboxes:
                    x1, y1, x2, y2, conf = box
                    score = round(float(conf), 2)
                    box = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
                    poly = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2}"
                    bbox_result.append([cat, poly, box, score])
            bbox_result = np.array(bbox_result)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT__0", bbox_result.astype(self.output_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
            # torch.cuda.empty_cache()

        return responses

    def build_model_signature(self):
        self.batch_size = 8

        self.add_input("INPUT__0", 4, "TYPE_UINT8")
        self.add_output("OUTPUT__0", 3, "TYPE_STRING")

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")


@sodaflow.main(config_path="configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:
    # For Testing
    print("For export python model.")

    args = {
        "model_name": "twincar-damage-model",
        "output_path": "./output/export/pymodels",
    }

    model = TritonPythonModel(
        model_name=args["model_name"], output_path=args["output_path"]
    )
    # model.initialize()
    model.build_config_pbtxt()


if __name__ == "__main__":
    run_app()
