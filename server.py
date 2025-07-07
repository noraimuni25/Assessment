import os
import sys
import traceback

import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware

import torch
from torchvision.models import resnet50

from inference import Inference
from schema import InferenceResponse, ErrorResponse

# Initialize API Server
app = FastAPI(
    title="AI Inference API",
    description="API for performing classification inference on images",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    # Initialize the pytorch model
    model = resnet50(pretrained=True)
    model.to('cuda')
    model.eval()
    inference_service = Inference(model)

    # add inference_service too app state
    app.package = {
        "inference_service": inference_service
    }


@app.post('/detect',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def do_detect(image_file: UploadFile = File(...)):
    """
    Perform prediction on input data
    """
    try:
        # Read the image file
        image = await image_file.read()
        result = app.package["inference_service"].predict(image)
        return result

    except ValueError as e:
        return ErrorResponse(error=True, message=str(e), traceback=traceback.format_exc())

    except Exception as e:
        logger.error(traceback.format_exc())
        return ErrorResponse(error=True, message=str(e), traceback=traceback.format_exc())


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("server:app", host="0.0.0.0", port=8082,
                reload=True)
