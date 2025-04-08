from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import os
import pickle
from tda_pipelines_PCA import VECTOR_STITCHING_PI_Pipeline_RGB
from data_preprocessing import load_cifar10_batch
from tempfile import NamedTemporaryFile

app = FastAPI()
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.post("/process")
async def process_data(
    file: UploadFile = File(...),
    binarizer_threshold: float = Form(0.35),
    sig: float = Form(0.3)
):
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        raw_data, labels = load_cifar10_batch(temp_file_path)
        input_height, input_width = raw_data.shape[1:3]

        pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
            binarizer_threshold=binarizer_threshold,
            sig=sig,
            input_height=input_height,
            input_width=input_width
        )

        processed_data = pipeline.fit_transform(raw_data)

        processed_file_path = os.path.join(
            PROCESSED_FOLDER, f"processed_{filename}.npz")
        np.savez_compressed(processed_file_path,
                            data=processed_data, labels=labels)

        return FileResponse(processed_file_path, filename=f"processed_{filename}.npz")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
