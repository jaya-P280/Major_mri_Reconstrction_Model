# PATOMIR: Probabilistic Anatomical Transformer for Optimized MRI Reconstruction

This repository contains the implementation of **PATOMIR**, a deep learning-based framework for **MRI reconstruction from under-sampled data**.

The system reconstructs high-quality MRI images from **k-space MRI data stored in `.h5` format** using a trained neural network model.

This project was developed as part of a **Major Project**.

---

# Project Overview

Magnetic Resonance Imaging (MRI) reconstruction is a critical process in medical imaging. Traditional reconstruction techniques require large sampling data and computational resources.

PATOMIR introduces a **deep learning-based reconstruction pipeline** that reconstructs MRI images from **under-sampled k-space data** stored in `.h5` files. The model improves reconstruction quality while reducing the amount of required sampling.

The project includes a **FastAPI-based inference API** that allows users to upload MRI `.h5` files and obtain reconstructed MRI images.

---

# Repository Structure

```id="f3a0s1"
Major_mri_Reconstrction_Model/
│
├── app.py                 # FastAPI server for prediction
├── model.py               # Model architecture and loading
├── utils.py               # MRI reconstruction utilities
├── requirements.txt       # Project dependencies
├── reconstructed.png      # Example reconstructed MRI output
├── zero_filled.png        # Example zero-filled MRI image
└── README.md              # Project documentation
```

---

# Trained Model

The trained model file is larger than GitHub's file size limit and therefore cannot be uploaded directly to the repository.

Download the trained model from Google Drive:

https://drive.google.com/file/d/1bV6Un-KpgenzaJfzHUGFD04RI7y94zjy/view

After downloading, place the model file inside the project directory:

```id="c9ru0j"
Major_mri_Reconstrction_Model/
│
├── trained_model.pt
├── app.py
├── model.py
└── utils.py
```

---

# Installation

Clone the repository:

```id="k0qu0g"
git clone https://github.com/jaya-P280/Major_mri_Reconstrction_Model.git
cd Major_mri_Reconstrction_Model
```

Create a virtual environment:

```id="e48qcc"
python -m venv venv
```

Activate the environment:

Windows

```id="xhyzjn"
venv\Scripts\activate
```

Linux / Mac

```id="9zxy0r"
source venv/bin/activate
```

Install required dependencies:

```id="kh36p0"
pip install -r requirements.txt
```

---

# Running the FastAPI Server

Start the FastAPI server:

```id="nprgo0"
uvicorn app:app --reload
```

The API server will run at:

```id="ggutn7"
http://127.0.0.1:8000
```

Interactive API documentation:

```id="99m0hu"
http://127.0.0.1:8000/docs
```

---

# API Usage

### Endpoint

```
POST /predict/
```

### Input

* MRI **`.h5` file** containing k-space MRI data.

### Output

* Reconstructed MRI image.

Example request using curl:

```id="udshx5"
curl -X POST "http://127.0.0.1:8000/predict/" \
-F "file=@sample_mri_data.h5"
```

---

# Example Results

The repository includes sample output images:

* **zero_filled.png** → MRI image reconstructed using basic zero-filling.
* **reconstructed.png** → MRI image reconstructed using the PATOMIR deep learning model.

These demonstrate the improved reconstruction quality of the model.

---

# Technologies Used

* Python
* PyTorch
* FastAPI
* NumPy
* OpenCV
* h5py
* Uvicorn

---

# Project Team

**Project Title**

PATOMIR: Probabilistic Anatomical Transformer for Optimized Medical Image Reconstruction

**Team Members**

* P. Spandith
* P. Jaya Prakash Goud

**Project Supervisor**

Dr. V. Bharathi

---

# Future Improvements

* Web interface for uploading `.h5` MRI data
* Docker deployment
* Real-time reconstruction visualization
* Integration with hospital imaging systems

---

# License

This project is developed for academic purposes as part of a major project.
