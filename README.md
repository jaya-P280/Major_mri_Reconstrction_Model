# PATOMIR: Probabilistic Anatomical Transformer for Optimized MRI Reconstruction

This repository contains the implementation of **PATOMIR**, a deep learning-based framework for **MRI image reconstruction**. The system reconstructs high-quality MRI images from under-sampled inputs using a trained neural network model.

This project was developed as part of a **Major Project**.

---

# Project Overview

Magnetic Resonance Imaging (MRI) reconstruction plays a critical role in medical imaging. Traditional reconstruction techniques can be slow and sometimes produce lower-quality outputs when data is incomplete.

PATOMIR leverages **deep learning techniques** to reconstruct MRI images more efficiently and accurately. The system processes MRI inputs and produces reconstructed images that improve visual quality and diagnostic usability.

The project also includes a **FastAPI backend** that allows users to upload MRI images and receive reconstructed outputs.

---

# Repository Structure

```
Major_mri_Reconstrction_Model/
│
├── app.py                 # FastAPI server for prediction
├── model.py               # Model architecture and loading
├── utils.py               # Inference and reconstruction utilities
├── requirements.txt       # Project dependencies
├── reconstructed.png      # Example reconstructed output
├── zero_filled.png        # Example zero-filled input
└── README.md              # Project documentation
```

---

# Trained Model

The trained model file is large and cannot be uploaded directly to GitHub due to GitHub's **100MB file size limitation**.

Download the trained model from Google Drive:

https://drive.google.com/file/d/1bV6Un-KpgenzaJfzHUGFD04RI7y94zjy/view

After downloading, place the model file inside the project directory:

```
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

```
git clone https://github.com/jaya-P280/Major_mri_Reconstrction_Model.git
cd Major_mri_Reconstrction_Model
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

Install required dependencies:

```
pip install -r requirements.txt
```

---

# Running the FastAPI Server

Start the API server using:

```
uvicorn app:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

# API Usage

Endpoint:

```
POST /predict/
```

Input:

* MRI image file

Output:

* Reconstructed MRI image

Example using curl:

```
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@test_image.png"
```

---

# Technologies Used

* Python
* PyTorch
* FastAPI
* NumPy
* OpenCV
* Uvicorn

---

# Example Results

The repository includes sample images:

* **zero_filled.png** – Input MRI image
* **reconstructed.png** – Output reconstructed MRI image

These demonstrate the effectiveness of the reconstruction model.

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

* Add web interface for uploading MRI images
* Deploy the system using Docker
* Improve reconstruction quality using advanced transformer architectures
* Integrate real-time MRI visualization tools

---

# License

This project was developed for academic purposes as part of a major project.
