# person_reid
#Person Re-Identification over Multiple Cameras

This project implements a **Person Re-Identification System** using deep learning, allowing users to upload an image of a person and search for that individual across multiple camera feeds using `torchreid` and `Streamlit`.

## Features

- Deep Learning–based ReID with OSNet
- Multi-camera dataset support (static images)
- Clean Streamlit web app UI
- Embedding-based cosine similarity search

## Structure

person-reid-project/
├── person_reid_app.py # Main Streamlit app
├── reid_utils.py # ReID logic & helpers
├── dataset/ # Simulated camera folders
│ ├── cam1/
│ └── cam2/
├── requirements.txt
└── README.md

## steps to initialize

-git clone https://github.com/harshalp11/person_reid.git
 cd person-reid-project
 
-create a virtual enviroment:
 eg-  python -m venv venv
      venv\Scripts\activate 

-install dependencies(torchreid ans streamlit)

-run the app
streamlit run person_reid_app.py




