# cloud-Serfers
High on the cloud life

# Contributers
## Martin Zhelyazkov
## Ravin Shalmashi
## Paschal Chukwubuikem Ifewulu

# The repository needs to be cloned and the data directory in the root of the repository has to be created so the data is stored there!!!!! <3


# AWS model
To run the aws_model.ipynb open a learner lab, create a S3 bucket upload the .parquet files then go ot Amazon Sagemaker AI and upload the notebook and run it!.



⚙️ Backend Setup (FastAPI)
    - cd backend
    - venv\Scripts\activate
    - pip install --upgrade pip
    - pip install -r requirements.txt
    - python -m uvicorn app.main:app --reload
    - http://127.0.0.1:8000/docs

POST /predict/housing 
    {
  "region": "East Midlands",
  "property_type": "D",
  "tenure": "F",
  "year": 2015,
  "month": 7,
  "is_new_build": false
}

POST /predict/electricity
    {
  "year": 2023,
  "month": 11,
  "day": 24,
  "hour": 18,
  "is_weekend": 0
}



🌐 Frontend Setup (Streamlit)
    - cd frontend
    - pip install streamlit requests
    - streamlit run app.py
    http://localhost:8501










For housing, we trained and tuned real models using PyCaret (including XGBoost, etc.), and chose our best model as the deployed one.

For electricity demand, we initially planned a separate model and did the EDA/cleaning. However, due to limited cloud credits and account deactivation issues, we couldn’t complete a separate electricity model on AWS/PyCaret in time.

To still demonstrate the deployment and API integration, our /predict/electricity endpoint reuses the deployed housing model with default feature values. This endpoint is included for technical completeness of the backend+frontend pipeline, but its predictions are not meant to be interpreted as meaningful electricity forecasts.
