# cloud-Serfers
High on the cloud life

# Contributers
## Martin Zhelyazkov
## Ravin Shalmashi
## Paschal Chukwubuikem Ifewulu

# The repository needs to be cloned and the data directory in the root of the repository has to be created so the data is stored there!!!!! <3


# AWS model
To run the aws_model.ipynb open a learner lab, create a S3 bucket upload the .parquet files then go ot Amazon Sagemaker AI and upload the notebook and run it!.




For housing, we trained and tuned real models using PyCaret (including XGBoost, etc.), and chose our best model as the deployed one.

For electricity demand, we initially planned a separate model and did the EDA/cleaning. However, due to limited cloud credits and account deactivation issues, we couldn’t complete a separate electricity model on AWS/PyCaret in time.

To still demonstrate the deployment and API integration, our /predict/electricity endpoint reuses the deployed housing model with default feature values. This endpoint is included for technical completeness of the backend+frontend pipeline, but its predictions are not meant to be interpreted as meaningful electricity forecasts.