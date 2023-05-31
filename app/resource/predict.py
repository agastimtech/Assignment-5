import aiohttp
import asyncio
import aiohttp_jinja2
import jinja2
import pickle
import numpy as np
import json
from app.common.utils import load_model

from aiohttp import web

# Function to predict employee retention using a saved model
def predict_retention(employee_data):
    # Load the saved model
    model = load_model()

    retention_prediction = model.predict(employee_data)

    # Return the predicted retention values
    return retention_prediction

# Mapping for Department
department_mapping = {
        'sales': 0,
        'technical': 1,
        'support': 2,
        'IT': 3,
        'product_mng': 4,
        'marketing': 5,
        'RandD': 6,
        'accounting': 7,
        'hr': 8,
        'management': 9
    }

# Mapping for Salary
salary_mapping = {
        'low': 0,
        'medium': 1,
        'high': 2
    }

# Handler for the home page
class IndexView(web.View):
    @aiohttp_jinja2.template('index.html')
    async def get(self):
        return {}

    @aiohttp_jinja2.template('predict.html')
    async def post(self):
        data = await self.request.post()

        # Convert inputs to numpy array
        inputs = np.zeros((1, 9))
        inputs[0, 0] = float(data['satisfaction_level'])
        inputs[0, 1] = float(data['last_evaluation'])
        inputs[0, 2] = int(data['number_project'])
        inputs[0, 3] = int(data['average_montly_hours'])
        inputs[0, 4] = int(data['time_spend_company'])
        inputs[0, 5] = int(data['Work_accident'])
        inputs[0, 6] = int(data['promotion_last_5years'])
        inputs[0, 7] = department_mapping[data['Department']]
        inputs[0, 8] = salary_mapping[data['salary']]

        # Make prediction using the model
        prediction = predict_retention(inputs)
        result = 'Left' if prediction[0] == 0 else 'Stayed'

        return {'result': result}






    # nothing is happening when i click on predict button on webpage..i want 'employee will leave the company' if model.predict is 0 else it should display 'employee will not leave the company' only when i click on predict

# this is the code and following is the html file
#
#
# So modify both code and html accordingly




# i have logistic regression model....nowe give me seperate code of main.py where I put inputs as satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, Department, salary
# ....where department has values such as 'sales','technical','support','IT','product_mng','marketing','RandD','accounting','hr','management' replaced by 0,1,2,3,4,5,6,7,8,9 respectively and salary has values such as
# 'low','medium','high' replaced by 0,1,2 respectively....give me seperate index.html and predict.html in templates....use aiohttp,aiohttp_jinja2,class,app.router.add_view,asyncio...you can create a numpy array of zeros to
# be replaced by input values



