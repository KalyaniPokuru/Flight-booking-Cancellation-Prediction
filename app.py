# install libraries ---
# pip install fastapi uvicorn

# 1. Library imports
import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
import pickle

# 2. Create the app object
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. load the model
rgModel = pickle.load(open("travel_pred.pkl", "rb"))

# 4. Index route, opens automatically on http://127.0.0.1:80


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get("/predictOutcome")
def gePredictOutcome(Age : int, ServicesOpted : int, FrequentFlyer_NoRecord : int, FrequentFlyer_Yes : int,AnnualIncomeClass_LowIncome : int,  AnnualIncomeClass_MiddleIncome: int, AccountSyncedToSocialMedia_Yes : int, BookedHotelOrNot_Yes : int ):

    # model.predict([[0,0,23,2017,1,35,30,2,5,2,0.0,0,0,15,5,3,0,0,0,0,0,0,394.000000,189.266735,0,2,96.14,0,0,1]])
    prediction = rgModel.predict([[ Age , ServicesOpted, FrequentFlyer_NoRecord , FrequentFlyer_Yes ,AnnualIncomeClass_LowIncome,  AnnualIncomeClass_MiddleIncome, AccountSyncedToSocialMedia_Yes, BookedHotelOrNot_Yes]])
    val = prediction[0]
    # print(val);
    # return {'Outcome': val}
    return {'message': str(val)}


# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0')
# http://127.0.0.1/predictOutcome?Age=36&ServicesOpted=1&FrequentFlyer_NoRecord=0&FrequentFlyer_Yes=1&AnnualIncomeClass_LowIncome=0&AnnualIncomeClass_MiddleIncome=0&AccountSyncedToSocialMedia_Yes=0&BookedHotelOrNot_Yes=0
# 100.0,3000.0,5,0.0,0.0,0,0,1,1
# 500.0	26000.0	0	        800.0	677.2	6	0	1	1
# 100.0,4300.0,5,0.0,0.0,0,1
