import pyrebase

firebase_config = {
    "apiKey": "AIzaSyBbdk5jF2hKAo0zcPuec4GwQsPL0MBYkIg",
    "authDomain": "budgetbuddy-ac47e.firebaseapp.com",
    "projectId": "budgetbuddy-ac47e",
    "storageBucket": "budgetbuddy-ac47e.appspot.com",
    "messagingSenderId": "23103361616",
    "appId": "1:23103361616:web:5c8baf11580a1138e93755",
    "measurementId": "G-F35EMDZ7TB",
    "databaseURL": ""
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()



