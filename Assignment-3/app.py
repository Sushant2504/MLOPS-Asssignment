import requests
import math
import json
import sys
import streamlit as st


def get_weather(city):
    api_key = "b3c62ae7f7ad5fc3cb0a7b56cb7cbda6"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors
    except requests.exceptions.HTTPError as err:
        print(f"Error: {err}")
    
    try:
        data = json.loads(response.text)
        if data['cod'] != 200:
            print(f"Error: {data['message']}")
    except json.JSONDecodeError as err:
        print(f"Error: Failed to parse response JSON - {err}")
        
    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)
    
    # Print the weather forecast
    st.write(f"Weather in {city}: {weather_description}")
    st.write(f"Temperature: {temperature}°C")
    st.write(f"Humidity: {humidity}")
    st.write(f"Pressure: {pressure}")

st.header("Check the current weather of the Any City")
st.image("https://images.theconversation.com/files/442675/original/file-20220126-17-1i0g402.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=1356&h=668&fit=crop")
city = st.text_input("Enter city name")
if st.button("Get Weather"):
    get_weather(city)

    