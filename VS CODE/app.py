from flask import Flask
import ghhops_server as hs
import rhino3dm
import myml

app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
     "/ml_prediction3",
     name="ML_prediction",
     description="Get the features, output the predictions",
     icon="smaller.png",
     inputs=[
         hs.HopsNumber("Min_Depth", "Min_Depth", "First Metric"),
         hs.HopsNumber("Max_Depth", "Max_Depth", "Second Metric"),
         hs.HopsNumber("Min_Heigth", "Min_Heigth", "Second Metric"),
         hs.HopsNumber("Max_Heigth", "Max_Heigth", "Second Metric"),
         hs.HopsNumber("Volume", "Volume", "Second Metric"),
         hs.HopsNumber("Area", "Area", "Second Metric"),
         hs.HopsNumber("Green_area", "Green_area", "Second Metric"),
         hs.HopsNumber("FAR", "FAR", "Second Metric"),
     ],
     outputs=[
         hs.HopsNumber("radiation", "radiation", "ml pedricted values"),
         hs.HopsNumber("green_comfort", "green_comfort", "ml pedricted values"),
         hs.HopsNumber("energy", "energy", "ml pedricted values"),
     ]
)

def ml_prediction3(Min_Depth, Max_Depth, Min_Heigth, Max_Heigth, Volume, Area, Green_area, FAR):
    prediction = myml.predictions(Min_Depth, Max_Depth, Min_Heigth, Max_Heigth, Volume, Area, Green_area, FAR)

    radiation = prediction[0]
    green_comfort = prediction[1]
    energy = prediction[2]

    return radiation, green_comfort, energy


if __name__ == "__main__":
    app.run()