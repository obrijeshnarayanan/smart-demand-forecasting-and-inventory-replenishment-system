import joblib

model = joblib.load("improved_demand_prediction_model.pkl")

for name, step in model.steps:
    print(name, "=>", type(step))

print("\nMAIN ALGORITHM:")
print(type(model.steps[-1][1]))