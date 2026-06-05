from predictor import predict

results = predict("CCO")

for k, v in results.items():
    print(f"{k}: {v}")
print(predict("CCO"))        # Ethanol
print(predict("c1ccccc1"))   # Benzene
print(predict("CC"))         # Ethane