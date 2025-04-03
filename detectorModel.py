from fastai.learner import load_learner
modelLocalPath = "model/model.pkl"
digImg = "/Users/nsri2/Documents/AI2 project/4442-Group-Project/data/test/Screenshot 2025-04-03 at 11.37.11 AM.png"

model = load_learner(modelLocalPath)

def detector(digImg):
    prediction = model.predict(digImg)[0]
    if prediction in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
        return int(prediction)
    elif prediction in ["x", "y", "z"]:
        return prediction
    else:
        match prediction:
            case "add":
                return "+"
            case "dec":
                return "-"
            case "div":
                return "/"
            case "eq":
                return "="
            case "mul":
                return "*"
            case "sub":
                return "-"
            
    
    return model.predict(digImg)[0]

print(detector(digImg))