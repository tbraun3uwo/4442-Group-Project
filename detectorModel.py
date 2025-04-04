from fastai.learner import load_learner
from PIL import Image
modelLocalPath = "model/model.pkl"

model = load_learner(modelLocalPath)

def detector(digImg):
    digImg = Image.open(digImg)
    digImg = digImg.convert("L")
    digImg = digImg.resize((25, 25))
    prediction = model.predict(digImg)[0]
    if prediction in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
        return int(prediction)
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
