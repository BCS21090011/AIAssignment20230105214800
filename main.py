import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sbrn

class ProcessDataset():
    def __init__(self, filepath):
        self.FileName = filepath
        self.OriData = pd.read_csv(filepath)
        self.ColsNeeded = ["Hospt", "Treat", "Outcome", "AcuteT", "Age", "Gender"]
        self.NumericCols = ["Hospt", "AcuteT", "Age", "Gender"]

        self.ValidataFile()
        if self.FileValid == True:
            self.Process()

    def ValidataFile(self)->bool:
        self.FileValid = False

        if set(self.ColsNeeded).issubset(self.OriData.columns):
            self.OutcomeValValid = False
            self.NumericColsValValid = True

            if self.OriData["Outcome"].unique().all() in ["No Recurrence", "Recurrence"]:
                self.OutcomeValValid = True

            for col in self.NumericCols:
                if is_numeric_dtype(self.OriData[col]) == False:
                    self.NumericColsValValid = False

            if ((self.OutcomeValValid == True) & (self.NumericColsValValid == True)):
                self.FileValid = True

    def Process(self):
        # Drop unneeded column:
        self.DropUnneededCols()

        # Mapping:
        self.OutcomeMap = {"No Recurrence": 0, "Recurrence": 1}
        self.Data["Outcome"] = self.Data["Outcome"].map(self.OutcomeMap)

        # Somehow one hot encoding(?):
        self.TreatColDum = pd.get_dummies(data=self.Data["Treat"])
        self.Data.drop("Treat", axis=1, inplace=True)
        self.Data = self.Data.join(self.TreatColDum)

        # Defining output column and features:
        self.OutCol = "Outcome"
        self.Features = self.Data.columns.tolist()
        self.Features.remove(self.OutCol)

        # Setting X and y:
        self.X = self.Data[self.Features].values
        self.y = self.Data[self.OutCol].values

        # Splitting:
        self.Xtrn, self.Xtst, self.ytrn, self.ytst = train_test_split(self.X, self.y, test_size=0.25, random_state=0)

        # Setting decision tree:
        self.Classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
        self.Classifier.fit(self.Xtrn, self.ytrn)

        # Confusion matrix:
        self.yPred = self.Classifier.predict(self.Xtst)
        self.PredAccuracy = accuracy_score(self.ytst, self.yPred)
        self.CM = confusion_matrix(self.ytst, self.yPred)

    def DropUnneededCols(self):
        unNeededBoolArray = ~(self.OriData.columns.isin(self.ColsNeeded))
        self.Data = self.OriData.drop(self.OriData.columns[unNeededBoolArray], axis=1)

    def Predict(self, inputs)->int:
        Xinputs = [inputs]
        pred = self.Classifier.predict(Xinputs)[0]

        return pred

    def CMSaveFig(self, fileName):
        sbrn.heatmap(self.CM, annot=True)
        plt.savefig(fileName)
        plt.clf()

    def DTSaveFig(self, saveName):
        tree.plot_tree(self.Classifier, feature_names=self.Features)
        plt.savefig(saveName)
        plt.clf()

    def __str__(self)->str:
        strOutput = f"FileName: {self.FileName: ^16} Valid: {self.FileValid}"
        return strOutput

# ________________Interface________________

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image

fileValid = False
inputsValid = False
TreatColVals = ["Nothing"]
de: ProcessDataset

def ValidLabelOutput(lbl, valid, oriText, invalidText=None):
    if valid == True:
        lbl["text"] = oriText
    else:
        if invalidText == None:
            lbl["text"] = oriText + "\t[Invalid]"
        else:
            lbl["text"] = invalidText

def PredButton_OnClick():
    ValidLabelOutput(BrowseFileLabel, fileValid, "Browse file")
    if fileValid == True:
        HosptValStr = HosptEntry.get()
        AcuteTValStr = AcuteTEntry.get()
        AgeValStr = AgeEntry.get()
        GenderValStr = GenderStrVar.get()

        HosptValValid = ValidStrInputIsIntNonNeg(HosptValStr)
        AcuteTValValid = ValidStrInputIsIntNonNeg(AcuteTValStr)
        AgeValValid = ValidStrInputIsIntNonNeg(AgeValStr)

        ValidLabelOutput(HosptLabel, HosptValValid, "Hospital ID:")
        ValidLabelOutput(AcuteTLabel, AcuteTValValid, "Depressed Time (days):")
        ValidLabelOutput(AgeLabel, AgeValValid, "Age:")

        if all([HosptValValid, AcuteTValValid, AgeValValid]) == True:
            global inputsValid
            inputsValid = True

            genderValsDict = {"Male": 2, "Female": 1}

            Hospt = int(HosptValStr)
            AcuteT= int(AcuteTValStr)
            Age = int(AgeValStr)
            Gender = int(genderValsDict[GenderValStr])
            Treat = TreatStrVar.get()

            TreatValsDict = {val: 0 for val in TreatColVals}
            TreatValsDict[Treat] = 1

            userInputsDict = {"Hospt": Hospt, "AcuteT": AcuteT, "Age": Age, "Gender": Gender}
            userInputsDict.update(TreatValsDict)

            userInputs = []

            for feat in de.Features:
                userInputs.append(userInputsDict[feat])

            predResult = de.Predict(userInputs)
            reversedOutcomeMap = {v: k for k, v in de.OutcomeMap.items()}
            predResultStr = reversedOutcomeMap[predResult]

            messagebox.showinfo(title="Prediction result:", message=f"Prediction result: {predResultStr:^16}")

def BrowseFileButton_OnClick():
    global fileValid
    global de

    fileName = filedialog.askopenfilename(title="Select a hospital data file (.csv)", filetypes=[("CSV files", "*.csv*")])
    BrowseFileAccuracyLabel["text"] = ''
    DiagramPanel.config(image=None)
    DiagramPanel.image = None

    if fileName == '':
        fileValid = False
    else:
        de = ProcessDataset(fileName)
        fileValid = de.FileValid
        if de.FileValid == True:
            global TreatColVals
            TreatColVals = de.TreatColDum.columns.tolist()
            BrowseFileAccuracyLabel["text"] = f"Prediction accuracy: {de.PredAccuracy * 100:^8} %"

            ResetOptionMenuOptions(TreatOptMenu, TreatStrVar, TreatColVals)

            de.CMSaveFig("CMatrix.png")
            de.DTSaveFig("DTree.png")
            de.DTSaveFig("DTree.svg")
            photo = ImageTk.PhotoImage(Image.open("DTree.png"))
            DiagramPanel.config(image=photo)
            DiagramPanel.image = photo

    ValidLabelOutput(BrowseFileLabel, fileValid, "Browse file")

def ResetOptionMenuOptions(optMenu, strVar, newOptions, defaultValIndex=None):
    menu = optMenu["menu"]
    menu.delete(0, "end")

    for opt in newOptions:
        menu.add_command(label=opt, command=tk._setit(strVar, opt))

    if defaultValIndex == None:
        defaultValIndex = 0

    if (defaultValIndex >= len(newOptions)) | (defaultValIndex < 0):
        defaultValIndex = 0

    strVar.set(newOptions[defaultValIndex])

def ValidStrInputIsFloatNonNeg(userInput)->bool:
    try:
        inVal = float(userInput)

        # Codes below won't run if float(userInput) have an error:
        # Check if user input is negative:
        if inVal < 0.0:
            return False
        else:
            return True

    except ValueError:
        return False

def ValidStrInputIsIntNonNeg(userInput)->bool:
    if userInput.find('.') == -1:
        return ValidStrInputIsFloatNonNeg(userInput)
    else:
        return False

def ValidEntryValFloatNonNeg(userInput)->bool:
    valid = ValidStrInputIsFloatNonNeg(userInput)

    if userInput == '':
        valid = True

    return valid

def ValidEntryValIntNonNeg(userInput)->bool:
    valid = ValidStrInputIsIntNonNeg(userInput)

    if userInput == '':
        valid = True

    return valid

def CustomEntry(root, validateCommand, entryDefaultVal=None)->tk.Entry:
    entry = tk.Entry(root, justify="center")
    entry.config(validate="key", validatecommand=(entry.register(validateCommand), '%P'))
    entryTxtVar = tk.StringVar(root)

    if entryDefaultVal != None:
        if validateCommand(entryDefaultVal) == True:
            entryTxtVar.set(entryDefaultVal)
            entry.config(textvariable=entryTxtVar)

    return entry

def CustomOptionMenu(root, options, defaultValIndex=None)->(tk.OptionMenu, tk.StringVar):
    if defaultValIndex == None:
        defaultValIndex = 0

    if (defaultValIndex >= len(options)) | (defaultValIndex < 0):
        defaultValIndex = 0

    defaultVal = tk.StringVar(root)
    defaultVal.set(options[defaultValIndex])
    optMenu = tk.OptionMenu(root, defaultVal, *options)

    return (optMenu, defaultVal)

root = tk.Tk()
root.title("Depression Treatment Prediction")
root.geometry("800x700")
root.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), weight=0)
root.grid_columnconfigure((0, 1, 2), weight=1)

BrowseFileLabel = tk.Label(root, text="Browse file", justify="center")
BrowseFileLabel.grid(column=0, row=0)
BrowseFileAccuracyLabel = tk.Label(root, justify="center")
BrowseFileAccuracyLabel.grid(column=1, row=0)
BrowseFileButton = tk.Button(root, text="Browse file", command=BrowseFileButton_OnClick)
BrowseFileButton.grid(column=2, row=0)

HosptLabel = tk.Label(root, text= "Hospital ID:", justify="center")
HosptLabel.grid(column=0, row=1)
HosptEntry = CustomEntry(root, ValidEntryValIntNonNeg)
HosptEntry.grid(column=2, row=1)

AcuteTLabel = tk.Label(root, text="Depressed Time (days):", justify="center")
AcuteTLabel.grid(column=0, row=2)
AcuteTEntry = CustomEntry(root, ValidEntryValIntNonNeg, "0")
AcuteTEntry.grid(column=2, row=2)

AgeLabel = tk.Label(root, text="Age:", justify="center")
AgeLabel.grid(column=0, row=3)
AgeEntry = CustomEntry(root, ValidEntryValIntNonNeg)
AgeEntry.grid(column=2, row=3)

GenderLabel = tk.Label(root, text="Gender:", justify="center")
GenderLabel.grid(column=0, row=4)
genderVals = ["Male", "Female"]
GenderOptMenu, GenderStrVar = CustomOptionMenu(root, genderVals)
GenderOptMenu.grid(column=2, row=4)

TreatLabel = tk.Label(root, text="Treat:", justify="center")
TreatLabel.grid(column=0, row=5)
TreatOptMenu, TreatStrVar = CustomOptionMenu(root, TreatColVals)
TreatOptMenu.grid(column=2, row=5)

PredButton = tk.Button(root, text="Predict", command=PredButton_OnClick)
PredButton.grid(column=1, row=6)

DiagramPanel = tk.Label(root)
DiagramPanel.grid(column=0, columnspan=3, row=7, rowspan=3)

root.mainloop()
