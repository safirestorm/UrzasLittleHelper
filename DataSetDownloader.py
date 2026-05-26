import roboflow

roboflow.login()

rf = roboflow.Roboflow()
project = rf.workspace("mikkels-workspace-aejbo").project("mtg2")
version = project.version(3)
dataset = version.download("yolov8")