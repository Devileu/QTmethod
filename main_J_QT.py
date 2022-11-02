import J_QT_method
# import J_QT_method as QT



# Drift detectors

# switch = ['qt', 'adwin', 'ddm', 'eddm', 'ph']
switch = ['qt', 'eddm']

# Stream File path
file = "C:/Users/Devileu/Desktop/Dataset/MIXED.csv"

# Stream File Information
detectiondelay = [1000]
driftposition = [5000]
fullsize = 10000


J_QT_method.QT(switch, file, detectiondelay, driftposition, fullsize)

