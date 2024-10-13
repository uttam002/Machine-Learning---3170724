from statistics import stdev,variance 
def calculateStatistics(data): 
stdDeviation = stdev(data) 
dataVariance = variance(data) 
return stdDeviation,dataVariance 
data1 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2] 
data2 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2000] 
data3 = [1, 2, 3, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000] 
data1stdDeviation,data1Variance = calculateStatistics(data1) 
data2stdDeviation,data2Variance = calculateStatistics(data2) 
data3stdDeviation,data3Variance = calculateStatistics(data3) 
print('For ', data1) 
print('Standard Deviation: ',data1stdDeviation) 
print('Variance: ',data1Variance) 
print('For ', data2) 
print('Standard Deviation: ',data2stdDeviation) 
print('Variance: ',data2Variance) 
print('For ', data3) 
print('Standard Deviation: ',data3stdDeviation) 
print('Variance: ',data3Variance) 
Output: 
For  [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2] 
Standard Deviation:  1.3868429375143148 
Variance:  1.9233333333333333 
For  [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2000] 
Standard Deviation:  399.4857235663539 
Variance:  159588.84333333332 
For  [1, 2, 3, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000] 
Standard Deviation:  974.1406935905567 
Variance:  948950.0909090909 
