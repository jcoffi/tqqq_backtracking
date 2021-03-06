import csv
import numpy as np
import matplotlib.pyplot as plt


def get_csv_chart (file_name):
    datas = []
    with open(file_name, newline='', encoding='UTF8' ) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader: #종가
            datas.append(row[1]) 
    datas = datas[1:]
    for i in range(len(datas)): datas[i] = float(datas[i])
    return np.array(datas)[::-1]


def get_MA(data, size):
    s = data.shape
    dataMA = np.zeros(s[0])
    for i in range(size):
        dataMA[size:] += data[i:s[0]-size+i]
    dataMA[:size] = dataMA[size]
    dataMA /= size
    return dataMA

def process_backtracking (x1,x2,ma_size):
    value = 1
    values = np.ones(len(x1))
    
    s_point = ma_size
    for i in range(ma_size,len(x1),1):
        if (x1[i] > x2[i]) & (x1[i-1] < x2[i-1]): s_point = i
        if (x1[i] < x2[i]) & (x1[i-1] > x2[i-1]):
            value *= x1[i]/x1[s_point]
        values[i] = value*x1[ma_size]
        
    value *= (x1[len(x1)-1]/x1[s_point]-1)*1+1
    week_benifit = value **(1/(len(x1)-40))
    year_benifit = value **(1/((len(x1)-40)/52))
    values[s_point:] = value*x1[ma_size]
    print(round(value,2), year_benifit)
    return values,value,week_benifit
    
def scatter_over_under(x,y1,y2,size=10):
    if size == -1 : size = y1
    else: size = np.ones(y1.shape[0])*size
    area1 = np.ma.masked_where(y1 >= y2, size)
    area2 = np.ma.masked_where(y1 <= y2, size)
    plt.scatter(x, y1, s=area1, marker='^')
    plt.scatter(x, y1, s=area2, marker='o')

def get_avg_value(value,size,benifit):
    avg_values = np.arange(len(value))
    avg_values[(avg_values-size)<=0] = 0
    avg_values = (benifit ** avg_values) * value[size]*1
    return avg_values

size = 45
y_values_00 = get_csv_chart('data.csv')
#y_values_00 = y_values_00[:280]
y_values_45 = get_MA(y_values_00, size)
x_values = np.arange(y_values_00.shape[0])

y_benifit,dump,week_benifit = process_backtracking(
    y_values_00,y_values_45,size)
print(week_benifit)
avg_values = get_avg_value(y_values_00,size,week_benifit+0.0003)


#scatter_over_under(x_values,y_values_00,y_values_40,-1)
plt.plot(x_values,y_values_00/avg_values,label='MA45 value/avg')
#plt.plot(x_values,avg_values,label='MA40')
#plt.plot(x_values,y_benifit,label='benifit')
plt.legend()
plt.show()
