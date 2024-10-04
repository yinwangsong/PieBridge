import csv

pths = [
    './res/train_log/resnet/Caltech101/',
    './res/train_log/resnet/Caltech256/',
    './res/train_log/resnet/Dogsvscats/',
    './res/train_log/resnet/DTD/',
]

baselines = [
    "ours",
    "PET",
    "FTAll",
]

data = [
    ['', 'Ours', 'PET', 'FTAll'],
    ['Time', 1, 1, 1],
    ['Acc.', 2, 2, 2],
]



for pth in pths:
    time_all = []
    acc_all = []

    for mode in baselines:
        times = []
        accs = []
        with open(pth+mode+'/log.txt', mode='r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                items = line.split()
                print(items)
                times.append(round(float(items[1])/60/60,2))
                accs.append(round(float(items[0])*100,2))
        time_all.append(times[-1])
        acc_all.append(max(accs))

    with open(pth+'results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for i in range(len(data)):
            for j in range(len(data[i])):
                if j==0:
                    continue
                if i == 1:
                    data[i][j] = time_all[j-1]
                if i == 2:
                    data[i][j] = acc_all[j-1]
        for row in data:
            writer.writerow(row)

    print("CSV files created.")