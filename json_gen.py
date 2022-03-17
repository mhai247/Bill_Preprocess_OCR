import glob
import csv
import json

datasets = ('20211015', '20211019', '20211020', '20211022/bi', '20211022/Phuong', '20211022/vinh', '20211028', '20211101')
arr = []
csv_file = open('drug_name.csv', 'w')
drug_names = []
writer = csv.writer(csv_file)
count = 0
for dataset in datasets:
    files = glob.glob('output/rule/' + dataset + '/csv/*.csv')
    for file in files:
        fstream = open(file, encoding='utf-8')
        reader = csv.reader(fstream)
        dict = {'id': file.split('/')[-1].split('.')[0], 'drugname': [], 'usage' : []}
        add = 1
        
        name = ''
        # print(name)
        for row in reader:
            if row[-1] == 'drug_name':
                # if name:
                #     add = 0
                #     break
                name += row[-2]
            if row[-1] == 'type':
                type = row[-2]
            if row[-1] == 'usage':
                usage = row[-2]
                if type.lower() != 'viên':
                    add = 0
                    break
                
                if name == "Fasthan 20mg (Pravastatin natri 20mg)1/2 viên 20hLimoren - 75mg + 100mg (Clopidogrel + Aspirin)":
                    print(file)
                dict['drugname'].append(name)
                dict['usage'].append(usage)
                name = ''
                type = ''
                usage = ''
        if add:
            count += 1
            arr.append(dict)
            for name in dict['drugname']:
                if name not in drug_names:
                    drug_names.append(name)
print(count)
drug_names = sorted(drug_names)
# print(drug_names)
for name in drug_names:
    writer.writerow((name,))
json_object = json.dumps(arr, indent=4, ensure_ascii=False)
# print(json_object)
with open("drug.json", "w", encoding='utf-8') as outfile:
    outfile.write(json_object)


