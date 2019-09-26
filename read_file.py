file = open('mammographic_masses.data')
cool_file = open("cool_mammographic_masses.data", 'w')

for line in file:
    vals = line.rsplit(',')
    if '?' in vals:
        pass 
    else:
        cool_file.write(line)
        
    